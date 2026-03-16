"""
DlaCart 训练模式串口协议解析（TRAINING_PROTOCOL.md）
22 字节二进制帧，CRC16-MODBUS 校验。

Author: DaliangAuto
"""
import struct
import time
import threading
from collections import deque
from dataclasses import dataclass, asdict
from typing import Optional

import serial

# 协议常量（TRAINING_PROTOCOL.md）
FRAME_LEN = 22
SOF = bytes([0xAA, 0x55])
PAYLOAD_LEN = 18  # 偏移 2~19

# 转向角归一化：协议 steer_01 单位 0.1°，左负右正
DEFAULT_MAX_STEER_DEG = 37.0

# 帧类型（DOCUMENTATION.md §5.3 + MCU 扩展）
TYPE_TRAINING = 0x01       # MCU→Host 训练模式状态
TYPE_FOLLOW_READY = 0x02   # MCU→Host 跟随就绪
TYPE_AUTONOMOUS_READY = 0x03  # MCU→Host 自动驾驶就绪（预留）
TYPE_MANUAL = 0x05         # MCU→Host 手动模式（点击切换时发一帧）
TYPE_REMOTE = 0x06         # MCU→Host 遥控模式（点击切换时发一帧）


def crc16_modbus(data: bytes) -> int:
    """CRC16-MODBUS，多项式 0xA001"""
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc


@dataclass
class Control:
    """解析后的控制量，与 TRAINING_PROTOCOL 对应"""
    ts: float              # PC 端接收时间
    steer: float = 0.0     # 归一化 -1..1（左负右正）
    throttle: float = 0.0  # 0..1
    brake: float = 0.0     # 0 或 1
    gear: int = 0          # 0=P 1=R 2=N 3=D
    speed: float = 0.0     # km/h
    seq: int = 0           # 包序号
    ts_ms: int = 0         # MCU 时间戳 ms
    raw: str = ""          # 帧 hex，调试用


def parse_frame_with_type(
    buf: bytes, max_steer_deg: float = DEFAULT_MAX_STEER_DEG
) -> tuple[int, Optional[Control]]:
    """
    解析 22 字节帧，校验 CRC。
    返回 (frame_type, Control|None)。
    TYPE 0x01 返回 Control；0x02/0x03/0x05/0x06 返回 (type, None)。
    失败返回 (0, None)。
    """
    if len(buf) != FRAME_LEN or buf[:2] != SOF:
        return (0, None)
    payload = buf[2:20]
    crc_recv = struct.unpack_from("<H", buf, 20)[0]
    if crc16_modbus(payload) != crc_recv:
        return (0, None)
    frame_type = buf[3]
    if frame_type == TYPE_TRAINING:
        now = time.time()
        seq = struct.unpack_from("<H", buf, 4)[0]
        ts_ms = struct.unpack_from("<I", buf, 6)[0]
        speed_01 = struct.unpack_from("<H", buf, 10)[0]
        steer_01 = struct.unpack_from("<h", buf, 12)[0]
        thr_ref = struct.unpack_from("<H", buf, 14)[0]
        brake = buf[16]
        gear = buf[17]
        steer_norm = steer_01 / (max_steer_deg * 10.0)
        steer_norm = max(-1.0, min(1.0, steer_norm))
        throttle_norm = thr_ref / 1000.0
        throttle_norm = max(0.0, min(1.0, throttle_norm))
        c = Control(
            ts=now,
            steer=steer_norm,
            throttle=throttle_norm,
            brake=float(brake),
            gear=int(gear),
            speed=speed_01 / 10.0,
            seq=seq,
            ts_ms=ts_ms,
            raw=buf.hex(),
        )
        return (frame_type, c)
    if frame_type in (TYPE_FOLLOW_READY, TYPE_AUTONOMOUS_READY, TYPE_MANUAL, TYPE_REMOTE):
        return (frame_type, None)
    return (0, None)


def _parse_frame(buf: bytes, max_steer_deg: float = DEFAULT_MAX_STEER_DEG) -> Optional[Control]:
    """兼容旧接口：只解析 TYPE=0x01，返回 Control"""
    t, c = parse_frame_with_type(buf, max_steer_deg)
    return c if t == TYPE_TRAINING else None


def _try_extract_frame(buf: bytearray) -> Optional[bytes]:
    """
    从缓冲区尝试提取一帧。找到 SOF 且长度足够则返回帧并移除，否则返回 None。
    """
    while len(buf) >= FRAME_LEN:
        idx = buf.find(SOF)
        if idx < 0:
            del buf[: len(buf) - 1]  # 保留最后 1 字节
            return None
        if idx > 0:
            del buf[:idx]
        if len(buf) < FRAME_LEN:
            return None
        frame = bytes(buf[:FRAME_LEN])
        del buf[:FRAME_LEN]
        return frame
    return None


def _interpolate_control(c0: Control, c1: Control, t: float) -> Control:
    """在 c0 和 c1 之间按时间 t 线性插值"""
    if c0.ts >= c1.ts:
        return c0
    r = (t - c0.ts) / (c1.ts - c0.ts)
    r = max(0.0, min(1.0, r))
    return Control(
        ts=t,
        steer=c0.steer + r * (c1.steer - c0.steer),
        throttle=c0.throttle + r * (c1.throttle - c0.throttle),
        brake=c0.brake + r * (c1.brake - c0.brake),
        gear=c1.gear if r >= 0.5 else c0.gear,
        speed=c0.speed + r * (c1.speed - c0.speed),
        seq=c1.seq if r >= 0.5 else c0.seq,
        ts_ms=int(c0.ts_ms + r * (c1.ts_ms - c0.ts_ms)),
        raw=c0.raw,
    )


MODE_IDLE = "idle"
MODE_TRAINING = "training"
MODE_FOLLOW = "follow"
MODE_AUTONOMOUS = "autonomous"


class UartControlReader:
    """
    串口控制数据读取器：后台线程持续解析 22 字节帧，
    维护最新 Control、历史队列，支持 get_mode/get_at_time 等查询。
    可与 main 共享串口（external_ser）或自行打开。
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.05,
        max_steer_deg: float = DEFAULT_MAX_STEER_DEG,
        external_ser: Optional[serial.Serial] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.max_steer_deg = max_steer_deg
        self._external_ser = external_ser

        self._ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._read_buf = bytearray()  # 接收缓冲区，用于拼帧

        self._lock = threading.Lock()
        self._latest: Control = Control(ts=time.time())  # 最新一帧解析结果
        self._history: deque = deque(maxlen=500)        # (ts, Control) 历史，用于插值
        self._last_received_ts: float = 0.0
        self._last_frame_type: int = 0
        self._mode: str = MODE_IDLE
        self._mode_since: float = 0.0
        self._last_mode_frame_ts: float = 0.0
        self._idle_since: float = 0.0  # 上次收到 0x05/0x06 切到 idle 的时间
        self._IDLE_GRACE_SEC: float = 0.8  # 切到 idle 后，此时间内忽略 0x01 残留

    def open(self, external_ser: Optional[serial.Serial] = None) -> None:
        ser = external_ser or self._external_ser
        if ser is not None:
            self._ser = ser
        else:
            self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        self._stop.clear()
        self._read_buf.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def get_mode(self) -> str:
        """返回当前锁存模式，而不是最后一帧瞬时类型"""
        with self._lock:
            return self._mode

    def _loop(self) -> None:
        """后台线程：读串口 → 提取帧 → 解析 → 更新 _latest 和 _history"""
        assert self._ser is not None
        while not self._stop.is_set():
            try:
                n = self._ser.in_waiting
                if n:
                    self._read_buf.extend(self._ser.read(n))  # 有数据则批量读
                else:
                    self._read_buf.extend(self._ser.read(1))  # 无数据则阻塞读 1 字节
                while True:
                    frame = _try_extract_frame(self._read_buf)
                    if frame is None:
                        break

                    ft, c = parse_frame_with_type(frame, self.max_steer_deg)
                    if ft == 0:
                        continue

                    with self._lock:
                        now = time.time()
                        self._last_frame_type = ft

                        # 明确进入类帧：立即锁存
                        if ft == TYPE_FOLLOW_READY:
                            self._mode = MODE_FOLLOW
                            self._mode_since = now
                            self._last_mode_frame_ts = now

                        elif ft == TYPE_AUTONOMOUS_READY:
                            self._mode = MODE_AUTONOMOUS
                            self._mode_since = now
                            self._last_mode_frame_ts = now

                        elif ft == TYPE_TRAINING:
                            # 若刚切到 idle（0x05/0x06），忽略 0x01 残留，避免继续保存
                            if self._mode == MODE_IDLE and self._idle_since > 0:
                                if (now - self._idle_since) < self._IDLE_GRACE_SEC:
                                    continue  # 不更新 _mode，保持 idle
                            self._mode = MODE_TRAINING
                            self._mode_since = now
                            self._last_mode_frame_ts = now

                        elif ft in (TYPE_MANUAL, TYPE_REMOTE):
                            # 0x05/0x06：切到 idle。仅对 follow 做 0.30s 保护（防 0x02 后跟 0x05 的过渡噪点）
                            # 对 training：立即接受 0x05，否则 0x01 和 0x05 同批次到达时 0x01 会刷新
                            # _mode_since，导致 now - _mode_since < 0.30，误忽略 0x05
                            if self._mode == MODE_FOLLOW and (now - self._mode_since) <= 0.30:
                                pass  # 在 follow 内刚进入时忽略 0x05，视为过渡帧
                            else:
                                self._mode = MODE_IDLE
                                self._idle_since = now
                                self._mode_since = now
                                self._last_mode_frame_ts = now

                        if c is not None:
                            self._latest = c
                            self._history.append((c.ts, c))
                            self._last_received_ts = c.ts
            except Exception:
                time.sleep(0.02)

    def latest(self) -> Control:
        """返回最新解析到的 Control（线程安全）。"""
        with self._lock:
            return self._latest

    def seconds_since_last_frame(self) -> float:
        """距上次收到 TYPE=0x01 有效帧的秒数，用于采集超时判断。"""
        with self._lock:
            ts = self._last_received_ts
        if ts <= 0:
            return float("inf")
        return time.time() - ts

    def has_received_frame(self) -> bool:
        """是否至少收到过一帧 TYPE=0x01。"""
        return self._last_received_ts > 0

    def reset_for_new_session(self) -> None:
        """重置接收状态，用于新一次采集前清空 has_received_frame。"""
        with self._lock:
            self._last_received_ts = 0.0

    def get_at_time(self, target_ts: float) -> Control:
        """按帧时间戳取最近邻或插值后的控制量"""
        with self._lock:
            h = list(self._history)
            latest = self._latest
        if not h:
            return latest
        before = [(t, c) for t, c in h if t <= target_ts]
        after = [(t, c) for t, c in h if t > target_ts]
        if not before and not after:
            return latest
        if not before:
            return after[0][1]
        if not after:
            return before[-1][1]
        c0, c1 = before[-1][1], after[0][1]
        return _interpolate_control(c0, c1, target_ts)

    def close(self, close_serial: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        if close_serial and self._ser is not None and self._external_ser is None:
            try:
                self._ser.close()
            except Exception:
                pass
        if close_serial and self._external_ser is None:
            self._ser = None
        self._thread = None

    @staticmethod
    def control_to_row(frame_ts: float, ctrl: Control, frame_idx: int, relpath: str) -> dict:
        """将 Control 转为 controls.csv 一行的字典，附加 frame_ts/frame_idx/image_path。"""
        d = asdict(ctrl)
        d.update(
            frame_ts=frame_ts,
            frame_idx=frame_idx,
            image_path=relpath,
        )
        return d
