"""
跟随模式实时推理：相机采集 → 5 帧窗口 → AutofollowModel → 后处理 → 串口下发。
与 TRAINING_README 训练架构对齐：4 输出 [steer, throttle, brake_logit, target_valid_logit]。
三线程：camera_loop（采集+预处理）、infer_loop（推理）、send_loop（发送）。
跟随模式 MCU 不发遥测，speed 填 0。

Author: DaliangAuto
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import math
import re
import time
import signal
import threading
import struct
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import torch
import yaml
import serial
from PIL import Image
from torchvision import transforms

from models import AutofollowModel
from src.camera import Camera
from src.uart_control import crc16_modbus
from utils import IMAGENET_MEAN, IMAGENET_STD

FRAME_LEN = 22           # 22 字节协议帧
SOF = bytes([0xAA, 0x55])
TYPE_FOLLOW_CMD = 0x10   # Host→MCU 控制指令
MAX_STEER_DEG = 37.0     # 转向角 ±37° 对应协议单位
SEQ_LEN = 5              # 模型输入 5 帧


def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _resolve_ckpt_path(ckpt_path: Path) -> Path:
    """
    解析 checkpoint 路径，支持 TRAINING_README 约定：
    - 直接文件：models/best_model.pth、checkpoints/20260313_1223_55/best_model.pth
    - checkpoints 或 checkpoints/latest：自动取最新 run 的 best_model.pth
    """
    if ckpt_path.suffix and ckpt_path.exists():
        return ckpt_path
    base = ckpt_path if ckpt_path.is_dir() else ckpt_path.parent
    if base.name == "latest":
        base = base.parent
    if not base.exists():
        return ckpt_path
    # 扫描 checkpoints 下 YYYYMMDD_HHMM_SS 子目录，取最新
    run_pattern = re.compile(r"^\d{8}_\d{4}_\d{2}$")
    runs = []
    for d in base.iterdir():
        if d.is_dir() and run_pattern.match(d.name):
            best = d / "best_model.pth"
            if best.exists():
                runs.append((d.name, best))
    if runs:
        runs.sort(key=lambda x: x[0], reverse=True)
        return runs[0][1]
    return ckpt_path


@dataclass
class PredCommand:
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    brake_prob: float = 0.0
    target_valid_prob: float = 0.0
    ts: float = 0.0


class CommandSender:
    """Jetson -> MCU 串口发送器（跟随模式 Host→MCU, TYPE=0x10）"""

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.02,
        external_ser: Optional[serial.Serial] = None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self._external_ser = external_ser
        self.ser: Optional[serial.Serial] = None
        self._seq = 0

    def open(self, external_ser: Optional[serial.Serial] = None):
        """打开串口或使用外部传入的 serial 对象（main 共享）。"""
        ser = external_ser or self._external_ser
        if ser is not None:
            self.ser = ser
        else:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

    def close(self, close_serial: bool = True):
        if close_serial and self.ser is not None and self._external_ser is None:
            try:
                self.ser.close()
            except Exception:
                pass
        if close_serial and self._external_ser is None:
            self.ser = None

    def pack_command(self, steer: float, throttle: float, brake: float) -> bytes:
        """将归一化 steer/throttle/brake 转为 22 字节协议帧（反归一化）。brake 已为 0/1。"""
        ts_ms = int(time.time() * 1000) & 0xFFFFFFFF
        self._seq = (self._seq + 1) & 0xFFFF
        seq = self._seq
        steer_01 = int(max(-1.0, min(1.0, steer)) * MAX_STEER_DEG * 10)  # 0.1° 单位
        thr_ref = int(max(0.0, min(1.0, throttle)) * 1000)               # 0~1000
        brake_u8 = 1 if brake > 0.5 else 0                              # 0/1
        speed_01 = 0
        gear = 3
        flags, res = 0, 0
        payload = struct.pack(
            "<BBHIHhHBBBB",
            0x01, TYPE_FOLLOW_CMD, seq, ts_ms, speed_01, steer_01,
            thr_ref, brake_u8, gear, flags, res,
        )
        crc = crc16_modbus(payload)
        return SOF + payload + struct.pack("<H", crc)

    def send(self, steer: float, throttle: float, brake: float):
        """打包并发送一帧控制指令到 MCU。"""
        if self.ser is None:
            return
        self.ser.write(self.pack_command(steer, throttle, brake))
        self.ser.flush()


class RealTimeInfer:
    """
    实时推理引擎：三线程协作。
    相机线程填充 frame_buffer/speed_buffer，推理线程周期性 forward，
    发送线程周期性将 latest_pred 下发。
    模型输出 4 维：pred[0]=steer, pred[1]=throttle, pred[2]=brake_logit, pred[3]=target_valid_logit
    """

    def __init__(
        self,
        cfg: dict,
        camera: Optional[Camera] = None,
        sender: Optional[CommandSender] = None,
    ):
        self.cfg = cfg
        self.stop = False
        self._owns_camera = camera is None
        self._owns_sender = sender is None

        infer_cfg = cfg["infer"]
        self.device = torch.device(
            infer_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.ckpt_path = Path(infer_cfg.get("ckpt", "models/best_model.pth"))
        if not self.ckpt_path.is_absolute():
            self.ckpt_path = Path(_root) / self.ckpt_path
        self.model = self._load_model(self.ckpt_path, self.device)

        if camera is not None:
            self.cam = camera
            self.camera_fps = 30
        else:
            cam_cfg = cfg["camera"]
            self.camera_fps = int(cam_cfg.get("fps", 30))
            self.cam = Camera(
                sensor_id=int(cam_cfg.get("sensor_id", 0)),
                width=int(cam_cfg["width"]),
                height=int(cam_cfg["height"]),
                fps=self.camera_fps,
                flip_method=int(cam_cfg.get("flip_method", 0)),
                capture_width=int(cam_cfg.get("capture_width", 1920)),
                capture_height=int(cam_cfg.get("capture_height", 1080)),
            )

        if sender is not None:
            self.sender = sender
        else:
            uart_cfg = cfg["uart"]
            self.sender = CommandSender(
                port=str(uart_cfg["port"]),
                baudrate=int(uart_cfg.get("baudrate", 115200)),
                timeout=float(uart_cfg.get("timeout", 0.02)),
            )

        self.infer_hz = float(infer_cfg.get("infer_hz", 10.0))
        self.send_hz = float(infer_cfg.get("send_hz", 20.0))
        self.frame_stride = int(infer_cfg.get("frame_stride", 3))
        self.max_idle_sec = float(infer_cfg.get("max_idle_sec", 1.0))

        # 后处理参数：低通 alpha、死区、限幅、4 输出阈值
        post_cfg = cfg.get("postprocess", {})
        self.steer_alpha = float(post_cfg.get("steer_alpha", 0.3))
        self.throttle_alpha = float(post_cfg.get("throttle_alpha", 0.2))
        self.steer_deadband = float(post_cfg.get("steer_deadband", 0.03))
        self.max_abs_steer = float(post_cfg.get("max_abs_steer", 0.6))
        self.max_throttle = float(post_cfg.get("max_throttle", 0.30))
        self.target_valid_threshold = float(post_cfg.get("target_valid_threshold", 0.5))
        self.brake_on_threshold = float(post_cfg.get("brake_on_threshold", 0.65))
        self.brake_off_threshold = float(post_cfg.get("brake_off_threshold", 0.35))
        self.brake_on_frames = int(post_cfg.get("brake_on_frames", 3))
        self.brake_off_frames = int(post_cfg.get("brake_off_frames", 5))
        self.throttle_release_limit = float(post_cfg.get("throttle_release_limit", 0.15))
        self.throttle_release_frames = int(post_cfg.get("throttle_release_frames", 10))

        self.brake_state = 0
        self.brake_on_count = 0
        self.brake_off_count = 0
        self.release_limit_count = 0

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        self.frame_buffer = deque(maxlen=SEQ_LEN)
        self.speed_buffer = deque(maxlen=SEQ_LEN)
        self.raw_frame_count = 0
        self.lock = threading.Lock()
        self.latest_pred = PredCommand(
            0.0, 0.0, 0.0, brake_prob=1.0, target_valid_prob=0.0, ts=time.time()
        )
        self.prev_sent = PredCommand(
            0.0, 0.0, 0.0, brake_prob=1.0, target_valid_prob=0.0, ts=time.time()
        )
        self.last_infer_ts = 0.0
        self.last_send_ts = 0.0
        self.last_camera_ok_ts = 0.0

    def _load_model(self, ckpt_path: Path, device: torch.device):
        """加载 4 输出 checkpoint 到 AutofollowModel（TRAINING_README 训练架构）。"""
        resolved = _resolve_ckpt_path(ckpt_path)
        if not resolved.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path} (resolved: {resolved})")
        model = AutofollowModel().to(device)
        ckpt = torch.load(resolved, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        print(f"[INF] Loaded model from: {resolved}")
        print(f"[INF] Device: {device}")
        return model

    def request_stop(self):
        self.stop = True
        try:
            if self._owns_camera:
                self.cam.close()
        except Exception:
            pass
        try:
            if self._owns_sender:
                self.sender.close()
        except Exception:
            pass

    def _preprocess_frame(self, frame_bgr):
        """BGR → RGB → Resize 224×224 → ToTensor → ImageNet 归一化。"""
        img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        return self.preprocess(img)

    def _clamp(self, x, lo, hi):
        return max(lo, min(hi, x))

    def _postprocess_pred(self, pred):
        """
        4 输出后处理：target_valid 最高优先级；brake 滞回+连续帧确认；解除刹车后短暂限油。
        """
        steer_raw = self._clamp(float(pred[0]), -1.0, 1.0)
        throttle_raw = self._clamp(float(pred[1]), 0.0, 1.0)
        brake_logit = float(pred[2])
        target_valid_logit = float(pred[3])
        brake_prob = _sigmoid(brake_logit)
        target_valid_prob = _sigmoid(target_valid_logit)

        with self.lock:
            prev = self.latest_pred

        # steer 低通、死区、限幅
        steer = (1.0 - self.steer_alpha) * prev.steer + self.steer_alpha * steer_raw
        if abs(steer) < self.steer_deadband:
            steer = 0.0
        steer = self._clamp(steer, -self.max_abs_steer, self.max_abs_steer)

        # throttle 低通、限幅
        throttle = (1.0 - self.throttle_alpha) * prev.throttle + self.throttle_alpha * throttle_raw
        throttle = self._clamp(throttle, 0.0, self.max_throttle)

        # brake 与 throttle 控制逻辑
        if target_valid_prob < self.target_valid_threshold:
            throttle = 0.0
            self.brake_state = 1
            self.brake_on_count = 0
            self.brake_off_count = 0
        else:
            if self.brake_state == 0:
                if brake_prob >= self.brake_on_threshold:
                    self.brake_on_count += 1
                else:
                    self.brake_on_count = 0
                if self.brake_on_count >= self.brake_on_frames:
                    self.brake_state = 1
                    self.brake_on_count = 0
                    self.brake_off_count = 0
            else:
                if brake_prob <= self.brake_off_threshold:
                    self.brake_off_count += 1
                else:
                    self.brake_off_count = 0
                if self.brake_off_count >= self.brake_off_frames:
                    self.brake_state = 0
                    self.brake_on_count = 0
                    self.brake_off_count = 0
                    self.release_limit_count = self.throttle_release_frames

        brake = float(self.brake_state)
        if brake > 0.5:
            throttle = 0.0
        elif self.release_limit_count > 0:
            throttle = min(throttle, self.throttle_release_limit)
            self.release_limit_count -= 1

        return PredCommand(
            steer=steer,
            throttle=throttle,
            brake=brake,
            brake_prob=brake_prob,
            target_valid_prob=target_valid_prob,
            ts=time.time(),
        )

    def _run_model(self):
        """5 帧满则 forward，输出 [1,4]：steer, throttle, brake_logit, target_valid_logit。"""
        if len(self.frame_buffer) < SEQ_LEN or len(self.speed_buffer) < SEQ_LEN:
            return None
        images = torch.stack(list(self.frame_buffer), dim=0).unsqueeze(0).to(self.device)
        speeds = torch.tensor(list(self.speed_buffer), dtype=torch.float32).view(1, SEQ_LEN, 1).to(self.device)
        with torch.no_grad():
            out = self.model(images, speeds)
            pred = out[0].detach().cpu().numpy()
        return pred

    def camera_loop(self):
        """相机线程：读帧，每 frame_stride 帧取 1，预处理后入 buffer，speed 填 0。"""
        while not self.stop:
            ok, frame = self.cam.read()
            now = time.time()
            if not ok or frame is None:
                time.sleep(0.002)
                continue
            self.last_camera_ok_ts = now
            self.raw_frame_count += 1
            if (self.raw_frame_count - 1) % self.frame_stride != 0:
                continue
            self.frame_buffer.append(self._preprocess_frame(frame))
            self.speed_buffer.append(0.0)

    def infer_loop(self):
        """推理线程：按 infer_hz 定时，相机超时则输出安全命令，否则 forward 并更新 latest_pred。"""
        period = 1.0 / max(self.infer_hz, 1e-6)
        next_t = time.time()
        while not self.stop:
            now = time.time()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue
            next_t += period
            # 相机超时：安全输出 steer=0 throttle=0 brake=1，同步刹车状态
            if self.last_camera_ok_ts > 0 and (now - self.last_camera_ok_ts) > self.max_idle_sec:
                self.brake_state = 1
                self.brake_on_count = 0
                self.brake_off_count = 0
                with self.lock:
                    self.latest_pred = PredCommand(
                        0.0, 0.0, 1.0,
                        brake_prob=1.0,
                        target_valid_prob=0.0,
                        ts=time.time(),
                    )
                continue
            pred = self._run_model()
            if pred is None:
                continue
            cmd = self._postprocess_pred(pred)
            with self.lock:
                self.latest_pred = cmd
            print(
                f"[INF] steer={cmd.steer:+.3f} thr={cmd.throttle:.3f} brk={cmd.brake:.1f} "
                f"brk_p={cmd.brake_prob:.2f} tv_p={cmd.target_valid_prob:.2f}"
            )

    def send_loop(self):
        """发送线程：按 send_hz 定时，将 latest_pred 的 steer/throttle/brake 下发到 MCU。"""
        period = 1.0 / max(self.send_hz, 1e-6)
        next_t = time.time()
        while not self.stop:
            now = time.time()
            if now < next_t:
                time.sleep(min(0.002, next_t - now))
                continue
            next_t += period
            with self.lock:
                cmd = self.latest_pred
            self.sender.send(cmd.steer, cmd.throttle, cmd.brake)

    def run(self, mode_check=None):
        """
        启动三线程并阻塞。mode_check 为可调用对象，返回 False 时退出（用于 main 检测模式切换）。
        """
        if self._owns_camera:
            self.cam.open()
        if self._owns_sender:
            self.sender.open()
        print("[INF] 相机/推理/发送三线程已启动，推理 {}Hz 发送 {}Hz".format(
            self.infer_hz, self.send_hz))
        t_cam = threading.Thread(target=self.camera_loop, daemon=True)
        t_inf = threading.Thread(target=self.infer_loop, daemon=True)
        t_send = threading.Thread(target=self.send_loop, daemon=True)
        t_cam.start()
        t_inf.start()
        t_send.start()
        try:
            while not self.stop:
                if mode_check is not None and not mode_check():
                    break
                time.sleep(0.2)
        finally:
            self.request_stop()
            t_cam.join(timeout=1.0)
            t_inf.join(timeout=1.0)
            t_send.join(timeout=1.0)
            print("[INF] 推理已停止")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/infer.yaml")
    args = parser.parse_args()
    config_path = args.config if os.path.isabs(args.config) else os.path.join(_root, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    runner = RealTimeInfer(cfg)

    def _sigint(_sig, _frm):
        print("\n[INF] Stopping...")
        runner.request_stop()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    runner.run()


if __name__ == "__main__":
    main()
