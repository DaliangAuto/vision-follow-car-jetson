"""
训练数据采集：相机 + 串口控制数据时间对齐，保存 frames/*.jpg 与 controls.csv。
三状态：空闲 → 采集 → 停止 → 空闲。支持模式轮询（get_mode）或按首帧触发。

Author: DaliangAuto
"""
import os
import sys

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

import cv2
import csv
import json
import time
import yaml
import signal
from datetime import datetime
from typing import Optional

from src.camera import Camera
from src.uart_control import UartControlReader


def now_str():
    """生成 run 目录时间戳，如 20250310_143025"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: str):
    """递归创建目录，已存在不报错。"""
    os.makedirs(p, exist_ok=True)


class Recorder:
    """三状态：空闲 → 采集 → 停止 → 空闲。相机常开，仅采集时保存。"""

    def __init__(
        self,
        cfg: dict,
        camera: Optional[Camera] = None,
        uart: Optional[UartControlReader] = None,
    ):
        self.cfg = cfg
        self.stop = False
        self._owns_camera = camera is None
        self._owns_uart = uart is None  # 是否由本类负责 open/close

        self.out_root = cfg["output"]["root"]
        self.target_fps = int(cfg["record"]["fps"])
        self.jpg_quality = int(cfg["record"].get("jpg_quality", 90))
        self.idle_timeout_sec = float(cfg["record"].get("idle_timeout_sec", 3.0))
        self.ctrl_skip_threshold = float(cfg["record"].get("ctrl_skip_threshold", 0.02))  # 控制变化阈值，小于此不写
        self.force_save_sec = float(cfg["record"].get("force_save_sec", 1.0))  # 至少每隔此秒写一帧

        if camera is not None:
            self.cam = camera
        else:
            cam_cfg = cfg["camera"]
            self.cam = Camera(
                sensor_id=int(cam_cfg.get("sensor_id", 0)),
                width=int(cam_cfg["width"]),
                height=int(cam_cfg["height"]),
                fps=int(cam_cfg["fps"]),
                flip_method=int(cam_cfg.get("flip_method", 0)),
                capture_width=int(cam_cfg.get("capture_width", 1920)),
                capture_height=int(cam_cfg.get("capture_height", 1080)),
            )

        if uart is not None:
            self.uart = uart
        else:
            uart_cfg = cfg["uart"]
            self.uart = UartControlReader(
                port=str(uart_cfg["port"]),
                baudrate=int(uart_cfg.get("baudrate", 115200)),
                timeout=float(uart_cfg.get("timeout", 0.05)),
                max_steer_deg=float(uart_cfg.get("max_steer_deg", 37.0)),
            )

    def _write_meta(self, run_dir: str, run_name: str):
        """写入 meta.json，记录 run 名、创建时间、配置快照。"""
        meta = {
            "run_name": run_name,
            "created_at": datetime.now().isoformat(),
            "config": self.cfg,
        }
        with open(os.path.join(run_dir, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def run(self):
        fieldnames = [
            "frame_idx", "frame_ts", "image_path", "ts", "steer", "throttle", "brake",
            "gear", "speed", "seq", "ts_ms", "raw",
        ]

        if self._owns_camera:
            print("[REC] Opening camera...")
            self.cam.open()
        if self._owns_uart:
            print("[REC] Opening UART...")
            self.uart.open()

        period = 1.0 / max(1, self.target_fps)  # 采样周期，如 10fps → 0.1s
        next_t = time.time()

        while not self.stop:
            # 有 get_mode 时按模式等待，否则等首帧
            if hasattr(self.uart, "get_mode"):
                while not self.stop and self.uart.get_mode() != "training":
                    # 非训练模式时持续读帧消耗相机 buffer，避免堆积
                    ok, _ = self.cam.read()
                    if not ok:
                        time.sleep(0.01)
                    time.sleep(0.02)
                if self.stop:
                    break
            else:
                print("[REC] 空闲，等待 MCU 数据（相机常开）...")
                self.uart.reset_for_new_session()
                while not self.stop and not self.uart.has_received_frame():
                    ok, _ = self.cam.read()
                    if not ok:
                        time.sleep(0.01)
                        continue
                    time.sleep(0.001)
            if self.stop:
                break

            # 进入采集：创建 run 目录
            run_name = f"run_{now_str()}"
            run_dir = os.path.join(self.out_root, run_name)
            frames_dir = os.path.join(run_dir, "frames")
            ensure_dir(run_dir)
            ensure_dir(frames_dir)
            self._write_meta(run_dir, run_name)

            controls_path = os.path.join(run_dir, "controls.csv")
            with open(controls_path, "w", newline="", encoding="utf-8") as fcsv:
                writer = csv.DictWriter(
                    fcsv, fieldnames=fieldnames,
                    quoting=csv.QUOTE_MINIMAL, escapechar="\\",
                )
                writer.writeheader()

                next_t = time.time()
                print(f"[REC] 采集中: {run_dir}")
                frame_idx = 0
                recording_stop = False
                last_saved_steer = None
                last_saved_throttle = None
                last_saved_brake = None
                last_force_save_t = time.time()

                while not self.stop and not recording_stop:
                    # 模式切换则立即停止采集
                    if hasattr(self.uart, "get_mode") and self.uart.get_mode() != "training":
                        new_mode = self.uart.get_mode()
                        print(f"[REC] 模式已切换 → {new_mode}，停止采集")
                        break

                    now = time.time()
                    if now < next_t:
                        time.sleep(min(0.002, next_t - now))
                        continue
                    next_t += period  # 按 target_fps 定时

                    # 串口超时：MCU 可能停止发送，结束本次 run
                    if self.uart.seconds_since_last_frame() > self.idle_timeout_sec:
                        print(f"[REC] {self.idle_timeout_sec}s 无串口数据，停止本次采集")
                        recording_stop = True
                        break

                    ok, frame = self.cam.read()
                    frame_ts = time.time()
                    if not ok or frame is None:
                        continue

                    ctrl = self.uart.get_at_time(frame_ts)  # 时间插值取控制量
                    # 控制去重：变化小于阈值且未到 force_save_sec 则跳过
                    if last_saved_steer is not None:
                        ds = abs(ctrl.steer - last_saved_steer)
                        dt = abs(ctrl.throttle - last_saved_throttle)
                        db = abs(ctrl.brake - last_saved_brake)
                        ctrl_unchanged = (
                            ds < self.ctrl_skip_threshold
                            and dt < self.ctrl_skip_threshold
                            and db < self.ctrl_skip_threshold
                        )
                        if ctrl_unchanged and (frame_ts - last_force_save_t) < self.force_save_sec:
                            continue

                    fname = f"{frame_idx:06d}.jpg"
                    relpath = os.path.join("frames", fname)
                    abspath = os.path.join(run_dir, relpath)

                    try:
                        ok_write = cv2.imwrite(
                            abspath, frame,
                            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpg_quality],
                        )
                        if not ok_write:
                            print(f"[REC] WARN: cv2.imwrite failed for {relpath}")
                            continue
                    except Exception as e:
                        print(f"[REC] ERROR: cv2.imwrite failed: {e}")
                        continue

                    row = UartControlReader.control_to_row(frame_ts, ctrl, frame_idx, relpath)
                    # raw 字段可能含换行，需转义
                    if "raw" in row and row["raw"] is not None:
                        row["raw"] = str(row["raw"]).replace("\r", "\\r").replace("\n", "\\n")
                    writer.writerow(row)
                    fcsv.flush()
                    last_saved_steer, last_saved_throttle, last_saved_brake = ctrl.steer, ctrl.throttle, ctrl.brake
                    last_force_save_t = frame_ts

                    if frame_idx % 30 == 0:
                        print(
                            f"[REC] idx={frame_idx} steer={row['steer']:.3f} thr={row['throttle']:.3f} "
                            f"brk={row['brake']:.3f} gear={row['gear']} spd={row['speed']:.1f} img={relpath}"
                        )
                    frame_idx += 1

            print(f"[REC] 本次采集结束，共 {frame_idx} 帧，回到空闲（相机保持开启）")

        if self._owns_uart:
            self.uart.close()
        if self._owns_camera:
            self.cam.close()

    def request_stop(self):
        """请求停止采集（供 SIGINT 等信号处理调用）。"""
        self.stop = True
        try:
            if self._owns_uart:
                self.uart.close()
        except Exception:
            pass
        try:
            if self._owns_camera:
                self.cam.close()
        except Exception:
            pass


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/record.yaml")
    args = ap.parse_args()

    config_path = args.config if os.path.isabs(args.config) else os.path.join(_root, args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not os.path.isabs(cfg["output"]["root"]):
        cfg["output"]["root"] = os.path.join(_root, cfg["output"]["root"])

    rec = Recorder(cfg)

    def _sigint(_sig, _frm):
        print("\n[REC] Stopping...")
        rec.request_stop()

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    rec.run()
    print("[REC] Done.")


if __name__ == "__main__":
    main()
