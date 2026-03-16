"""
DlaCart 统一入口：根据 MCU 帧 TYPE 切换模式
- TYPE=0x01 训练模式：采集图像+控制数据
- TYPE=0x02 跟随模式：启动推理，发送控制指令给 MCU
- TYPE=0x03 自动驾驶模式：预留

Author: DaliangAuto
"""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
if _root not in sys.path:
    sys.path.insert(0, _root)

import signal
import time
import yaml

import serial

from src.camera import Camera
from src.uart_control import MODE_AUTONOMOUS, MODE_FOLLOW, MODE_TRAINING, UartControlReader


def main():
    """统一入口主函数：打开串口与相机，根据 MCU 上报的帧 TYPE 循环切换训练/跟随/自动驾驶模式。"""
    import argparse
    ap = argparse.ArgumentParser(description="DlaCart 统一入口，根据 MCU 帧 TYPE 切换模式")
    ap.add_argument("--config", default="config/autodrive.yaml", help="配置文件路径")
    args = ap.parse_args()

    # 加载配置，输出目录转为绝对路径
    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_root, config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not os.path.isabs(cfg.get("output", {}).get("root", "data/raw")):
        cfg["output"]["root"] = os.path.join(_root, cfg["output"]["root"])

    stop = False

    def on_sig(sig, frm):
        nonlocal stop
        print("\n[MAIN] Stopping...")
        stop = True

    signal.signal(signal.SIGINT, on_sig)
    signal.signal(signal.SIGTERM, on_sig)

    # 共享串口（读写复用）
    uart_cfg = cfg["uart"]
    ser = serial.Serial(
        port=str(uart_cfg["port"]),
        baudrate=int(uart_cfg.get("baudrate", 115200)),
        timeout=float(uart_cfg.get("timeout", 0.05)),
    )
    uart_reader = UartControlReader(
        port=str(uart_cfg["port"]),
        baudrate=int(uart_cfg.get("baudrate", 115200)),
        timeout=float(uart_cfg.get("timeout", 0.05)),
        max_steer_deg=float(uart_cfg.get("max_steer_deg", 37.0)),
        external_ser=ser,
    )
    uart_reader.open()

    # 共享相机
    cam_cfg = cfg["camera"]
    cam = Camera(
        sensor_id=int(cam_cfg.get("sensor_id", 0)),
        width=int(cam_cfg["width"]),
        height=int(cam_cfg["height"]),
        fps=int(cam_cfg.get("fps", 30)),
        flip_method=int(cam_cfg.get("flip_method", 0)),
        capture_width=int(cam_cfg.get("capture_width", 1920)),
        capture_height=int(cam_cfg.get("capture_height", 1080)),
    )
    cam.open()

    print("[MAIN] 后台运行，等待 MCU 模式切换...")
    print("[MAIN] TYPE=0x01 训练 | 0x02 跟随 | 0x03 自动驾驶(预留) | 0x05 手动 | 0x06 遥控")

    try:
        # 主循环：轮询模式，根据 TYPE 进入对应分支
        while not stop:
            mode = uart_reader.get_mode()

            if mode == MODE_TRAINING:
                from src.record_data import Recorder

                print("[MAIN] 进入训练模式，开始采集...")
                rec = Recorder(cfg, camera=cam, uart=uart_reader)
                rec.run()
                if stop:
                    break
                print("[MAIN] 采集结束，回到等待...")

            elif mode == MODE_FOLLOW:
                from src.inference import CommandSender, RealTimeInfer

                print("[MAIN] 检测到 TYPE=0x02 跟随就绪，进入跟随模式...")
                # 跟随模式：CommandSender 复用 main 打开的串口
                sender = CommandSender(
                    port=str(uart_cfg["port"]),
                    baudrate=int(uart_cfg.get("baudrate", 115200)),
                    timeout=float(uart_cfg.get("timeout", 0.02)),
                    external_ser=ser,
                )
                sender.open()
                ckpt = cfg["infer"].get("ckpt", "models/best_model.pth")
                print(f"[MAIN] 加载模型 {ckpt}，启动推理...")
                infer = RealTimeInfer(cfg, camera=cam, sender=sender)
                print("[MAIN] 跟随模式已就绪，正在推理并发送 TYPE=0x10 控制帧给 MCU")

                def still_follow():
                    return uart_reader.get_mode() == MODE_FOLLOW

                infer.run(mode_check=still_follow)  # mode_check 用于检测串口屏切换模式
                sender.close(close_serial=False)   # 串口由 main 管理，不关闭
                if stop:
                    break
                print("[MAIN] 跟随模式已退出，回到等待...")

            elif mode == MODE_AUTONOMOUS:
                print("[MAIN] 自动驾驶模式(预留)，暂不处理")
                time.sleep(1.0)

            else:
                # idle 或未收到帧，短暂休眠后继续轮询
                time.sleep(0.1)

    finally:
        # 清理：uart_reader 不关串口，由 main 统一关闭
        uart_reader.close(close_serial=False)
        try:
            ser.close()
        except Exception:
            pass
        try:
            cam.close()
        except Exception:
            pass
        print("[MAIN] Done.")


if __name__ == "__main__":
    main()
