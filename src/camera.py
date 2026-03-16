"""
CSI 摄像头封装，使用 GStreamer + nvarguscamerasrc 采集。
适用于 Jetson 平台（IMX219 等 CSI 传感器）。

Author: DaliangAuto
"""
import time
import cv2


def gstreamer_pipeline(
    sensor_id=0,
    width=640,
    height=360,
    fps=30,
    flip_method=0,
    capture_width=1920,
    capture_height=1080,
):
    """
    构建 GStreamer 管道字符串。
    先用相机原生 16:9 输出，再在 nvvidconv 里缩放到目标尺寸，避免拉伸变形。
    drop=1 max-buffers=1 sync=false：降低延迟，丢弃旧帧。
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){fps}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){width}, height=(int){height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1 max-buffers=1 sync=false"
    )


class Camera:
    """CSI 相机封装：GStreamer 管道，OpenCV VideoCapture 读取。"""

    def __init__(
        self,
        sensor_id=0,
        width=640,
        height=360,
        fps=30,
        flip_method=0,
        capture_width=1920,
        capture_height=1080,
    ):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.flip_method = flip_method
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.cap = None  # cv2.VideoCapture，open() 时初始化

    def open(self) -> None:
        pipeline = gstreamer_pipeline(
            sensor_id=self.sensor_id,
            width=self.width,
            height=self.height,
            fps=self.fps,
            flip_method=self.flip_method,
            capture_width=self.capture_width,
            capture_height=self.capture_height,
        )

        # 最多重试 5 次，每次失败间隔 2 秒
        for i in range(5):
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            if self.cap.isOpened():
                print(f"[CAM] Camera opened on attempt {i + 1}", flush=True)
                return

            print(f"[CAM] Open failed on attempt {i + 1}, retrying...", flush=True)
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            time.sleep(2)

        raise RuntimeError(
            "Failed to open CSI camera. "
            "Check IMX219 connection, nvargus-daemon, and GStreamer pipeline."
        )

    def read(self):
        """读取一帧，返回 (ok, frame)，frame 为 BGR numpy 数组。"""
        if self.cap is None:
            raise RuntimeError("Camera not opened")
        ok, frame = self.cap.read()
        return ok, frame

    def close(self) -> None:
        """释放相机资源。"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
