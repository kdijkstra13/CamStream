from typing import Union
import cv2
from utils import maintain_aspect_ratio_resize
from typing import List, Any


class OpenCVCapture:
    def __init__(self, src: Union[int, str] = 0):
        self.src = src
        self.capture = cv2.VideoCapture(src)
        self.prev = None
        self.next = None

    def __del__(self):
        self.capture.release()

    def __call__(self, captures: List[Any]) -> List[Any]:
        frame = self.get_frame()
        captures.append(frame)
        return captures

    def get_frame(self):
        if not self.capture.isOpened():
            return None

        success = False
        frame = None

        while not success:
            success, frame = self.capture.read()
            if success:
                frame = maintain_aspect_ratio_resize(frame, width=800)
            else:
                print("Re-initialize video capture device.")
                del self.capture
                self.capture = cv2.VideoCapture(self.src)
        return frame

    def start(self, block: bool = False):
        return

    def stop(self):
        return

    def wait(self):
        return False
