from threading import Thread
import cv2
import numpy as np
from utils import maintain_aspect_ratio_resize


class ImageCapture:
    def __init__(self, src: str = "image.jpg", detector=None):
        self.filename = src
        self.detector = detector

        self.status = None
        self.frame = None
        self.has_window = False

        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while True:
            self.frame = cv2.imread(self.filename)

    def get_frame(self):
        if self.frame is not None:
            frame = maintain_aspect_ratio_resize(self.frame, width=800)
            image_thumb = maintain_aspect_ratio_resize(frame, width=200)

            if self.detector and self.detector.image_to_process is None:
                self.detector.image_to_process = frame.copy()

            h, w, _ = image_thumb.shape
            if self.detector.processed_image is not None:
                canvas = self.detector.processed_image.copy()
            else:
                canvas = np.zeros_like(frame)

            canvas[0:h+3, 0:w+3, :] = 255
            canvas[0:h, 0:w, :] = image_thumb

            return canvas
        return None

