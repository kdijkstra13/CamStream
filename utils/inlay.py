from typing import List, Any
from utils import maintain_aspect_ratio_resize


class Inlay:
    def __init__(self, input_idx=-1, thumb_input_idx=-2, factor=4):
        self.input_idx = input_idx
        self.thumb_input_idx = thumb_input_idx
        self.factor = factor

        self.prev = None
        self.next = None

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])

        canvas = captures[self.input_idx].copy()
        if canvas is None:
            captures.append(None)
            return captures
        else:
            thumbnail = captures[self.thumb_input_idx]
            if thumbnail is not None:
                _, w, _ = canvas.shape
                w = w // self.factor
                thumbnail = maintain_aspect_ratio_resize(thumbnail, width=w)
                h, w, _ = thumbnail.shape
                canvas[0:h+2, 0:w+2, :] = 255
                canvas[0:h, 0:w, :] = thumbnail
            captures.append(canvas)
            return captures

    def start(self, block: bool = False):
        return

    def stop(self):
        return

    def wait(self, timeout=3):
        return False
