from typing import List, Any
from utils import maintain_aspect_ratio_resize
import numpy as np


class SBS:
    def __init__(self, first_idx, second_idx, factor=2, flip=False):
        self.first_index = first_idx
        self.second_index = second_idx
        self.factor = factor
        self.flip = flip
        self.prev = None
        self.next = None

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])

        first = captures[self.first_index]
        second = captures[self.second_index]

        if first is None and second is None:
            captures.append(None)
            return captures

        if first is None:
            captures.append(second)
            return captures

        if second is None:
            captures.append(first)
            return captures

        first = maintain_aspect_ratio_resize(first, width=first.shape[1] // self.factor)
        second = maintain_aspect_ratio_resize(second, width=second.shape[1] // self.factor)
        h1, w1, _ = first.shape
        h2, w2, _ = second.shape

        if not self.flip:
            h = max(h1, h2)
            w = w1 + w2 + 2
            canvas = np.zeros([h, w, 3], dtype=first.dtype)
            canvas[0:h1, 0:w1, :] = first
            canvas[0:h2, w2+2:, :] = second
        else:
            h = h1 + h2 + 2
            w = max(w1, w2)
            canvas = np.zeros([h, w, 3], dtype=first.dtype)
            canvas[0:h1, 0:w1, :] = first
            canvas[h2+2:, 0:w2:, :] = second

        captures.append(canvas)
        return captures

    def start(self, block: bool = False):
        return

    def stop(self):
        return

    def wait(self, timeout=3):
        return False
