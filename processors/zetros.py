from typing import Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import urllib.request

from utils import text_box, ALIGNMENT_LEFT, ALIGNMENT_TOP, ALIGNMENT_CENTER, Buffer


class LegoZetros:

    def __init__(self, host, image_idx=-1, text_idx=-2, dummy=False, only_when_updated=True):
        self.host = host
        self.image_idx = image_idx
        self.text_idx = text_idx
        self.dummy = dummy
        self.only_when_updated = only_when_updated  # only send command if all buffers have updates
        self.prev = None
        self.next = None
        self.last_input_text = ""

    def draw_text(self, image, text):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(40)
        text_box(text, draw, font,
                 (20, 20, image.size[0] -  20,  image.size[1]),
                 ALIGNMENT_CENTER,
                 ALIGNMENT_CENTER,
                 fill=(255, 255, 255)
                 )
        return image

    def parse_direction(self, command):
        if "left" in command:
            return "left"
        elif "right" in command:
            return "right"
        elif "center" in command:
            return "forward"
        elif "backward" in command:
            return "backward"
        else:
            return ""

    def overlay_answer(self, image, text):
        img = Image.fromarray(image)
        self.draw_text(img, text)
        return np.array(img)

    def should_update(self, input_text):
        if not self.only_when_updated :
            return True
        return self.last_input_text != input_text

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])
        input_image = captures[self.image_idx].copy()
        input_text = captures[self.text_idx]
        if input_text is not None and input_image is not None:
            direction = self.parse_direction(input_text)
            overlay_image = self.overlay_answer(input_image, direction)
            if not self.dummy:
                if self.should_update(input_text): # do not actually move if the buffer is not updated
                    urllib.request.urlopen(self.host + "/" + direction)
                    self.last_input_text = input_text
            captures.append(overlay_image)
            captures.append(direction)
        else:
            captures.append(None)
            captures.append("None")
        return captures

    def start(self, block: bool = False):
        return

    def stop(self):
        return

    def wait(self, timeout=3):
        return False
