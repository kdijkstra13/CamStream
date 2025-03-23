from typing import Any, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from processors.llava_infer import LavaInfer
from utils import text_box, ALIGNMENT_LEFT, ALIGNMENT_TOP


class Llava:

    def __init__(self, llava_type='small', ooi=None, input_idx=-1):

        # Initialize Llava model
        self.lava_infer =  LavaInfer(llava_type)
        self.counter = 0

        self.input_idx = input_idx
        self.prev = None
        self.next = None
        self.ooi = ooi

    def draw_text(self, image, text):
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(35)
        text_box(text, draw, font,
                 (20, 20, image.size[0] -  20,  image.size[1]),
                 ALIGNMENT_LEFT,
                 ALIGNMENT_TOP,
                 fill=(255, 255, 255)
                 )
        return image

    def get_answer(self, image, prompt):
        img = Image.fromarray(image)
        answer = self.lava_infer.infer(prompt, [img])
        return answer

    def overlay_answer(self, image, text):
        img = Image.fromarray(image // 2)
        self.draw_text(img, text)
        return np.array(img)

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])
        input_image = captures[self.input_idx].copy()
        if input_image is not None:
            if self.ooi is None:
                prompt = "What is the object on the floor, and where is it in the image, left, center or right?"
            else:
                prompt = f"Where is the {self.ooi} in the image, left, center or right?"
            answer = self.get_answer(input_image, prompt=prompt)
            captures.append(self.overlay_answer(input_image, answer))
            captures.append(f"{self.counter}: {answer}")
            self.counter += 1 # make every answer unique
        else:
            captures.append(None)
            captures.append("")
        return captures

    def start(self, block: bool = False):
        return

    def stop(self):
        return

    def wait(self, timeout=3):
        return False
