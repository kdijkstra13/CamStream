import mmdet
from mmdet.apis import init_detector, inference_detector
import mmcv
from threading import Lock
from typing import Any, List
import os
import requests
import shutil
import time


class Zoo:

    def __init__(self, path=os.path.join("~", "zoo")):
        self.path = path

    def __call__(self, url: str) -> str:
        fn = os.path.expanduser(os.path.join(self.path, os.path.basename(url)))
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        if url[:4] == "http":
            if not os.path.isfile(fn):
                print(f"Download {url} to {fn}")
                r = requests.get(url, allow_redirects=True)
                open(fn, 'wb').write(r.content)
        elif os.path.isfile(url):
            if not os.path.isfile(fn):
                print(f"Copy {url} to {fn}")
                shutil.copyfile(url, fn)
        else:
            raise RuntimeError(f"Cannot locate url or file {url}")
        return fn


# Global lock for matplotlib
lock = Lock()


class MMDetect:

    def __init__(self,
                 config_file='~/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py',
                 checkpoint_file='https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth',
                 input_idx=-1,
                 dev="cuda"):

        checkpoint_file = Zoo()(checkpoint_file)

        self.predictor = init_detector(config_file, checkpoint_file, device=dev)
        self.input_idx = input_idx

        self.prev = None
        self.next = None

    def process_image(self, image):
        result = inference_detector(self.predictor, image)
        with lock:
            out = self.predictor.show_result(image, result)
        return out

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])
        input_image = captures[self.input_idx].copy()
        if input_image is not None:
            output_image = self.process_image(input_image)
            captures.append(output_image)
        else:
            captures.append(None)
        return captures

    def start(self, block: bool = False):
        return

    def stop(self):
        return

    def wait(self, timeout=3):
        return False
