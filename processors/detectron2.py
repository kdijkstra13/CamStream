from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from typing import List, Any
import time


class Detectron2:

    def __init__(self, config, input_idx=-1):
        self.config = config
        self.predictor = self.create_predictor(self.config)
        self.input_idx = input_idx

        self.prev = None
        self.next = None

    def process_image(self, image):
        outputs = self.predictor(image)
        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.config.DATASETS.TRAIN[0]), scale=1)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        return out.get_image()[:, :, ::-1]

    @staticmethod
    def create_config(path="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                      checkpoint="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
                      threshold=0.5, dev="cuda"):
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(path))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(checkpoint)
        cfg.MODEL.DEVICE = dev
        return cfg

    @staticmethod
    def create_predictor(config):
        predictor = DefaultPredictor(config)
        return predictor

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])
        input_image = captures[self.input_idx]
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
