from processors import Detectron2, MMDetect
from viewers import FlaskServer, FlaskViewer
from capture import OpenCVCapture
from pipelines import LinkedListPipeline
from utils import Inlay, Buffer, SBS, Merge


def two_models_sbs():
    stream_link = 'http://10.0.0.119:81/stream'
    dtron_model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    dtron_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    mmdet_model = "~/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
    mmdet_weights = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    pipeline = LinkedListPipeline()
    pipeline.add(Buffer(OpenCVCapture(stream_link)))
    pipeline.add(Buffer(Detectron2(Detectron2.create_config(dtron_model, dtron_weights, dev="cuda:0"), input_idx=-1)))
    pipeline.add(Inlay(-1, -2, factor=4))
    pipeline.add(Buffer(MMDetect(mmdet_model, mmdet_weights, input_idx=-3, dev="cuda:1")))
    pipeline.add(Inlay(-1, -4, factor=4))
    pipeline.add(SBS(-1, -3, factor=1))
    pipeline.add(FlaskViewer(FlaskServer()))
    pipeline.start(block=True)


def two_models_three_pipelines():
    stream_link = 'http://10.0.0.119:81/stream'
    dtron_model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    dtron_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    mmdet_model = "~/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
    mmdet_weights = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"

    cap = Buffer(OpenCVCapture(stream_link))
    server = FlaskServer(name="processing_server", port=5000)
    server2 = FlaskServer(name="stream_server", port=5005)

    mmdet = LinkedListPipeline()
    mmdet.add(cap)
    mmdet.add(Buffer(MMDetect(mmdet_model, mmdet_weights, input_idx=-1, dev="cuda:1")))
    mmdet_inlay = Inlay(factor=4)
    mmdet.add(mmdet_inlay)
    mmdet.add(FlaskViewer(server, stream_url="/mmdet",  stream_name="MMDetection"))

    dtron = LinkedListPipeline()
    dtron.add(cap)
    dtron.add(Buffer(Detectron2(Detectron2.create_config(dtron_model, dtron_weights, dev="cuda:0"))))
    dtron_inlay = Inlay(factor=4)
    dtron.add(dtron_inlay)
    dtron.add(FlaskViewer(server, stream_url="/dtron", stream_name="Detectron"))

    stream = LinkedListPipeline()
    stream.add(Merge(mmdet_inlay, dtron_inlay, first_idx=-1, second_idx=-1))
    stream.add(SBS(-1, -2, factor=1))
    stream.add(FlaskViewer(server2, stream_url="/stream", stream_name="Side by Side"))

    mmdet.start(block=False)
    dtron.start(block=False)
    stream.start(block=False)
    server.start(block=False)
    server2.start(block=True)


def big_brother():
    def get_pipeline(capture_element, model_element, flask_server, url):
        p = LinkedListPipeline()
        p.add(capture_element)
        p.add(Buffer(model_element))
        inlay = Inlay(factor=4)
        p.add(inlay)
        p.add(FlaskViewer(flask_server, stream_url=url, stream_name=url))
        return p, inlay

    def create_4x4_overview(a, b, c, d, flask_server, url):
        p = LinkedListPipeline()
        p.add(Merge(a, b, first_idx=-1, second_idx=-1))
        p.add(SBS(-1, -2, factor=1))
        p.add(Merge(c, d, first_idx=-1, second_idx=-1))
        p.add(SBS(-1, -2, factor=1))
        sbs = SBS(-1, -4, factor=1, flip=True)
        p.add(sbs)
        p.add(FlaskViewer(flask_server, stream_url=url, stream_name=url))
        return p, sbs
    
    stream1_link = 'http://10.0.0.119:81/stream'
    stream2_link = 'http://10.0.0.133:81/stream'

    dtron1_model = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    dtron1_weights = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    dtron2_model = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    dtron2_weights = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

    mmdet1_model = "~/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py"
    mmdet1_weights = "https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
    mmdet2_model = "~/mmdetection/configs/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco.py"
    mmdet2_weights = "https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_x101_64x4d_fpn_1x_coco/mask_rcnn_x101_64x4d_fpn_1x_coco_20200201-9352eb0d.pth"

    cam1 = Buffer(OpenCVCapture(stream1_link))
    cam2 = Buffer(OpenCVCapture(stream2_link))
    server_cam1 = FlaskServer(name="camera1_server", port=5000)
    server_cam2 = FlaskServer(name="camera2_server", port=5005)
    server_stream = FlaskServer(name="stream_server", port=5010)

    mmdet1_cam1, mmdet1_inlay_cam1 = \
        get_pipeline(capture_element=cam1,
                     model_element=MMDetect(mmdet1_model, mmdet1_weights, input_idx=-1, dev="cuda:1"),
                     flask_server=server_cam1, url="/mmdet1")
    mmdet2_cam1, mmdet2_inlay_cam1 = \
        get_pipeline(capture_element=cam1,
                     model_element=MMDetect(mmdet2_model, mmdet2_weights, input_idx=-1, dev="cuda:1"),
                     flask_server=server_cam1, url="/mmdet2")
    dtron1_cam1, dtron1_inlay_cam1 = \
        get_pipeline(capture_element=cam1,
                     model_element=Detectron2(Detectron2.create_config(dtron1_model, dtron1_weights, dev="cuda:0")),
                     flask_server=server_cam1, url="/dtron1")
    dtron2_cam1, dtron2_inlay_cam1 = \
        get_pipeline(capture_element=cam1,
                     model_element=Detectron2(Detectron2.create_config(dtron2_model, dtron2_weights, dev="cuda:0")),
                     flask_server=server_cam1, url="/dtron2")

    cam1, cam1_grid = create_4x4_overview(mmdet1_inlay_cam1,
                                          mmdet2_inlay_cam1,
                                          dtron1_inlay_cam1,
                                          dtron2_inlay_cam1,
                                          flask_server=server_cam1,
                                          url="/stream")

    mmdet1_cam2, mmdet1_inlay_cam2 = \
        get_pipeline(capture_element=cam2,
                     model_element=MMDetect(mmdet1_model, mmdet1_weights, input_idx=-1, dev="cuda:1"),
                     flask_server=server_cam2, url="/mmdet1")
    mmdet2_cam2, mmdet2_inlay_cam2 = \
        get_pipeline(capture_element=cam2,
                     model_element=MMDetect(mmdet2_model, mmdet2_weights, input_idx=-1, dev="cuda:1"),
                     flask_server=server_cam2, url="/mmdet2")
    dtron1_cam2, dtron1_inlay_cam2 = \
        get_pipeline(capture_element=cam2,
                     model_element=Detectron2(Detectron2.create_config(dtron1_model, dtron1_weights, dev="cuda:0")),
                     flask_server=server_cam2, url="/dtron1")
    dtron2_cam2, dtron2_inlay_cam2 = \
        get_pipeline(capture_element=cam2,
                     model_element=Detectron2(Detectron2.create_config(dtron2_model, dtron2_weights, dev="cuda:0")),
                     flask_server=server_cam2, url="/dtron2")

    cam2, cam2_grid = create_4x4_overview(mmdet1_inlay_cam2,
                                          mmdet2_inlay_cam2,
                                          dtron1_inlay_cam2,
                                          dtron2_inlay_cam2,
                                          flask_server=server_cam2,
                                          url="/stream")

    p = LinkedListPipeline()
    p.add(Merge(cam1_grid, cam2_grid, first_idx=-1, second_idx=-1))
    p.add(SBS(-1, -2, factor=1))
    p.add(FlaskViewer(server_stream, stream_url="/stream", stream_name="stream"))

    mmdet1_cam1.start(block=False)
    mmdet2_cam1.start(block=False)
    dtron1_cam1.start(block=False)
    dtron2_cam1.start(block=False)
    cam1.start(block=False)
    server_cam1.start(block=False)

    mmdet1_cam2.start(block=False)
    mmdet2_cam2.start(block=False)
    dtron1_cam2.start(block=False)
    dtron2_cam2.start(block=False)
    cam2.start(block=False)
    server_cam2.start(block=False)

    p.start(block=True)


if __name__ == '__main__':
    big_brother()
