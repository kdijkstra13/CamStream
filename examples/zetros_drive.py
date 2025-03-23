from capture import OpenCVCapture
from pipelines import LinkedListPipeline
from processors.llava import Llava

from processors.zetros import LegoZetros
from utils import Buffer, SBS
from viewers import FlaskServer, FlaskViewer

C_STREAM = "http://10.13.13.136/capture"
C_ZETROS = "http://10.13.13.108:5000"

def zetros_async():
    stream_link = C_STREAM
    pipeline = LinkedListPipeline()
    pipeline.add(Buffer(OpenCVCapture, stream_link, True))
    pipeline.add(Buffer(Llava, "small", "small toy", default_idx=[-1, "no answer"]))
    pipeline.add(Buffer(LegoZetros, C_ZETROS, -2, -1, False, True, default_idx=[-2, "stay"]))
    pipeline.add(SBS(first_idx=-2, second_idx=-5, factor=1))
    pipeline.add(FlaskViewer(FlaskServer()))
    pipeline.start(block=True)

def zetros_sync():
    stream_link = C_STREAM
    pipeline = LinkedListPipeline()
    pipeline.add(OpenCVCapture(stream_link, flip_h=True))
    pipeline.add(Llava("small", ooi="small toy"))
    pipeline.add(LegoZetros(host=C_ZETROS, image_idx=-2, text_idx=-1, dummy=False))
    pipeline.add(SBS(first_idx=-2, second_idx=-5, factor=1))
    pipeline.add(FlaskViewer(FlaskServer()))
    pipeline.start(block=True)

if __name__ == '__main__':
    zetros_async()
