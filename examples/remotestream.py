from pipelines import LinkedListPipeline
from utils import Buffer
from viewers import FlaskViewer, FlaskServer
from capture import OpenCVCapture

pipeline = LinkedListPipeline()
pipeline.add(Buffer(OpenCVCapture("http://141.252.222.139:5000/stream")))  # Open remote stream
pipeline.add(FlaskViewer(FlaskServer()))
pipeline.start(block=True)
