from pipelines import LinkedListPipeline
from utils import Buffer
from viewers import FlaskViewer, FlaskServer
from capture import OpenCVCapture

pipeline = LinkedListPipeline()
pipeline.add(Buffer(OpenCVCapture(0)))  # Open the first camera
pipeline.add(FlaskViewer(FlaskServer()))
pipeline.start(block=True)
