from viewers import FlaskServer, FlaskViewer
from capture import GymCapture
from pipelines import LinkedListPipeline
from utils import Buffer, Trigger


def exercise():
    # gym = "Hopper-v2"
    gym = "CartPole-v0"

    pipeline = LinkedListPipeline()
    pipeline.add(Buffer(GymCapture, gym, verbose=False, threaded=False, speed=15, report_freq=2))
    pipeline.add(FlaskViewer(FlaskServer()))
    pipeline.start(block=True)


if __name__ == '__main__':
    exercise()
