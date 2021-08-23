from viewers import FlaskServer, FlaskViewer
from capture import GymCapture
from pipelines import LinkedListPipeline
from utils import Buffer


def exercise():
    gym = "Hopper-v2"
    #gym = "CartPole-v0"

    pipeline = LinkedListPipeline()
    pipeline.add(Buffer(GymCapture(gym)))
    pipeline.add(FlaskViewer(FlaskServer()))
    pipeline.start(block=True)


if __name__ == '__main__':
    exercise()
