from viewers import FlaskServer, FlaskViewer
from capture import GymCapture, QLearning, QSaver
from pipelines import LinkedListPipeline
from utils import Buffer, Trigger


def exercise():
    # gym = "Hopper-v2"
    gym = "CartPole-v0"
    saver = QSaver("/tmp/qlearning/data", verbose=True, save_freq=100)

    pipeline = LinkedListPipeline()
    pipeline.add(Buffer(QLearning, gym, save_func=saver, threaded=True, verbose=False))
    pipeline.add(Trigger(triggers_per_second=15, report_freq=0))
    pipeline.add(FlaskViewer(FlaskServer()))
    pipeline.start(block=True)


if __name__ == '__main__':
    exercise()
