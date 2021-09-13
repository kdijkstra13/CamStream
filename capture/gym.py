import time

from xvfbwrapper import Xvfb
import gym
from typing import List, Any
from threading import Thread, Lock
import timeit
# based on: https://github.com/deepakkavoor/cartpole-rl/blob/master/cartpole-q_learning.py


class GymCapture:
    def __init__(self, name="CartPole-v0", verbose=True, report_freq=2, threaded=True, speed=0):
        """
        A generic OpenAI gym object

        :param name: Name of the environment
        :param verbose: Print messages about the status of the gym
        :param report_freq: Frequency of printing messages
        :param threaded: If True the gym constantly updates in the background, when False the gym only updates on
            a call to __call__().
        :param speed: Force a certain amount of maximum steps per second (0 = no limit).
        """
        self.thread = Thread(target=self.update)
        self.lock = Lock()
        self.report_freq = report_freq
        self.name = name
        self.env = gym.make(name)
        self.vdisplay = Xvfb()
        self.counter = 0
        self.verbose = verbose
        self.terminate = False
        self.prev = None
        self.next = None
        self.begin = timeit.default_timer()
        self.iters = 0
        self.threaded = threaded
        self.speed = speed
        self.start_loop = timeit.default_timer()

    def __call__(self, captures: List[Any]) -> List[Any]:
        if not self.threaded:
            self.update_one()
        with self.lock:
            image = self.env.render(mode='rgb_array')
            captures.append(image)
        return captures

    def step(self):
        # take a random action by default
        observation, reward, done, info = self.env.step(self.env.action_space.sample())
        return done

    def update_one(self):
        # run one iter
        if self.step():
            if self.verbose:
                print(f"Reset #{self.counter} {self.name}")
            self.counter += 1
            with self.lock:
                self.env.reset()

        self.iters += 1

        # Calculate iters/sec
        seconds = timeit.default_timer() - self.begin

        if seconds > self.report_freq != 0:
            print(f"Gym @ {self.iters / seconds:.1f} iters/s")
            self.begin = timeit.default_timer()
            self.iters = 0

        # force iters/sec?
        if self.speed > 0:
            end_loop = timeit.default_timer()
            elapsed = end_loop - self.start_loop
            wait_time = max(0., (1 / self.speed) - elapsed)
            time.sleep(wait_time)
            self.start_loop = timeit.default_timer()
        else:
            time.sleep(0)

    def update(self):
        while not self.terminate:
            self.update_one()
            time.sleep(0)
        print(f"Stopped {self.__class__} {id(self)}")

    def start(self, block: bool = False):
        print(f"Start {self.__class__} {id(self)}")
        self.vdisplay.start()
        self.env.reset()
        self.thread.start() if self.threaded else None
        print(f"Started {self.__class__} {id(self)}")
        if block:
            self.wait()

    def stop(self):
        self.terminate = True
        self.env.close()
        self.vdisplay.stop()

    def wait(self, timeout=None):
        self.vdisplay.proc.wait()
        self.thread.join(timeout)
        return self.vdisplay.proc.is_alive() or self.thread.is_alive()

    def __del__(self):
        self.stop()
