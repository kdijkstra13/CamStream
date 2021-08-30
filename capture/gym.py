from xvfbwrapper import Xvfb
import gym
from typing import List, Any
from threading import Thread, Lock
import timeit
# based on: https://github.com/deepakkavoor/cartpole-rl/blob/master/cartpole-q_learning.py


class GymCapture:
    def __init__(self, name="CartPole-v0", verbose=True, report_freq=2):
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

    def __call__(self, captures: List[Any]) -> List[Any]:
        with self.lock:
            image = self.env.render(mode='rgb_array')
            captures.append(image)
            return captures

    def update(self):
        begin = timeit.default_timer()
        iters = 0
        while not self.terminate:
            observation, reward, done, info = self.env.step(self.env.action_space.sample())  # take a random action
            if done:
                if self.verbose:
                    print(f"Reset #{self.counter} {self.name}")
                self.counter += 1
                with self.lock:
                    self.env.reset()
            iters += 1
            seconds = timeit.default_timer() - begin
            if seconds > self.report_freq != 0:
                print(f"Gym @ {iters/seconds:.1f} iters/s")
                begin = timeit.default_timer()
                iters = 0
        print(f"Stopped {self.__class__} {id(self)}")

    def start(self, block: bool = False):
        print(f"Start {self.__class__} {id(self)}")
        self.vdisplay.start()
        self.env.reset()
        self.thread.start()
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
