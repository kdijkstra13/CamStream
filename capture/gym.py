from xvfbwrapper import Xvfb
import gym
from typing import List, Any


class GymCapture:
    def __init__(self, name="CartPole-v0", verbose=True):
        self.name = name
        self.env = gym.make(name)
        self.vdisplay = Xvfb()
        self.counter = 0
        self.verbose = verbose
        self.prev = None
        self.next = None

    def __call__(self, captures: List[Any]) -> List[Any]:
        image = self.next_step()
        captures.append(image)
        return captures

    def next_step(self):
        image = self.env.render(mode='rgb_array')
        observation, reward, done, info = self.env.step(self.env.action_space.sample())  # take a random action
        if done:
            if self.verbose:
                print(f"Reset #{self.counter} {self.name}")
            self.counter += 1
            self.env.reset()
        return image

    def start(self, block: bool = False):
        self.vdisplay.start()
        self.env.reset()
        return

    def stop(self):
        self.env.close()
        self.vdisplay.stop()

    def wait(self):
        self.vdisplay.proc.wait()
        return self.vdisplay.proc.is_alive()
