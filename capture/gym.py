import os
import pickle
import time
from collections import deque

from xvfbwrapper import Xvfb
import gym
from typing import List, Any, Callable, Optional
from threading import Thread, Lock
import timeit
import numpy as np
import math
import random

class GymCapture:
    def __init__(self, name="CartPole-v0", verbose=False, report_freq=0, threaded=True, speed=0):
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
        self.episode = 0
        self.verbose = verbose
        self.terminate = False
        self.prev = None
        self.next = None
        self.begin = timeit.default_timer()
        self.iter = 0
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

    def step(self, episode, iter, restart=False):
        if restart:
            observation = self.reset_env()
            done = False
        else:
            observation, reward, done, info = self.env.step(self.env.action_space.sample())
        return done

    def reset_env(self):
        with self.lock:
            return self.env.reset()

    def update_one(self):
        done = self.step(self.episode, self.iter, restart=self.iter == 0)
        if done:
            if self.verbose:
                print(f"Done @ episode #{self.episode} and iter #{self.iter} with {self.name}")
            self.begin = timeit.default_timer()
            self.iter = 0
            self.episode += 1
        else:
            self.iter += 1

        # Calculate iters/sec
        seconds = timeit.default_timer() - self.begin

        if seconds > self.report_freq != 0:
            print(f"Gym @ {self.iter / seconds:.1f} iters/s")

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


# based on: https://github.com/deepakkavoor/cartpole-rl/blob/master/cartpole-q_learning.py
class QLearning(GymCapture):
    def __init__(self, name=None, threaded=False, save_func: Optional[Callable[['QLearning'], bool]] = None, verbose=True):
        GymCapture.__init__(self, name=name, threaded=threaded, verbose=verbose)

        # Hyper parameter settings
        self.buckets = (1, 1, 6, 3)  # (position, velocity, angle, angular velocity)
        self.min_epsilon = 0.1
        self.min_alpha = 0.1
        self.gamma = 0.99

        # The q table
        self.q_table = np.zeros(self.buckets + (self.env.action_space.n,))  # action space (left, right)

        # State information
        self.episode = 0
        self.curr_state = None

        # Score and reward information
        self.episode_reward = 0
        self.scores = deque(maxlen=100)

        self.save_func = save_func
        self.verbose = verbose

    def get_avg_score(self):
        return sum(self.scores) / len(self.scores)

    def select_action(self, state, epsilon):
        # implement the epsilon-greedy approach
        if random.random() <= epsilon:
            return self.env.action_space.sample()  # sample a random action with probability epsilon
        else:
            return np.argmax(self.q_table[state])  # choose greedy action with highest Q-value

    def get_epsilon(self, episode_number):
        # choose decaying epsilon in range [min_epsilon, 1]
        return max(self.min_epsilon, min(1., 1 - math.log10((episode_number + 1) / 25)))

    def get_alpha(self, episode_number):
        # choose decaying alpha in range [min_alpha, 1]
        return max(self.min_alpha, min(1., 1 - math.log10((episode_number + 1) / 25)))

    def update_table(self, old_state, action, reward, new_state, alpha):
        # updates the state-action pairs based on future reward
        new_state_q_value = np.max(self.q_table[new_state])
        self.q_table[old_state][action] += alpha * (reward + self.gamma * new_state_q_value - self.q_table[old_state][action])

    def discretize_state(self, state):
        upper_bounds = self.env.observation_space.high  # upper and lower bounds of state dimensions
        lower_bounds = self.env.observation_space.low

        upper_bounds[1] = 0.5
        upper_bounds[3] = math.radians(50)  # setting manual bounds for velocity and angular velocity
        lower_bounds[1] = -0.5
        lower_bounds[3] = -math.radians(50)

        # discretizing each input dimension into one of the buckets
        width = [upper_bounds[i] - lower_bounds[i] for i in range(len(state))]
        ratios = [(state[i] - lower_bounds[i]) / width[i] for i in range(len(state))]
        bucket_indices = [int(round(ratios[i] * (self.buckets[i] - 1))) for i in range(len(state))]

        # making the range of indices to [0, bucket_length]
        bucket_indices = [max(0, min(bucket_indices[i], self.buckets[i] - 1)) for i in range(len(state))]

        return tuple(bucket_indices)

    def step(self, episode, iter, restart=False) -> bool:
        # Get the new hyper parameters
        alpha = self.get_alpha(self.episode)
        epsilon = self.get_epsilon(self.episode)

        # Check for restart conditions
        if restart:
            if episode + iter > 0:
                self.scores.append(self.episode_reward)
            if len(self.scores) > 0:
                if self.verbose:
                    print(f"Cumulative episode {episode} reward {self.episode_reward}, Average of last {len(self.scores)} iters = {sum(self.scores) / len(self.scores):.2f}")
                self.save_func(self) if self.save_func else None
                self.episode_reward = 0

            # Get state for the newly initialized environment.
            obs = self.reset_env()
            self.curr_state = self.discretize_state(obs)

        # Select the next action
        action = self.select_action(self.curr_state, epsilon)

        # Progress the environment
        obs, reward, done, info = self.env.step(action)
        new_state = self.discretize_state(obs)

        # Update the q-table, states and reward
        self.update_table(self.curr_state, action, reward, new_state, alpha)
        self.curr_state = new_state
        self.episode_reward += reward

        return done


class QSaver:
    def __init__(self, base_filename: str, append_episode=True, save_freq=10, verbose=True):
        self.base_filename = base_filename
        self.append_episode = append_episode
        self.save_freq = save_freq
        self.verbose = verbose
        self.last_score = 0
        os.makedirs(os.path.dirname(base_filename), exist_ok=True)

    def save(self, obj: QLearning) -> Optional[str]:
        if obj.get_avg_score() <= self.last_score:
            return
        self.last_score = obj.get_avg_score()

        d = {'episode': obj.episode,
             'score': obj.get_avg_score(),
             'q_table': obj.q_table}

        if self.append_episode:
            filename = f"{self.base_filename}_{obj.episode}.txt"
        else:
            filename = f"{self.base_filename}.txt"

        with open(filename, "wb") as f:
            pickle.dump(d, f)
        if self.verbose:
            print(f"Saved to {filename} with score {obj.get_avg_score():.3f}")
        return filename

    def __call__(self, obj: QLearning) -> Optional[str]:
        if obj.episode % self.save_freq == 0 and obj.episode != 0:
            filename = self.save(obj)
            return filename

# class QLoader:
#     def __init__(self, base_filename: str, append_episode=True, verbose=True):
#         self.base_filename: str


