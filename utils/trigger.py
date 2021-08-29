from threading import Thread
from typing import List, Any
import time, timeit


class Trigger:
    def __init__(self, report_freq=5):
        self.thread = Thread(target=self.trigger)
        self.terminate = False
        self.output_captures = None
        self.prev = None
        self.next = None
        self.report_freq = report_freq

    def __call__(self, captures: List[Any]) -> List[Any]:
        while not self.output_captures:
            time.sleep(0)
        return self.output_captures

    def trigger(self):
        if not self.prev:
            raise RuntimeError("Trigger needs a previous element. Terminating")
        iters = 0
        start = timeit.default_timer()
        while not self.terminate:
            self.output_captures = self.prev([])
            time.sleep(0)
            iters += 1
            el = timeit.default_timer()
            if el - start > self.report_freq and not self.report_freq == 0:
                print(f"{iters / el:.2f} iters/s")
                start = timeit.default_timer()
                iters = 0

        print(f"Stopped {self.__class__} {id(self)}")

    def start(self, block: bool = False):
        print(f"Starting {self.__class__} {id(self)}")
        self.thread.start()
        print(f"Started {self.__class__} {id(self)}")
        if block:
            self.wait()

    def stop(self):
        self.terminate = True

    def wait(self, timeout=None):
        self.thread.join(timeout)
        return self.thread.is_alive()
