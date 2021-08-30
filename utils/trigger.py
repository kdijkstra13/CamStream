from threading import Thread
from typing import List, Any
import time, timeit


class Trigger:
    def __init__(self, triggers_per_second=15, report_freq=2):
        """
        Trigger the pipeline a predefined iterations per second

        :param triggers_per_second: Number of triggers per second
        :param report_freq: Each report_freq seconds the actual iters per second is printed.
        """
        self.thread = Thread(target=self.trigger)
        self.terminate = False
        self.output_captures = None
        self.prev = None
        self.next = None
        self.triggers_per_second = triggers_per_second
        self.report_freq = report_freq

    def __call__(self, captures: List[Any]) -> List[Any]:
        while not self.output_captures:
            time.sleep(0)
        return self.output_captures

    def trigger(self):
        if not self.prev:
            raise RuntimeError("Trigger needs a previous element. Terminating")
        iters = 0
        begin = timeit.default_timer()
        while not self.terminate:
            start = timeit.default_timer()
            self.output_captures = self.prev([])
            end = timeit.default_timer()
            sleep = max(0., (1. / self.triggers_per_second) - (end - start))
            time.sleep(sleep)
            iters += 1
            seconds = timeit.default_timer() - begin
            if seconds > self.report_freq != 0:
                print(f"Trigger @ {iters/seconds:.1f} iters/s")
                begin = timeit.default_timer()
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

    def __del__(self):
        self.stop()
        self.wait()
