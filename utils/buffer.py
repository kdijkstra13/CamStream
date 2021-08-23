from threading import Thread, Lock
from typing import Any, List
import time


class Buffer:
    input_captures = None
    output_captures = None

    def __init__(self, element, input_idx=-1, continuous=False):
        self.lock = Lock()
        self.thread = Thread(target=self.update, args=())
        self.continuous = continuous
        self.thread.daemon = True
        self.terminate = False
        self.element = element
        self.input_idx = input_idx
        self.next = None
        self.prev = None

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])

        num_caps = len(captures)
        with self.lock:
            if self.input_captures is None:
                # start processing the next image
                self.input_captures = captures.copy()

        if len(captures) == 0:
            # If there is no output and no captures we have to block
            while self.output_captures is None:
                time.sleep(0)

        if self.output_captures is None:
            # If there is no output return the input instead
            output_captures = captures.copy()
            output_captures.append(output_captures[self.input_idx].copy())
            return output_captures
        else:
            # Intended output
            with self.lock:
                # return the new result (keep te original content of captures)
                output_captures = captures.copy()
                for i in range(num_caps, len(self.output_captures)):
                    output_captures.append(self.output_captures[i])
                return output_captures

    def update(self):
        while not self.terminate:
            if self.input_captures is not None:
                output_captures = self.element(self.input_captures)
                with self.lock:
                    self.output_captures = output_captures
                    self.input_captures = None
            time.sleep(0)
        print(f"Stopped {self.__class__} {id(self)}")

    def start(self, block: bool = False):
        print(f"Starting {self.__class__} {id(self)}")
        if not self.thread.is_alive():
            self.element.start()
            self.thread.start()
            print(f"Started  {self.__class__} {id(self)}")
        else:
            print(f"Already started  {self.__class__} {id(self)}")

    def stop(self):
        print(f"Stopping {self.__class__} {id(self)}")
        self.terminate = True
        self.element.stop()

    def wait(self, timeout=3):
        self.element.wait()
        self.thread.join(timeout=timeout)
        return self.thread.is_alive()
