from threading import Thread
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)  # compatible with CUDA
from multiprocessing import Process, Pipe, Value
from typing import Any, List
import ctypes
import time


def update_element(input_captures_conn, output_captures_conn, element, args, kwargs, terminate):
    """
    This method is only used when use_mp is True
    """
    try:
        if type(element) == type:
            element = element(*args, **kwargs)
        element.start()
        while not terminate.value:
            input_captures = input_captures_conn.recv()
            output_captures = element(input_captures)
            output_captures_conn.send(output_captures)
    finally:
        print(f"Stopped Buffer process")


class Buffer:
    def __init__(self, element, *args, default_idx=-1, use_mp=True, **kwargs):
        """
        This class wraps an element and creates a non-blocking __call__() method. The element is either executed using
            threading if use_mp is False and with multiprocessing is otherwise.

        :param element: The element to wrap. This is either an instance of an element or a class
            that will be constructed with *args and **kwargs. Note: instantiation is useful when the state of a class
            instance cannot be transferred to another process by pickling in the case of spawning a new process.
        :param args: Arguments for the element-instance construction.
        :param default_idx: The input index to use when the wrapped element has no captures ready.
        :param use_mp: Use multiprocessing if this is True
        :param kwargs: Keyword arguments for the element-instance construction.
        """
        self.terminate = Value(ctypes.c_bool)
        self.terminate.value = False
        self.input_captures_conn, input_captures_child = Pipe()
        self.output_captures_conn, output_captures_child = Pipe()
        if use_mp:
            # Create a process that creates the element
            self.element = None
            self.process = Process(target=update_element, args=(
            input_captures_child, output_captures_child, element, args, kwargs, self.terminate))
        else:
            # Create the element or assign it
            self.element = element(*args, **kwargs) if type(element) == type else element
            self.process = None
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True

        self.default_idx = default_idx
        self.input_captures = None
        self.output_captures = None
        self.next = None
        self.prev = None

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])

        num_caps = len(captures)
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
            output_captures.append(output_captures[self.default_idx].copy())
            return output_captures
        else:
            # Intended output
            # return the new result (keep te original content of captures)
            output_captures = captures.copy()
            for i in range(num_caps, len(self.output_captures)):
                output_captures.append(self.output_captures[i])
            return output_captures

    def update(self):
        try:
            while not self.terminate.value:
                if self.input_captures is not None:
                    if self.process is not None:
                        # Use MP
                        self.input_captures_conn.send(self.input_captures)
                        output_captures = self.output_captures_conn.recv()
                    else:
                        # Only Threading
                        output_captures = self.element(self.input_captures)
                    self.output_captures = output_captures
                    self.input_captures = None
                time.sleep(0)
            if self.process is not None:
                self.process.join()
        finally:
            print(f"Stopped thread {self.__class__} {id(self)}")

    def start(self, block: bool = False):
        print(f"Starting {self.__class__} {id(self)}")
        if not self.thread.is_alive():
            if self.process is not None:
                self.process.start()
            elif self.element:
                self.element.start()
            self.thread.start()
            print(f"Started  {self.__class__} {id(self)}")
            if block:
                self.wait()
        else:
            print(f"Already started  {self.__class__} {id(self)}")

    def stop(self):
        print(f"Stopping {self.__class__} {id(self)}")
        self.terminate = True
        if self.element:
            self.element.stop()

    def wait(self, timeout=None):
        if self.process is not None:
            self.process.join(timeout=timeout)
        elif self.element:
            self.element.wait(timeout=timeout)
        self.thread.join(timeout=timeout)
        return self.thread.is_alive()
