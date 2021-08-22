from threading import Thread
import time


class LinkedListPipeline:
    def __init__(self):
        self.elements = []
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.terminate = False

    def add(self, element):
        if len(self.elements) > 0:
            if element.prev is None:
                element.prev = self.elements[-1]
            if self.elements[-1].next is None:
                self.elements[-1].next = element
        self.elements.append(element)

    def update(self):
        while not self.terminate:
            time.sleep(1)
        print(f"Stopped {self.__class__} {id(self)}")

    def start(self, block: bool = False):
        print(f"Starting {self.__class__} {id(self)}")
        for i in range(len(self.elements)):
            print(f"{self.elements[i].__class__} {id(self)}")
            self.elements[i].start(block=block)
        self.thread.start()
        print(f"Started {self.__class__} {id(self)}")
        if block:
            self.wait()

    def stop(self):
        print(f"Stopping {self.__class__} {id(self)}")
        for i in range(len(self.elements)):
            self.elements[i].stop()
        self.terminate = True

    def wait(self, timeout=3):
        self.thread.join(timeout=timeout)
        return self.thread.is_alive()
