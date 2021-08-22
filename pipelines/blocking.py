from threading import Thread


class BlockingPipeline:
    def __init__(self):
        self.elements = []
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.terminate = False

    def add(self, element):
        self.elements.append(element)

    def update(self):
        while not self.terminate:
            captures = []
            for i in range(len(self.elements)):
                captures = self.elements[i](captures)
        print(f"Stopped {self.__class__}")

    def start(self, block: bool = False):
        print(f"Starting {self.__class__}")
        # for i in range(len(self.elements)):
        #     self.elements[i].start()
        self.thread.start()
        print(f"Started {self.__class__}")
        if block:
            self.wait()

    def stop(self):
        print(f"Stopping {self.__class__}")
        for i in range(len(self.elements)):
            self.elements[i].stop()
        self.terminate = True

    def wait(self, timeout=3):
        self.thread.join(timeout=timeout)
        return self.thread.is_alive()
