from threading import Thread, Lock
import cv2
from typing import List, Any, Callable


class ViewerThread:
    def __init__(self, viewer: 'OpenCVViewer'):
        self.viewer = viewer
        self.terminate = False

    def __call__(self):
        while not self.terminate:
            self.viewer.show_frame()
        cv2.destroyAllWindows()
        print(f"Stopped {self.viewer.__class__} {id(self.viewer)}")


class OpenCVViewer:

    def __init__(self, input_idx=-1, on_stop: Callable = lambda: None):
        self.has_window = False
        self.input_idx = input_idx
        self.image = None
        self.lock = Lock()
        self.viewer_thread = ViewerThread(self)
        self.thread = Thread(target=self.viewer_thread, args=())
        self.thread.daemon = True
        self.on_stop = on_stop

        self.prev = None
        self.next = None

    def show_frame(self, name="droidClass"):
        with self.lock:
            if self.image is not None:
                key = cv2.waitKey(1)
                if self.has_window and cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) == 0 or key == ord('q'):
                    self.on_stop()
                else:
                    cv2.imshow(name, self.image)
                    self.has_window = True

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])
        if captures[self.input_idx] is not None:
            with self.lock:
                self.image = captures[self.input_idx].copy()
        return captures

    def start(self):
        print(f"Starting {self.__class__} {id(self)}")
        self.thread.start()
        print(f"Started {self.__class__} {id(self)}")

    def stop(self):
        print(f"Stopping {self.__class__} {id(self)}")
        if self.thread.is_alive():
            self.viewer_thread.terminate = True
        else:
            print(f"Already stopped {self.__class__} {id(self)}")

    def wait(self, timeout=3):
        self.thread.join(timeout=timeout)
        return self.thread.is_alive()
