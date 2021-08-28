from flask import Flask, Response
import cv2
from typing import Any, List
from threading import Thread
import time


class GetImage:
    def __init__(self, viewer):
        self.viewer = viewer
        self.prev_result = None

    def gen(self):
        while True:
            frame = self.viewer.get_image()
            # frame = np.random.uniform(0, 255, [100, 100, 3])

            if frame is None:
                print("Missed frame (buffer error)")
                yield self.prev_result

            success, a_numpy = cv2.imencode('.jpg', frame)
            if not success:
                print("Missed frame (unable to decode jpg)")
                yield self.prev_result

            jpg_frame = a_numpy.tostring()

            result = (b'--frame\r\n' 
                      b'Content-Type: image/jpeg\r\n\r\n' + jpg_frame + b'\r\n')

            self.prev_result = result
            yield result

    def __call__(self):
        content = self.gen()
        if content is not None:
            return Response(content, mimetype='multipart/x-mixed-replace; boundary=frame')
        else:
            return '', 204


class FlaskViewer:

    def __init__(self, server: 'FlaskServer', input_idx=-1, stream_url="/stream", stream_name="stream"):

        self.server = server
        self.server.app.add_url_rule(stream_url, stream_name, GetImage(self))

        self.input_idx = input_idx
        self.image = None

        self.prev = None
        self.next = None

    def __call__(self, captures: List[Any]) -> List[Any]:
        if self.prev:
            captures = self.prev([])
        if captures[self.input_idx] is not None:
            self.image = captures[self.input_idx].copy()
        return captures

    def get_image(self):
        if self.prev:
            self([])
        return self.image

    def start(self, block: bool = False):
        self.server.start(block)

    def stop(self):
        self.server.stop()

    def wait(self, timeout=3):
        return self.server.wait(timeout)


class FlaskServer:

    def __init__(self, name="flask_server", port=5000):
        self.app = Flask(name)
        self.port = port

        self.thread = Thread(target=self.start_flask, args=())
        self.thread.daemon = True

    def __call__(self, captures: List[Any]) -> List[Any]:
        return captures

    def start_flask(self):
        self.app.run(host='0.0.0.0', debug=False, use_reloader=False, port=self.port)
        print(f"Stopped  {self.__class__} {id(self)}")

    def start(self, block: bool = False):
        if not self.thread.is_alive():
            print(f"Starting {self.__class__} {id(self)}")
            self.thread.start()
            print(f"Started {self.__class__} {id(self)}")
        if block:
            self.thread.join()

    def stop(self):
        print(f"Cannot stop {self.__name__} {id(self)}")

    def wait(self, timeout=3):
        return self.thread.join(timeout=timeout)
