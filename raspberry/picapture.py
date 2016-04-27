#!/usr/bin/python2
import camera
import SimpleHTTPServer
import SocketServer
import cv2


class CameraCaptureHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        with camera.Camera() as cam:
            bgr = cam.capture()

        cv2.imwrite("image.png", bgr)
        self.path = "image.png"
        return SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET()


if __name__ == "__main__":
    server = SocketServer.TCPServer(("", 1337), CameraCaptureHandler)

    server.serve_forever()
