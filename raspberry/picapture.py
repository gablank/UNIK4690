#!/usr/bin/python2
import camera
import SimpleHTTPServer
import SocketServer
import cv2


class CameraCaptureHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.find("/new_image.png") >= 0:
            exposure_pos = self.path.find("&exposure=")
            exposure = None
            if exposure_pos >= 0:
                exposure = int(self.path[exposure_pos+len("&exposure="):])
                print("Setting exposure to {}".format(exposure))

            with camera.Camera() as cam:
                if exposure is not None:
                    cam.set(camera.EXPOSURE, exposure)
                bgr = cam.capture()
                print(cam)

            cv2.imwrite("image.png", bgr)
            self.path = "image.png"
            return SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

        self.send_error(402)


if __name__ == "__main__":
    server = SocketServer.TCPServer(("", 1337), CameraCaptureHandler)

    try:
        server.serve_forever()
    except Exception as e:
        print(e)
        import traceback
        traceback.print_tb(e.__traceback__)
    finally:
        server.server_close()
