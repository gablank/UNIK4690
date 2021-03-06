#!/usr/bin/python2
import SimpleHTTPServer
import SocketServer
import cv2
from urlparse import urlparse, parse_qs
import lifecamstudiocamera
import raspberrycamera


class CameraCaptureHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/lifecam_image.png":
            with lifecamstudiocamera.LifeCamStudioCamera() as cam:
                frame_width = None
                frame_height = None
                parsed_query = parse_qs(parsed.query)
                for property, value_list in parsed_query.items():
                    if len(value_list) != 1:
                        continue

                    value = int(value_list[0])
                    if property == "width":
                        frame_width = value
                    elif property == "height":
                        frame_height = value
                    else:
                        cam.set(property, value)

                if frame_width is not None and frame_height is not None:
                    cam.set_resolution(frame_width, frame_height)

                bgr = cam.capture()
                print(cam)

            cv2.imwrite("lifecam_image.png", bgr)
            self.path = "lifecam_image.png"

            return SimpleHTTPServer.SimpleHTTPRequestHandler.do_GET(self)

        elif parsed.path == "/raspberry_image.png":
            with raspberrycamera.RaspberryCamera() as cam:
                frame_width = None
                frame_height = None
                parsed_query = parse_qs(parsed.query)
                for property, value_list in parsed_query.items():
                    if len(value_list) != 1:
                        continue

                    value = int(value_list[0])
                    if property == "width":
                        frame_width = value
                    elif property == "height":
                        frame_height = value
                    else:
                        cam.set(property, value)

                if frame_width is not None and frame_height is not None:
                    cam.set_resolution(frame_width, frame_height)

                bgr = cam.capture()
                print(cam)

            cv2.imwrite("raspberry_image.png", bgr)
            self.path = "raspberry_image.png"

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
