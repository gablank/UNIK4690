#!/usr/bin/python3
import cv2
import datetime


if __name__ == "__main__":
    import utilities
    from camera import Camera

    image_dir = "/home/anders/UNIK4690/project/images/microsoft_cam/red_balls/"

    with Camera() as camera:
        while True:
            camera.set(cv2.CAP_PROP_EXPOSURE, 1)
            frame = camera.capture()
            key = utilities.show(frame, time_ms=0, text="Exposure: 1")

            if key == ' ':
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.png")
                cv2.imwrite(image_dir + "1/" + filename, frame)

            if key == 'n':
                break

        while True:
            camera.set(cv2.CAP_PROP_EXPOSURE, 2)
            frame = camera.capture()
            key = utilities.show(frame, time_ms=0, text="Exposure: 2")

            if key == ' ':
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.png")
                cv2.imwrite(image_dir + "2/" + filename, frame)

            if key == 'n':
                break

        while True:
            camera.set(cv2.CAP_PROP_EXPOSURE, 5)
            frame = camera.capture()
            key = utilities.show(frame, time_ms=0, text="Exposure: 5")

            if key == ' ':
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S.png")
                cv2.imwrite(image_dir + "5/" + filename, frame)

        #        if ret:


#            ret = subprocess.call(["scp", "-P 22000", image_file, "anders@31.45.53.135:/var/www/unik4690_project/public_html/{}".format(filename)])
 #           print(ret)
  #          ret = subprocess.call(["ssh", "-p 22000", "anders@31.45.53.135", "/var/www/unik4690_project/link_newest.sh"])
   #         print(ret)


