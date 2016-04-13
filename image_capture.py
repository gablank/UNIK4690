#!/usr/bin/python3
import cv2
import datetime
import subprocess

EAST_CAMERA = ("east", 3)
SOUTH_CAMERA = ("south", 2)
cameras = [SOUTH_CAMERA]


if __name__ == "__main__":
    with open("test", "w") as f:
        f.write("running!")

    for folder, camera_idx in cameras:
        cap = cv2.VideoCapture(camera_idx)

        if not cap.isOpened():
            with open("log", "a+") as f:
                f.write("{}: Unable to open {}!\n".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), folder))
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        ctr = 0
        ret = False
        # Capture 50 images before saving so the camera has time to focus
        while ctr < 50:
            ret, frame = cap.read()
            ctr += 1

        image_file = "/home/anders/UNIK4690/project/temp.png"

        if ret:
            cv2.imwrite(image_file, frame)

            filename = str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")) + ".png"
            ret = subprocess.call(["scp", "-P 22000", image_file, "anders@31.45.53.135:/var/www/unik4690_project/public_html/{}".format(filename)])
            print(ret)
            ret = subprocess.call(["ssh", "-p 22000", "anders@31.45.53.135", "/var/www/unik4690_project/link_newest.sh"])
            print(ret)


