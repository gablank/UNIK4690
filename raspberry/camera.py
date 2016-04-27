#!/usr/bin/python2
# -*- coding: utf-8 -*-
import cv2
import subprocess

_v4l2_cmd = "v4l2-ctl"
_v4l2_select_device = "-d"
_camera_device = "/dev/video{}"
_get_property_cmd = "--get-ctrl={}"
_set_property_cmd = "--set-ctrl={}={}"
_list_devices_cmd = "--list-devices"

cv2.CAP_PROP_AUTO_WHITE_BALANCE = 0x1337


class Camera:
    def __init__(self, camera_idx=None):
        if camera_idx is None:
            camera_idx = self._detect_microsoft_lifecam()
            if camera_idx is None:
                raise RuntimeError("Unable to detect Microsoft® LifeCam Studio(TM)!")

            print("Found Microsoft® LifeCam Studio(TM) at {}".format(_camera_device.format(camera_idx)))

        self.cap = cv2.VideoCapture(camera_idx)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera!")

        self.camera_idx = camera_idx

        self.set_defaults()

    def capture(self):
        return self.cap.read()[1]

    def set_defaults(self):
        self.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.set(cv2.CAP_PROP_SATURATION, 0.5)
        self.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
        self.set(cv2.CAP_PROP_EXPOSURE, 1)
        self.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.set(cv2.CAP_PROP_FOCUS, 0.05)
        self.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)
        self.set(cv2.CAP_PROP_SHARPNESS, 50)
        self.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, 5000)
        self.set(cv2.CAP_PROP_AUTO_WHITE_BALANCE, 0)

    def set(self, property, value):
        if property in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT,
                        cv2.CAP_PROP_SATURATION, cv2.CAP_PROP_CONTRAST,
                        cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_AUTOFOCUS,
                        cv2.CAP_PROP_BRIGHTNESS):
            self.cap.set(property, value)
        else:
            property_string, low, high = self._property_to_string(property)
            if low > value or high < value:
                raise ValueError("Value is outside of property range: {} <= {} <= {}".format(low, property_string, high))

            self._call(_v4l2_cmd,
                       _v4l2_select_device,
                       _camera_device.format(self.camera_idx),
                       _set_property_cmd.format(property_string, value))

        for _ in range(20):
            pass
            #self.capture()

    def get(self, property):
        if property in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT):
            return self.cap.get(property)
        else:
            property_as_string = self._property_to_string(property)[0]

            output = self._call(_v4l2_cmd,
                                _v4l2_select_device,
                                _camera_device.format(self.camera_idx),
                                _get_property_cmd.format(property_as_string))

            output = output.split("\n")[0].strip()
            colon_position = output.find(":")
            value = output[colon_position+1:]
            return int(value)

    def _property_to_string(self, cv2_property):
        return {
            cv2.CAP_PROP_AUTO_WHITE_BALANCE: ("white_balance_temperature_auto", 0, 1),
            cv2.CAP_PROP_WHITE_BALANCE_RED_V: ("white_balance_temperature", 2500, 10000),
            cv2.CAP_PROP_WHITE_BALANCE_BLUE_U: ("white_balance_temperature", 2500, 10000),
            cv2.CAP_PROP_AUTO_EXPOSURE: ("exposure_auto", 1, 3),
            cv2.CAP_PROP_EXPOSURE: ("exposure_absolute", 1, 10000),
            cv2.CAP_PROP_ZOOM: ("zoom_absolute", 0, 317),
            cv2.CAP_PROP_SHARPNESS: ("sharpness", 0, 50)
        }[cv2_property]

    def _detect_microsoft_lifecam(self):
        output = self._call(_v4l2_cmd, _list_devices_cmd)

        lifecam = False
        for line in output.split("\n"):
            line_str = line.strip()
            if lifecam:
                dev_str = line_str
                break

            if line_str.find("Microsoft® LifeCam Studio(TM)") >= 0:
                lifecam = True

        if not lifecam:
            return None

        idx = dev_str.find("/dev/video") + len("/dev/video")
        return int(dev_str[idx:])

    def _call(self, cmd, *args):
        c = [cmd] + [i for i in args]
        pid = subprocess.Popen(c, stdout=subprocess.PIPE)
        output = pid.communicate()[0]
        return output.decode("UTF-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    def __str__(self):
        camera_device = _camera_device.format(self.camera_idx)
        print("Camera read from: {}".format(camera_device))
        print("Camera settings:")
        print("Resolution: {}x{}".format(self.get(cv2.CAP_PROP_FRAME_WIDTH), self.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("Saturation:", self.get(cv2.CAP_PROP_SATURATION))
        print("Auto exposure:", self.get(cv2.CAP_PROP_AUTO_EXPOSURE))
        print("Exposure:", self.get(cv2.CAP_PROP_EXPOSURE))
        print("Auto focus:", self.get(cv2.CAP_PROP_AUTOFOCUS))
        print("Focus:", self.get(cv2.CAP_PROP_FOCUS))
        print("Brightness:", self.get(cv2.CAP_PROP_BRIGHTNESS))
        print("Sharpness:", self.get(cv2.CAP_PROP_SHARPNESS))
        print("White balance:", self.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))


if __name__ == "__main__":
    with Camera() as camera:

        while True:
            import utilities
            frame = camera.capture()
            frame = utilities.as_float32(frame)
            frame = utilities.as_uint8(frame)

            utilities.show(frame, time_ms=30, draw_histograms=True)
