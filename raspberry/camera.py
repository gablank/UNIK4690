#!/usr/bin/python3
# -*- coding: utf-8 -*-
import cv2
import subprocess

_v4l2_cmd = "v4l2-ctl"
_v4l2_select_device = "-d"
_camera_device = "/dev/video{}"
_get_property_cmd = "--get-ctrl={}"
_set_property_cmd = "--set-ctrl={}={}"
_list_devices_cmd = "--list-devices"
_set_frame_size_cmd = "--set-fmt-video=width={},height={}"
_get_frame_size_cmd = "--get-fmt-video"

BRIGHTNESS, \
CONTRAST, \
SATURATION, \
WHITE_BALANCE_TEMPERATURE_AUTO, \
WHITE_BALANCE_TEMPERATURE, \
SHARPNESS, \
BACKLIGHT_COMPENSATION, \
EXPOSURE_AUTO, \
EXPOSURE, \
FOCUS_AUTO, \
FOCUS, \
ZOOM \
    = range(12)


class Camera:
    def __init__(self, camera_idx=None):
        if camera_idx is None:
            camera_idx = self._detect_microsoft_lifecam()
            if camera_idx is None:
                raise RuntimeError(u"Unable to detect Microsoft® LifeCam Studio(TM)!")

            print(u"Found Microsoft® LifeCam Studio(TM) at {}".format(_camera_device.format(camera_idx)))

        self.cap = cv2.VideoCapture(camera_idx)

        if not self.cap.isOpened():
            raise RuntimeError("Unable to open camera!")

        self.camera_idx = camera_idx

        self._frame_width = None
        self._frame_height = None
        self.set_defaults()

    def capture(self):
        return self.cap.read()[1]

    def set_defaults(self):
        self.set_resolution(1920, 1080)
        self.set(BRIGHTNESS, 110)
        self.set(CONTRAST, 5)
        self.set(SATURATION, 100)
        self.set(WHITE_BALANCE_TEMPERATURE_AUTO, 0)
        self.set(WHITE_BALANCE_TEMPERATURE, 5000)
        self.set(SHARPNESS, 50)
        self.set(BACKLIGHT_COMPENSATION, 0)
        self.set(EXPOSURE_AUTO, 1)  # Not a bool, but mapping. 1 means manual, 3 is auto
        self.set(EXPOSURE, 1)
        self.set(FOCUS_AUTO, 0)
        self.set(FOCUS, 0)
        self.set(ZOOM, 0)

    def set(self, property, value):
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
            BRIGHTNESS: ("brightness", 30, 255),
            CONTRAST: ("contrast", 0, 10),
            SATURATION: ("saturation", 0, 200),
            WHITE_BALANCE_TEMPERATURE_AUTO: ("white_balance_temperature_auto", 0, 1),
            WHITE_BALANCE_TEMPERATURE: ("white_balance_temperature", 2500, 10000),
            SHARPNESS: ("sharpness", 0, 50),
            BACKLIGHT_COMPENSATION: ("backlight_compensation", 0, 10),
            EXPOSURE_AUTO: ("exposure_auto", 1, 3),
            EXPOSURE: ("exposure_absolute", 1, 10000),
            FOCUS_AUTO: ("focus_auto", 0, 1),
            FOCUS: ("focus_absolute", 0, 40),
            ZOOM: ("zoom_absolute", 0, 317),
        }[cv2_property]

    def set_resolution(self, width, height):
        self._call(_v4l2_cmd,
                   _v4l2_select_device,
                   _camera_device.format(self.camera_idx),
                   _set_frame_size_cmd.format(width, height))

        new_w, new_h = self.get_resolution()
        if (new_w, new_h) != (width, height):
            raise RuntimeError("Unable to set resolution!")

    def get_resolution(self):
        output = self._call(_v4l2_cmd,
                            _v4l2_select_device,
                            _camera_device.format(self.camera_idx),
                            _get_frame_size_cmd)

        output = output.split("\n")
        resolution_line = None
        for line in output:
            if line.find("Width/Height") >= 0:
                resolution_line = line
                break
        if resolution_line is not None:
            colon_position = resolution_line.find(":")
            resolution = resolution_line[colon_position+1:].split("/")
            self._frame_width, self._frame_height = int(resolution[0].strip()), int(resolution[1].strip())

        return self._frame_width, self._frame_height

    def _detect_microsoft_lifecam(self):
        output = self._call(_v4l2_cmd, _list_devices_cmd)

        lifecam = False
        for line in output.split("\n"):
            line_str = line.strip()
            if lifecam:
                dev_str = line_str
                break

            if line_str.find(u"Microsoft® LifeCam Studio(TM)") >= 0:
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
        self.get_resolution()
        print("Camera read from: {}".format(camera_device))
        print("Camera settings:")
        print("Resolution: {}x{}".format(self._frame_width, self._frame_height))
        print("Saturation:", self.get(SATURATION))
        print("Auto exposure:", self.get(EXPOSURE_AUTO))
        print("Exposure:", self.get(EXPOSURE))
        print("Auto focus:", self.get(FOCUS_AUTO))
        print("Focus:", self.get(FOCUS))
        print("Brightness:", self.get(BRIGHTNESS))
        print("Sharpness:", self.get(SHARPNESS))
        print("Auto white balance:", self.get(WHITE_BALANCE_TEMPERATURE_AUTO))
        print("White balance:", self.get(WHITE_BALANCE_TEMPERATURE))


if __name__ == "__main__":
    with Camera() as camera:

        while True:
            import utilities
            frame = camera.capture()
            frame = utilities.as_float32(frame)
            frame = utilities.as_uint8(frame)

            utilities.show(frame, time_ms=30, draw_histograms=True)
