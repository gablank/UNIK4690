#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
This file must be compatible with Python 2 and 3!
"""
import camera
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


class LocalCamera(camera.Camera):
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
        self._camera_device = _camera_device.format(self.camera_idx)

        super(LocalCamera, self).__init__()

    def capture(self):
        # Make sure any new settings have been applied
        for _ in range(8):
            self.cap.read()
        return self.cap.read()[1]

    def set_defaults(self):
        self.set_resolution(1920, 1080)
        self.set(camera.BRIGHTNESS, 110)
        self.set(camera.CONTRAST, 5)
        self.set(camera.SATURATION, 100)
        self.set(camera.WHITE_BALANCE_TEMPERATURE_AUTO, 0)
        self.set(camera.WHITE_BALANCE_TEMPERATURE, 5000)
        self.set(camera.SHARPNESS, 50)
        self.set(camera.BACKLIGHT_COMPENSATION, 0)
        self.set(camera.EXPOSURE_AUTO, 1)  # Not a bool, but mapping. 1 means manual, 3 is auto
        self.set(camera.EXPOSURE, 1)
        self.set(camera.FOCUS_AUTO, 0)
        self.set(camera.FOCUS, 0)
        self.set(camera.ZOOM, 0)

    def set(self, property, value):
        property_string = property

        if isinstance(property, int):
            property_string, low, high = self._property_to_string(property)

            if low > value or high < value:
                raise ValueError("Value is outside of property range: {} <= {} <= {}".format(low, property_string, high))

        self._call(_v4l2_cmd,
                   _v4l2_select_device,
                   _camera_device.format(self.camera_idx),
                   _set_property_cmd.format(property_string, value))

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

    def set_resolution(self, width, height):
        self.cap.set(3, width)
        self.cap.set(4, height)

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


if __name__ == "__main__":
    with LocalCamera() as camera:

        while True:
            import utilities
            frame = camera.capture()

            utilities.show(frame, time_ms=30, draw_histograms=True)
