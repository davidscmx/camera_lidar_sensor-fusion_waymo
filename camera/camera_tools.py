

import cv2
import io
import numpy as np

from PIL import Image

from utils.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2
from easydict import EasyDict as edict


class CameraTools:
    def __init__(self):
        self.frame = None
        self.camera_names = edict()
        self.camera_names.front = dataset_pb2.CameraName.FRONT
        self.camera_names.side_left = dataset_pb2.CameraName.SIDE_LEFT
        self.camera_names.side_right = dataset_pb2.CameraName.SIDE_RIGHT
        self.camera_names.front_right = dataset_pb2.CameraName.FRONT_RIGHT
        self.camera_names.front_left = dataset_pb2.CameraName.FRONT_LEFT

    def set_frame(self, frame):
        self.frame = frame

    def set_selected_camera(self, camera_name):
        self.camera_name = camera_name

    def load_camera_data_structure(self) -> dataset_pb2.CameraImage:
        return [obj for obj in self.frame.images if obj.name == self.camera_name][0]

    def convert_image_to_rgb(self, camera_image: dataset_pb2.CameraImage):
        image_bytes = camera_image.image
        img = np.array(Image.open(io.BytesIO(image_bytes)))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize_img(self, img, factor=0.5):
        dim = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        return cv2.resize(img, dim)

    def resize_img_to_dims(self, img, dims=(360, 480)):
        return cv2.resize(img, dims)

    def decode_single_image(self):
        camera_image = self.load_camera_data_structure()
        img = self.convert_image_to_rgb(camera_image)
        return img

    def concatenate_all_camera_images(self):
        ys = np.array([])
        for camera_name in self.camera_names:
            self.set_selected_camera(camera_name)
            resized = self.decode_single_image()
            ys = np.hstack([ys, resized]) if ys.size else resized
        return ys
