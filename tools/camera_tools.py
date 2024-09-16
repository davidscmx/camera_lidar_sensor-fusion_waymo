
from PIL import Image
import cv2
import io
import numpy as np
import unittest
from pathlib import Path

from tools.loader_tools import prepare_waymo_dataset
from waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2
from waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
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
        return waymo_utils.get(self.frame.images, self.camera_name)

    def convert_image_to_rgb(self, camera_image: dataset_pb2.CameraImage):
        image_bytes = camera_image.image
        img = np.array(Image.open(io.BytesIO(image_bytes)))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def resize_img(self, img, factor = 0.5):
        dim = (int(img.shape[1] * factor), int(img.shape[0] * factor))
        return cv2.resize(img, dim)

    def resize_img_to_dims(self, img, dims=(360,480)):
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

    #def display_images(self, camera_name):
    #    img = self.prepare_single_image(camera_name)
    #    self.show_image(img, "all images")
#
    #def display_images_side_by_side(self):
    #    ys = self.prepare_all_images()
    #    self.show_image(ys, "all images")

data_filename = 'training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord'

class TestCameraTools(unittest.TestCase):

    def setUp(self):
        data_iter = prepare_waymo_dataset(data_filename)
        self.frame = next(data_iter)
        data_iter.file.close()

        self.cam_tools = CameraTools()
        self.cam_tools.set_frame(self.frame)
        self.cam_tools.set_selected_camera(self.cam_tools.camera_names.front)

        self.camera_image = self.cam_tools.load_camera_data_structure()

    def test_camera_names(self):
        self.assertEqual(self.cam_tools.camera_names.front, dataset_pb2.CameraName.FRONT)
        self.assertEqual(self.cam_tools.camera_names.front_right, dataset_pb2.CameraName.FRONT_RIGHT)

    def test_load_camera_data_structure(self):
        self.assertIsInstance(self.camera_image, dataset_pb2.CameraImage)

    def test_convert_image_to_rgb(self):
        self.img = self.cam_tools.convert_image_to_rgb(self.camera_image)
        self.assertEqual(self.img.shape[2], 3)

    def test_resize_img(self):
        img = self.cam_tools.convert_image_to_rgb(self.camera_image)
        original_shape = img.shape
        resized_img = self.cam_tools.resize_img(img, factor=0.5)
        self.assertEqual(original_shape[0]/2, resized_img.shape[0])
        self.assertEqual(original_shape[1]/2, resized_img.shape[1])

        resized_img_to_dims = self.cam_tools.resize_img_to_dims(img, (500,500))
        self.assertEqual(500, resized_img_to_dims.shape[0])
        self.assertEqual(500, resized_img_to_dims.shape[1])

    def test_decode_single_image(self):
        img = self.cam_tools.decode_single_image()
        ref_img = cv2.imread("tools/training_segment-10072231702153043603_5725_000_5745_000_with_camera_0.png")
        self.assertIsNone(np.testing.assert_array_equal(img, ref_img, verbose=True))

if __name__ == "__main__":
    unittest.main()