

import cv2
import numpy as np
import unittest
from pathlib import Path

from camera.camera_tools import CameraTools
from utils.loader_tools import prepare_waymo_dataset
from utils.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2

filename = "training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord"


class TestCameraTools(unittest.TestCase):
    def setUp(self):
        data_iter = prepare_waymo_dataset(filename)
        self.frame = next(data_iter)
        data_iter.file.close()

        self.cam_tools = CameraTools()
        self.cam_tools.set_frame(self.frame)
        self.cam_tools.set_selected_camera(self.cam_tools.camera_names.front)

        self.camera_image = self.cam_tools.load_camera_data_structure()

    def test_camera_names(self):
        self.assertEqual(self.cam_tools.camera_names.front, dataset_pb2.CameraName.FRONT)

        self.assertEqual(self.cam_tools.camera_names.front_right,
                         dataset_pb2.CameraName.FRONT_RIGHT)

        self.assertEqual(self.cam_tools.camera_names.front_left,
                         dataset_pb2.CameraName.FRONT_LEFT)

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

        resized_img_to_dims = self.cam_tools.resize_img_to_dims(img, (500, 500))
        self.assertEqual(500, resized_img_to_dims.shape[0])
        self.assertEqual(500, resized_img_to_dims.shape[1])

    def test_decode_single_image(self):
        img = self.cam_tools.decode_single_image()
        ref_img_str = "training_segment-10072231702153043603_5725_000_5745_000_with_camera_0.png"
        ref_img_dir = Path("tests/test_data/")
        ref_img_path = ref_img_dir / ref_img_str
        assert ref_img_path.exists(), "Image not found"
        ref_img = cv2.imread(str(ref_img_path))
        self.assertIsNone(np.testing.assert_array_equal(img, ref_img, verbose=False))


if __name__ == "__main__":
    unittest.main()
