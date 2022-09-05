
import cv2
from PIL import Image
from enum import Enum
import math
import numpy as np
import zlib

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2
from .types import RANGE_IMAGE_CELL

def load_range_image(frame, lidar_name):
    # get laser data structure from frame
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0]
    range_image = []
    # use first response
    if len(lidar.ri_return1.range_image_compressed) > 0:
        range_image = dataset_pb2.MatrixFloat()
        range_image.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        range_image = np.array(range_image.data).reshape(range_image.shape.dims)

    return range_image

def get_range_image_shape(frame, lidar_name):
    range_image = load_range_image(frame, lidar_name)
    return range_image.shape

def print_pitch_resolution(frame, lidar_name):
    range_image = load_range_image(frame, lidar_name)
    # compute vertical field-of-view from lidar calibration
    lidar_calibration = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]
    min_pitch = lidar_calibration.beam_inclination_min
    max_pitch = lidar_calibration.beam_inclination_max
    vertical_fov = max_pitch - min_pitch

    # compute pitch resolution and convert it to angular minutes
    pitch_resolution_radians = vertical_fov / range_image.shape[0]
    pitch_resolution_degrees = pitch_resolution_radians * (180/np.pi)
    print("pitch angle resolution " + "{0:.2f}".format(pitch_resolution_degrees)+" degrees")

def get_min_max_distance(frame, lidar_name):
    range_image = load_range_image(frame, lidar_name)
    range_image[range_image<0] = 0.0

    return (round(np.amin(range_image[:,:,0]),2), round(np.amax(range_image[:,:,0]),2))

def contrast_adjustment(img):
    return np.amax(img)/2 * img * 255 / (np.amax(img) - np.amin(img))

def map_to_8bit(range_image, channel):
    img_channel = range_image[:,:,channel]

    if channel == RANGE_IMAGE_CELL.RANGE.value:
        img_channel = img_channel * 255 / (np.amax(img_channel) - np.amin(img_channel))
    elif channel == RANGE_IMAGE_CELL.INTENSITY.value:
        img_channel = contrast_adjustment(img_channel)

    img_channel = img_channel.astype(np.uint8)
    return img_channel

def crop_channel_azimuth(img_channel, division_factor):
    opening_angle = int(img_channel.shape[1] / division_factor)
    img_channel_center = int(img_channel.shape[1] / 2)
    img_channel = img_channel[:, img_channel_center - opening_angle : img_channel_center + opening_angle]
    return img_channel

def visualize_selected_channel(frame, lidar_name, channel):
    range_image = load_range_image(frame, lidar_name)
    range_image[range_image<0] = 0.0

    img_range = map_to_8bit(range_image, channel = channel.value)
    img_range = crop_channel_azimuth(img_range, division_factor=8)

    cv2.imshow(channel.name, img_range)
    cv2.waitKey(0)


def range_image_to_point_cloud(frame, lidar_name):

    range_image = load_range_image(frame, lidar_name)
    range_image[range_image<0]=0.0
    img_range = range_image[:,:,0]

    height = img_range.shape[0]

    lidar_calibration = [obj for obj in frame.context.laser_calibrations if obj.name == lidar_name][0]

    # inclinations/pitches have to be reversed in order so that the
    # first angle corresponds to the top-most measurement.
    inclination_min = lidar_calibration.beam_inclination_min
    inclination_max = lidar_calibration.beam_inclination_max

    # TODO: check -> Convert to angular minutes
    inclinations = np.linspace(inclination_min, inclination_max, height)*60
    inclinations = np.flip(inclinations)

    extrinsic_matrix = np.array(lidar_calibration.extrinsic.transform).reshape(4,4)

    # Compute the α angle betweetn the x and y axis from the extrinsic calibration matrix
    #         [cosαcosβ ... ]
    # [R,t] = [sinαcosβ ... ]
    #         [-sinβ ...    ]
    width = img_range.shape[1]
    azimuth_correction = math.atan2(extrinsic_matrix[1,0], extrinsic_matrix[0,0])
    print(azimuth_correction)
    azimuth = np.linspace(np.pi,-np.pi, width) - azimuth_correction


    print(azimuth)