
from PIL import Image
import numpy as np
import zlib

## Waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2

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

def print_range_image_shape(frame, lidar_name):

    range_image = load_range_image(frame, lidar_name)
    print(range_image.shape)
