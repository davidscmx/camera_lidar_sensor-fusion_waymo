# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Process the point-cloud and prepare it for object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------

# general package imports
import cv2
import numpy as np
import torch
import open3d
# add project directory to python path to enable relative imports
import os
import sys
import time
from enum import Enum
import zlib
import matplotlib.pyplot as plt


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools

class RangeImgChannel(Enum):
    Range = 0
    Intensity = 1
    Elongation = 2
    Is_in_no_label_zone = 3

#class
# 0 x
# 1 y
# 2 height
# 3 intensity


vis = open3d.visualization.VisualizerWithKeyCallback()
lidar_frame_counter = 0
def show_pcl(pcl):
    global lidar_frame_counter

    if not lidar_frame_counter:
        vis.create_window()

    pcd = open3d.geometry.PointCloud()
    # Remove intensity channel
    pcl = pcl[:,:-1]
    pcd.points = open3d.utility.Vector3dVector(pcl)
    open3d.visualization.draw_geometries([pcd])

    if not lidar_frame_counter:
        vis.add_geometry(pcd)
    else:
        vis.clear_geometries()
        #vis.add_geometry(pcd)
        vis.update_geometry(pcd)

    vis.poll_events()
    vis.update_renderer()


    #vis.capture_screen_image('cameraparams.png')

    # image = vis.capture_screen_float_buffer()
    # Close
    time.sleep(2)
    vis.capture_screen_image(f"./lidar_images/depth_image_{lidar_frame_counter}.png")

    #image = vis.capture_screen_float_buffer()

    #image = vis.capture_screen_float_buffer(False)
    #image = np.asarray(image) * 256
    #image = image.astype(np.uint8)

    #image = cv2.resize(image,(264,264))
    #image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    #print(image.shape)
    #cv2.imwrite("test.png",image)
    #cv2.imshow("test", image)
    #if cv2.waitKey(10) & 0xFF == 27:
    #    break
    #plt.imshow(image)

    #print(type(image))
    #plt.savefig(f"./lidar_images/depth_image_{lidar_frame_counter}.png")
    #vis.destroy_window()


    lidar_frame_counter+=1

def draw_1D_map(custom_map, name):
    custom_map = custom_map * 256
    custom_map = custom_map.astype(np.uint8)
    while (1):
        cv2.imshow(name, custom_map)
        if cv2.waitKey(10) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

def crop_channel_azimuth(img_channel, division_factor):
    opening_angle = int(img_channel.shape[1] / division_factor)
    img_channel_center = int(img_channel.shape[1] / 2)
    img_channel = img_channel[:, img_channel_center - opening_angle : img_channel_center + opening_angle]
    return img_channel


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

def contrast_adjustment(img):
    return np.amax(img)/2 * img * 255 / (np.amax(img) - np.amin(img))

def map_to_8bit(range_image, channel):
    img_channel = range_image[:,:,channel]

    if channel == RangeImgChannel.Range.value:
        img_channel = img_channel * 255 / (np.amax(img_channel) - np.amin(img_channel))

    elif channel == RangeImgChannel.Intensity.value:
        img_channel = contrast_adjustment(img_channel)

    img_channel = img_channel.astype(np.uint8)
    return img_channel

def get_selected_channel(frame, lidar_name, channel, crop_azimuth=True):
    range_image = load_range_image(frame, lidar_name)
    range_image[range_image<0] = 0.0

    img_selected = map_to_8bit(range_image, channel = channel.value)
    if crop_azimuth:
        img_selected = crop_channel_azimuth(img_selected, division_factor=8)
    return img_selected

def show_range_image(frame, lidar_name, crop_azimuth=True):
    print("student task ID_S1_EX1")
    img_channel_range = get_selected_channel(frame, lidar_name, RangeImgChannel.Range, crop_azimuth)
    img_channel_intensity = get_selected_channel(frame, lidar_name, RangeImgChannel.Intensity, crop_azimuth)
    img_range_intensity = np.vstack([img_channel_range, img_channel_intensity])
    return img_range_intensity

def crop_point_cloud(lidar_pcl, config):
    lim_x = config.lim_x
    lim_y = config.lim_y
    lim_z = config.lim_z

    mask = np.where((lidar_pcl[:, 0] >= lim_x[0]) & (lidar_pcl[:, 0] <= lim_x[1]) &
                    (lidar_pcl[:, 1] >= lim_y[0]) & (lidar_pcl[:, 1] <= lim_y[1]) &
                    (lidar_pcl[:, 2] >= lim_z[0]) & (lidar_pcl[:, 2] <= lim_z[1]))

    lidar_pcl = lidar_pcl[mask]

    return lidar_pcl

# create birds-eye view of lidar data

def discretize_for_bev(lidar_pcl, configs):
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    lidar_pcl_cpy = np.copy(lidar_pcl)
    # remove lidar points outside detection area and with too low reflectivity
    lidar_pcl_cpy = crop_point_cloud(lidar_pcl_cpy, configs)

    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))
    # Make sure that no negative bev-coordinates occur by adding half of the width
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]

    return lidar_pcl_cpy

def get_sorted_lidar_pcl_according_to_dim(lidar_pcl_cpy, configs, dim2sort = None):
    lidar_pcl_cpy[lidar_pcl_cpy[:, 3] > 1.0, 3] = 1.0
    indices = np.lexsort((-lidar_pcl_cpy[:, dim2sort], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))

    lidar_pcl_cpy = lidar_pcl_cpy[indices]
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_top_sorted = lidar_pcl_cpy[indices]

    return lidar_top_sorted

def get_intensity_map_from_pcl(lidar_pcl_cpy, configs):
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    lidar_pcl_int = get_sorted_lidar_pcl_according_to_dim(lidar_pcl_cpy, configs, dim2sort = 3)

    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = \
        lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:, 3]) - np.amin(lidar_pcl_int[:, 3]))

    return intensity_map


def get_height_map_from_pcl(lidar_pcl_cpy, configs):

    lidar_pcl_height = get_sorted_lidar_pcl_according_to_dim(lidar_pcl_cpy, configs, dim2sort = 2)

    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

    height_map[np.int_(lidar_pcl_height[:, 0]), np.int_(lidar_pcl_height[:, 1])] = \
        lidar_pcl_height[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))

    return height_map

def get_density_map_from_pcl(lidar_pcl_cpy, configs):
    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

    lidar_pcl_top = get_sorted_lidar_pcl_according_to_dim(lidar_pcl_cpy, configs, dim2sort = 2)
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts

    return density_map


def assemble_bev_from_maps(density_map, intensity_map, height_map, configs):
    # assemble 3-channel bev-map from individual maps
    bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
    bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

    # expand dimension of bev_map before converting into a tensor
    s1, s2, s3 = bev_map.shape
    bev_maps = np.zeros((1, s1, s2, s3))
    bev_maps[0] = bev_map

    bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
    input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

    return input_bev_maps

def bev_from_pcl(lidar_pcl, configs, vis=False):
    ####### ID_S2_EX1 START #######
    print("student task ID_S2_EX1")
    lidar_pcl_cpy = discretize_for_bev(lidar_pcl, configs)
    ####### ID_S2_EX1 END #######
    ####### ID_S2_EX2 START #######
    print("student task ID_S2_EX2")
    intensity_map = get_intensity_map_from_pcl(lidar_pcl_cpy, configs)
    if vis:
        draw_1D_map(intensity_map, "intensity_map")
    ####### ID_S2_EX2 END #######
    ####### ID_S2_EX3 START #######
    print("student task ID_S2_EX3")
    height_map = get_height_map_from_pcl(lidar_pcl_cpy, configs)
    if vis:
        draw_1D_map(height_map, "height_map")
    ####### ID_S2_EX3 END #######
    density_map = get_density_map_from_pcl(lidar_pcl_cpy, configs)

    # Assemble BEV from maps
    input_bev_maps = assemble_bev_from_maps(density_map, intensity_map, height_map, configs)

    return input_bev_maps


