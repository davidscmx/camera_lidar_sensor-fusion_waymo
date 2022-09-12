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

# add project directory to python path to enable relative imports
import os
import sys
from enum import Enum
import zlib

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2

# object detection tools and helper functions
import misc.objdet_tools as tools

class RANGE_IMAGE_CELL_CHANNELS(Enum):
    RANGE = 0
    INTENSITY = 1
    ELONGATION = 2
    IS_IN_NO_LABEL_ZONE = 3

# visualize lidar point-cloud
def show_pcl(pcl):

    print("student task ID_S1_EX2")
    # step 1 : initialize open3d with key callback and create window
    vis = open3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    # step 2 : create instance of open3d point-cloud class
    pcd = open3d.geometry.PointCloud()
    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = open3d.utility.Vector3dVector(pcl)
    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    vis.add_geometry(pcd)
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)

    #######
    ####### ID_S1_EX2 END #######

def crop_channel_azimuth(img_channel, division_factor):
    opening_angle = int(img_channel.shape[1] / division_factor)
    img_channel_center = int(img_channel.shape[1] / 2)
    img_channel = img_channel[:, img_channel_center - opening_angle : img_channel_center + opening_angle]
    return img_channel

# visualize range image
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

    if channel == RANGE_IMAGE_CELL_CHANNELS.RANGE.value:
        img_channel = img_channel * 255 / (np.amax(img_channel) - np.amin(img_channel))
    elif channel == RANGE_IMAGE_CELL_CHANNELS.INTENSITY.value:
        img_channel = contrast_adjustment(img_channel)

    img_channel = img_channel.astype(np.uint8)
    return img_channel

def get_selected_channel(frame, lidar_name, channel):
    range_image = load_range_image(frame, lidar_name)
    range_image[range_image<0] = 0.0

    img_selected = map_to_8bit(range_image, channel = channel.value)
    img_selected = crop_channel_azimuth(img_selected, division_factor=8)
    return img_selected

def show_range_image(frame, lidar_name):

    ####### ID_S1_EX1 START #######
    print("student task ID_S1_EX1")
    img_channel_range = get_selected_channel(frame, lidar_name, RANGE_IMAGE_CELL_CHANNELS.RANGE)
    img_channel_intensity = get_selected_channel(frame, lidar_name, RANGE_IMAGE_CELL_CHANNELS.INTENSITY)
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
def bev_from_pcl(lidar_pcl, configs):

    # remove lidar points outside detection area and with too low reflectivity
    lidar_pcl = crop_point_cloud(lidar_pcl, configs)
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]

    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######
    #######
    print("student task ID_S2_EX1")

    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur

     # step 4 : visualize point-cloud using the function show_pcl from a previous task

    #######
    ####### ID_S2_EX1 END #######


    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######
    #######
    print("student task ID_S2_EX2")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map

    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)

    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task

    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud

    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background

    #######
    ####### ID_S2_EX2 END #######

    #TODO
    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######
    #######
    print("student task ID_S2_EX3")

    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map

    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map

    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background

    #######
    ####### ID_S2_EX3 END #######

    #TODO remove after implementing all of the above steps
    lidar_pcl_cpy = []
    lidar_pcl_top = []
    height_map = []
    intensity_map = []

    # Compute density layer of the BEV map
    density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts

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


