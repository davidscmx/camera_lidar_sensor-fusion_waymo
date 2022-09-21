
Knowledge
ASK A MENTOR
ACTIVITY
Search for your question…


0

ID_S3_EX2 display bb
Profile image
Atanasko Boris M
6 months ago
Hi,

I try to display bb but I do not understand why my detection list is empty and no bb are displayed. When I look into the threshold value in the detection array after "post_processing" they are bellow 0.5 (configs.conf_thresh = 0.5) and the detections[0][1] for example is empty. Any advice how to proceed here?

Do I correctly implement following steps for conversion using the limits for x, y and z set in the configs structure?

Thanks in advance,

Atanasko

Github or Project Link
Self Driving Car Engineer
Midterm: 3D Object Detection
1 answer

1


Accepted Answer
Dear Atanasko,

Thanks for your query.

I believe you are missing the required checks at the time of performing the conversion step, please refer to sample code below:

    ####### ID_S3_EX2 START #######
    #######
    # Extract 3d bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = []
    ## step 1 : check whether there are any detections
    for obj in detections:
        id, bev_x, bev_y, z, h, bev_w, bev_l, yaw = obj

        ## step 2 : loop over all detections
        x = bev_y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
        y = bev_x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0]) - (configs.lim_y[1] - configs.lim_y[0])/2.0
        w = bev_w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
        l = bev_l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])

            ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
        if ((x >= configs.lim_x[0]) and (x <= configs.lim_x[1])
            and (y >= configs.lim_y[0]) and (y <= configs.lim_y[1])
            and (z >= configs.lim_z[0]) and (z <= configs.lim_z[1])):

            ## step 4 : append the current object to the 'objects' array
            objects.append([1, x, y, z, h, w, l, yaw])

Thanks,

Neha

Profile image
Neha
Mentor

6 months ago
Dear Neha,

Thank you for your support. I do changes in the code but I still do not see the BB in the displayed images. Any idea what is a problem in my code?

Thanks in advance,

Atanasko

Profile image
Atanasko Boris M
6 months ago
Dear Atanasko,

Please share screenshot of the outcome you are getting.

Thanks,

Neha

Profile image
Neha
6 months ago
Dear Neha,

I'm sharing the screenshot

bb_display.png
Profile image
Atanasko Boris M
6 months ago
Dear Atanasko,

There are issues in objdet_detect.py and objdet_pcl.py as well .

Please follow below changes to fix it.

objdet_detect.py

objects = []
    ## step 1 : check whether there are any detections
    if len(detections) > 0:
        ## step 2 : loop over all detections
        for det in detections:
            id, bev_x, bev_y, z, h, bev_w, bev_l, yaw = det
            ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
            x = bev_y / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            y = bev_x / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0]) - (configs.lim_y[1] - configs.lim_y[0])/2.0
            w = bev_w / configs.bev_width * (configs.lim_y[1] - configs.lim_y[0])
            l = bev_l / configs.bev_height * (configs.lim_x[1] - configs.lim_x[0])
            if ((x >= configs.lim_x[0]) and (x <= configs.lim_x[1])
                and (y >= configs.lim_y[0]) and (y <= configs.lim_y[1])
                and (z >= configs.lim_z[0]) and (z <= configs.lim_z[1])):
                    ## step 4 : append the current object to the 'objects' array
                    objects.append([1, x, y, z, h, w, l, yaw])
objdet_pcl.py, I suspect issue in bev_from_pcl function and S2_EX3 code in your github. Please find working code below for objdet_pcl.py.

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
#
# general package imports
import cv2
import numpy as np
import torch
import zlib
import open3d as o3d
# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
# waymo open dataset reader
from tools.waymo_reader.simple_waymo_open_dataset_reader import utils as waymo_utils
from tools.waymo_reader.simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
# object detection tools and helper functions
import misc.objdet_tools as tools
def key_action_callback(vis, action, mods):
    print(action)
    if action == 1:  # key down
        vis.close()
    elif action == 0:  # key up
        pass
    elif action == 2:  # key repeat
        pass
    return True
# visualize lidar point-cloud
def show_pcl(pcl):
    ####### ID_S1_EX2 START #######
    #######
    print("student task ID_S1_EX2")
    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()
    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(pcl[:, 0:3]) # need to remove the intensity first - lidar_pcl is a 4D vector with 4th dim being intensity
    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    vis.add_geometry(pcd)

    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    vis.register_key_action_callback(262, key_action_callback)
    vis.run()
    vis.destroy_window()
    #######
    ####### ID_S1_EX2 END #######

# visualize range image
def show_range_image(frame, lidar_name):
    ####### ID_S1_EX1 START #######
    #######
    print("student task ID_S1_EX1")
    # step 1 : extract lidar data and range image for the roof-mounted lidar
    lidar = [obj for obj in frame.lasers if obj.name == lidar_name][0] # get laser data structure from frame
    ri = []
    if len(lidar.ri_return1.range_image_compressed) > 0: # use first response
        ri = dataset_pb2.MatrixFloat()
        ri.ParseFromString(zlib.decompress(lidar.ri_return1.range_image_compressed))
        ri = np.array(ri.data).reshape(ri.shape.dims)
    # step 2 : extract the range and the intensity channel from the range image
    ri_range = ri[:,:,0]
    ri_intensity = ri[:,:,1]
    # step 3 : set values <0 to zero
    ri[ri<0]=0.0
    # step 4 : map the range channel onto an 8-bit scale and make sure that the full range of values is appropriately considered
    ri_range = ri_range * 255 / (np.amax(ri_range) - np.amin(ri_range))
    img_range = ri_range.astype(np.uint8)
    # step 5 : map the intensity channel onto an 8-bit scale and normalize with the difference between the 1- and 99-percentile to mitigate the influence of outliers
    ri_intensity = np.amax(ri_intensity)/2 * ri_intensity * 255 / (np.amax(ri_intensity) - np.amin(ri_intensity))
    img_intensity = ri_intensity.astype(np.uint8)
    ## this was requested in the rubric but not in the code
    # focus on +/- 90° around the image center
    deg = int(img_range.shape[1] / 4)
    ri_center = int(img_range.shape[1]/2)
    img_range = img_range[:,ri_center-deg:ri_center+deg]
    deg = int(img_intensity.shape[1] / 4)
    ri_center = int(img_intensity.shape[1]/2)
    img_intensity = img_intensity[:,ri_center-deg:ri_center+deg]
    # step 6 : stack the range and intensity image vertically using np.vstack and convert the result to an unsigned 8-bit integer
    img_range_intensity = np.vstack( (img_range, img_intensity) ).astype(np.uint8)

    #######
    ####### ID_S1_EX1 END #######

    return img_range_intensity
# create birds-eye view of lidar data
def bev_from_pcl(lidar_pcl, configs):
    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]) &
                    (lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]
    # convert sensor coordinates to bev-map coordinates (center is bottom-middle)
    ####### ID_S2_EX1 START #######
    #######
    print("student task ID_S2_EX1")
    ## step 1 :  compute bev-map discretization by dividing x-range by the bev-image height (see configs)
    bev_discret = (configs.lim_x[1] - configs.lim_x[0]) / configs.bev_height
    ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    lidar_pcl_cpy = np.copy(lidar_pcl)
    lidar_pcl_cpy[:, 0] = np.int_(np.floor(lidar_pcl_cpy[:, 0] / bev_discret))
    # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
    lidar_pcl_cpy[:, 1] = np.int_(np.floor(lidar_pcl_cpy[:, 1] / bev_discret) + (configs.bev_width + 1) / 2)
    lidar_pcl_cpy[:, 2] = lidar_pcl_cpy[:, 2] - configs.lim_z[0]
    # step 4 : visualize point-cloud using the function show_pcl from a previous task
    # show_pcl(lidar_pcl_cpy)

    #######
    ####### ID_S2_EX1 END #######


    # Compute intensity layer of the BEV map
    ####### ID_S2_EX2 START #######
    #######
    print("student task ID_S2_EX2")
    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
    lidar_pcl_cpy[lidar_pcl_cpy[:,3]>1.0,3] = 1.0
    idx_intensity = np.lexsort((-lidar_pcl_cpy[:, 3], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_cpy = lidar_pcl_cpy[idx_intensity]
    ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
    _, indices = np.unique(lidar_pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = lidar_pcl_cpy[indices]
    ## step 4 : assign the intensity value of each unique entry in lidar_top_pcl to the intensity map
    ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible
    ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = \
                lidar_pcl_int[:, 3] / (np.amax(lidar_pcl_int[:, 3])-np.amin(lidar_pcl_int[:, 3]))
    ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    img_intensity = intensity_map * 256
    img_intensity = img_intensity.astype(np.uint8)
    # while (1):
    #     cv2.imshow('img_intensity', img_intensity)
    #     if cv2.waitKey(10) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX2 END #######
    # Compute height layer of the BEV map
    ####### ID_S2_EX3 START #######
    #######
    print("student task ID_S2_EX3")
    ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map
    ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
    idx_height = np.lexsort((-lidar_pcl_cpy[:, 2], lidar_pcl_cpy[:, 1], lidar_pcl_cpy[:, 0]))
    lidar_pcl_top = lidar_pcl_cpy[idx_height] # this has the highest point for each x, y coordinate
    _, idx_height_unique = np.unique(lidar_pcl_top[:, 0:2], axis=0, return_index=True)
    lidar_pcl_top = lidar_pcl_top[idx_height_unique]
    height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
    height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))
    ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
    img_height = height_map * 256
    img_height = img_height.astype(np.uint8)
    # while (1):
    #     cv2.imshow('img_height', img_height)
    #     if cv2.waitKey(10) & 0xFF == 27:
    #         break
    # cv2.destroyAllWindows()
    #######
    ####### ID_S2_EX3 END #######
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

