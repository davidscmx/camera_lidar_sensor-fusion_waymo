
from student.filter import Filter
from student.trackmanagement import Trackmanagement
from student.association import Association
from student.measurements import Sensor, Measurement
from misc.evaluation import plot_tracks, plot_rmse, make_movie
import misc.params as params

import subprocess

import numpy as np
from config import *
import copy

import misc.objdet_tools as tools
from misc.helpers import load_object_from_file
from easydict import EasyDict as edict

data_filename = '/media/ofli/Intenso/home/waymo_dataset/training/training_segment-10072231702153043603_5725_000_5745_000_with_camera_labels.tfrecord' # Sequence 2
# data_filename = 'training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels.tfrecord' # Sequence 3


# True = use groundtruth labels as objects, False = use model-based detection
configs_det.use_labels_as_objects = False
configs_det.save_results = False
configs_det.lim_y = [-5, 15]

KF = Filter() # set up Kalman filter
association = Association() # init data association
manager = Trackmanagement() # init track manager
lidar = None# init lidar sensor object
camera = None # init camera sensor object
np.random.seed(10) # make random values pre dictable

vis_pause_time = 0

##################
## Perform detection & tracking over all selected frames
all_labels = []
det_performance_all = []
np.random.seed(0) # make random values predictable

config = edict()
config.lim_x = [0, 50]
config.lim_y = [-25, 25]
config.lim_z = [-1, 3]
config.bev_width = 608
config.bev_height = 608
config.conf_thresh = 0.5
config.model = 'darknet'

show_only_frames = [65, 100] # show only frames in interval for debugging

for cnt_frame in range(show_only_frames[0], show_only_frames[1]):
    try:
        ## Get next frame from Waymo dataset
        frame = next(datafile_iter)
        print('------------------------------')
        print('processing frame #' + str(cnt_frame))

        # Extract calibration data and front camera image from frame
        lidar_name = dataset_pb2.LaserName.TOP
        camera_name = dataset_pb2.CameraName.FRONT
        lidar_calibration = waymo_utils.get(frame.context.laser_calibrations, lidar_name)
        camera_calibration = waymo_utils.get(frame.context.camera_calibrations, camera_name)
        image = tools.extract_front_camera_image(frame)

        lidar_pcl = load_object_from_file(results_fullpath, data_filename, 'lidar_pcl', cnt_frame)
        detections = load_object_from_file(results_fullpath, data_filename, 'detections', cnt_frame)
        valid_label_flags = tools.validate_object_labels(frame.laser_labels, lidar_pcl, configs_det, 0 if configs_det.use_labels_as_objects==True else 10)

        # set up sensor objects
        if lidar is None:
            lidar = Sensor('lidar', lidar_calibration)
        if camera is None:
            camera = Sensor('camera', camera_calibration)

        # preprocess lidar detections
        meas_list_lidar = []

        for detection in detections:
            # check if measurement lies inside specified range
            if detection[1] > configs_det.lim_x[0] and detection[1] < configs_det.lim_x[1] and detection[2] > configs_det.lim_y[0] and detection[2] < configs_det.lim_y[1]:
                meas_list_lidar = lidar.generate_measurement(cnt_frame, detection[1:], meas_list_lidar)

        # preprocess camera detections
        meas_list_cam = []

        for label in frame.camera_labels[0].labels:
            if(label.type == label_pb2.Label.Type.TYPE_VEHICLE):
                box = label.box
                # use camera labels as measurements and add some random noise
                z = [box.center_x, box.center_y, box.width, box.length]
                z[0] = z[0] + np.random.normal(0, params.sigma_cam_i)
                z[1] = z[1] + np.random.normal(0, params.sigma_cam_j)
                meas_list_cam = camera.generate_measurement(cnt_frame, z, meas_list_cam)

        # Kalman prediction
        for track in manager.track_list:
            KF.predict(track)
            track.set_t((cnt_frame - 1)*0.1) # save next timestamp

        # associate all lidar measurements to all tracks
        association.associate_and_update(manager, meas_list_lidar, KF)
        # associate all camera measurements to all tracks
        association.associate_and_update(manager, meas_list_cam, KF)

        # save results for evaluation
        result_dict = {}

        for track in manager.track_list:
            result_dict[track.id] = track

        manager.result_list.append(copy.deepcopy(result_dict))
        label_list = [frame.laser_labels, valid_label_flags]
        all_labels.append(label_list)

    except StopIteration:
        # if StopIteration is raised, break from loop
        print("StopIteration has been raised\n")
        break

def save_tracks_reference(result_list_to_save, name_of_file = "regression_files/track_reference.txt"):
    with open(name_of_file, "w") as f:
        f.write("id, width, height, length, x[0], x[1], x[2], yaw\n")
        for i, result_dict in enumerate(result_list_to_save):
            for t_id, t in result_dict.items():
                f.write(f"{t_id},")
                f.write(f"{t.width},")
                f.write(f"{t.height},")
                f.write(f"{t.length},")
                f.write(f"{t.x[0]},")
                f.write(f"{t.x[1]},")
                f.write(f"{t.x[2]},")
                f.write(f"{t.yaw}\n")

def compare_tracks_with_reference(result_list_to_save):
    save_tracks_reference(result_list_to_save, name_of_file="regression_files/track_reference_tmp.txt")
    print("Initiating regression test")

    with open('regression_files/track_reference.txt', 'r') as file1:
        with open('regression_files/track_reference_tmp.txt', 'r') as file2:
            difference = set(file1).difference(file2)

    if len(difference)>0:
        print("Files are not equal!! Regression test failed.")
    else:
        print("Files are equal. Regression test passed.")

    #subprocess.run(["rm", "regression_files/track_reference_tmp.txt"])

compare_tracks_with_reference(manager.result_list)