# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------

# imports
import numpy as np
import collections

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params

class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, measurement, id):
        print('creating track no.', id)
        M_rot = measurement.sensor.sensor_to_vehicle[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates

        ############
        #TODO Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        # transform measurement to vehicle coordinates
        # homogeneous coordinates
        position_sensor = np.ones((4, 1))
        position_sensor[0:3] = measurement.z[0:3]
        position_vehicle = measurement.sensor.sensor_to_vehicle * position_sensor

        # save initial state from measurement
        self.x = np.zeros((6,1))
        self.x[0:3] = position_vehicle[0:3]
        # set up position estimation error covariance
        P_pos = M_rot * measurement.R * np.transpose(M_rot)

        # set up velocity estimation error covariance
        sigma_p44 = 50 # initial setting for estimation error covariance P entry for vx
        sigma_p55 = 50 # initial setting for estimation error covariance P entry for vy
        sigma_p66 = 5 # initial setting for estimation error covariance P entry for vz

        P_vel = np.matrix([[sigma_p44**2, 0, 0],
                        [0, sigma_p55**2, 0],
                        [0, 0, sigma_p66**2]])

        # overall covariance initialization
        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_vel

        self.state = 'confirmed'
        self.score = 0

        ############
        # END student code
        ############

        # other track attributes
        self.id = id
        self.width = measurement.width
        self.length = measurement.length
        self.height = measurement.height
        # transform rotation from sensor to vehicle coordinates
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(measurement.yaw) + M_rot[0,1]*np.sin(measurement.yaw))
        self.t = measurement.t

    def set_x(self, x):
        self.x = x

    def set_P(self, P):
        self.P = P

    def set_t(self, t):
        self.t = t

    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if measurement.sensor.name == 'lidar':
            c = params.weight_dim
            self.width = c*measurement.width + (1 - c)*self.width
            self.length = c*measurement.length + (1 - c)*self.length
            self.height = c*measurement.height + (1 - c)*self.height
            M_rot = measurement.sensor.sensor_to_vehicle
            self.yaw = np.arccos(M_rot[0,0]*np.cos(measurement.yaw) + M_rot[0,1]*np.sin(measurement.yaw)) # transform rotation from sensor to vehicle coordinate


class Trackmanagement:
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        self.N = 0 # current number of tracks
        self.track_list = []
        self.last_id = -1
        self.result_list = []

    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):
        ############
        # TODO Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############

        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility
            if meas_list: # if not empty
                if meas_list[0].sensor.in_fov(track.x):
                    # your code goes here
                    pass

        # delete old tracks

        # initialize new track with unassigned measurement
        for j in unassigned_meas:
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])

    def addTrackToList(self, track):
        self.track_list.append(track)
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1)
        self.addTrackToList(track)

    def delete_track(self, track):
        print('deleting track no.', track.id)
        self.track_list.remove(track)

    def handle_updated_track(self, track):
        ############
        # TODO Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        pass