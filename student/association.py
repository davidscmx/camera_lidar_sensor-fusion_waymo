# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import misc.params as params

class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self):
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_measurements = []

    def associate(self, track_list, meas_list, KF):

        ############
        # TODO Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############

        # the following only works for at most one track and one measurement
        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements
        self.unassigned_tracks = list(range(N))
        self.unassigned_measurements = list(range(M))

        # initialize association matrix
        self.association_matrix = np.inf*np.ones((N, M))

        # loop over all tracks and all measurements to set up association matrix
        for i in range(N):
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                dist = self.MHD(track, meas, KF)

                if self.gating(dist, meas.sensor):
                    self.association_matrix[i,j] = dist

    def get_closest_track_and_meas(self):
        ############
        # TODO Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_measurements
        # - return this track and measurement
        ############
        A = self.association_matrix
        if np.min(self.association_matrix) == np.inf:
            return np.nan, np.nan

         # get indices of minimum entry
        ij_min = np.unravel_index(np.argmin(A, axis=None), A.shape)

        track_idx = ij_min[0]
        measurement_idx = ij_min[1]

         # delete row and column for next update
        A = np.delete(A, track_idx, 0)
        A = np.delete(A, measurement_idx, 1)
        self.association_matrix = A

        # the following only works for at most one track and one measurement
        update_track = self.unassigned_tracks[track_idx]
        update_meas = self.unassigned_measurements[measurement_idx]

        # remove from list
        self.unassigned_tracks.remove(update_track)
        self.unassigned_measurements.remove(update_meas)

        return update_track, update_meas

    def gating(self, mdist, sensor):
        limit = chi2.ppf(params.gating_threshold, sensor.dim_meas)
        if mdist < limit:
            return True
        else:
            return False

    def MHD(self, track, meas, KF):
        H = meas.sensor.get_H(track.x)
        S = KF.S(track, meas, H)
        gamma = KF.gamma(track, meas)
        MHD = gamma.transpose() * np.linalg.inv(S) * gamma

        return MHD

    def associate_and_update(self, manager, meas_list, KF):
        self.associate(manager.track_list, meas_list, KF)

        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:

            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]

            # check visibility, only update tracks in fov
            if not meas_list[0].sensor.in_fov(track.x):
                continue

            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])

            # update score and track state
            manager.handle_updated_track(track)

            # save updated track
            manager.track_list[ind_track] = track

        # run track management
        manager.set_unassigned_tracks(self.unassigned_tracks)
        manager.set_unassigned_measurements(self.unassigned_measurements)
        manager.set_measurements(meas_list)
        manager.manage_tracks()

        for track in manager.track_list:
            print('track', track.id, 'score =', track.score)