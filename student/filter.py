# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
import misc.params as params

class Filter:
    '''Kalman filter class'''
    def __init__(self):
        pass

    @property
    def F(self):
        # x' = x + xdot*dt
        # x_dot = x_dot
         return np.matrix([[1, 0, 0, dt, 0, 0],
                           [0, 1, 0, 0, dt, 0],
                           [0, 0, 1, 0, 0, dt],
                           [0, 0, 0, 1, 0,  0],
                           [0, 0, 0, 0, 1,  0],
                           [0, 0, 0, 0, 0,  1]])

    @property
    def Q(self):
        return np.matrix([[0, 0],
                        [0, 0]])

    def predict(self, track):
        x = self.F @ track.x
        P = self.F @ P @ self.F.transpose() + self.Q
        track.set_x(x)
        track.set_P(P)

    def update(self, track, meas, P):
        H = meas.sensor.get_H()
        gamma = self.gamma(track, meas)
        K = track.P * H.transpose() * np.linalg.inv(S)
        x = x + K*gamma
        I = np.identity(params.dim_state)
        P = (I - K*H) * P
        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)

    def gamma(self, track, meas):
        H = meas.sensor.get_H()
        gamma = meas.z - H @ track.x
        return gamma

    def S(self, track, meas):
        H = meas.sensor.get_H()
        S =  H @ self.Q @ H.transpose() + meas.R
        return S

