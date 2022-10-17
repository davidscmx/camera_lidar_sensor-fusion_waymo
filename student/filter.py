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
        # process model dimension
        self.dim_state = params.dim_state
        # time increment
        self.dt = params.dt
        # process noise variable for Kalman filter Q
        self.q = params.q

    @property
    def F(self):
        # x' = x + xdot*dt
        # x_dot = x_dot
        dt = self.dt
        return np.matrix([[1, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, dt],
                          [0, 0, 0, 1, 0,  0],
                          [0, 0, 0, 0, 1,  0],
                          [0, 0, 0, 0, 0,  1]])

    @property
    def Q(self):
        q = self.q
        dt = self.dt
        q1 = ((dt**3)/3) * q
        q2 = ((dt**2)/2) * q
        q3 = dt * q
        return np.matrix([[q1, 0, 0, q2, 0, 0],
                          [0, q1, 0, 0, q2, 0],
                          [0,  0, q1, 0, 0, q2],
                          [q2, 0, 0, q3, 0, 0],
                          [0, q2, 0, 0, q3,0],
                          [0,  0, q2, 0, 0, q3],
                          ])

    def predict(self, track):
        x = self.F * track.x
        P = self.F * track.P * self.F.transpose() + self.Q

        track.set_x(x)
        track.set_P(P)

    def update(self, track, meas):
        H = meas.sensor.get_H(track.x)
        gamma = self.gamma(track, meas)

        S = self.S(track, meas)

        K = (track.P * H.transpose()) * np.linalg.inv(S)

        x = track.x + K*gamma
        I = np.identity(params.dim_state)
        P = (I - K*H) * track.P

        track.set_x(x)
        track.set_P(P)
        track.update_attributes(meas)

    def gamma(self, track, meas):
        hx = meas.sensor.get_hx(track.x)
        gamma = meas.z - hx
        return gamma

    def S(self, track, meas):
        H = meas.sensor.get_H(track.x)
        S =  H * track.P * H.transpose() + meas.R
        return S

