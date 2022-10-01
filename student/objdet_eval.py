# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Evaluate performance of object detection
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import matplotlib
matplotlib.use('wxagg') # change backend so that figure maximizing works on Mac as well
import matplotlib.pyplot as plt

import torch
from shapely.geometry import Polygon
from operator import itemgetter

# add project directory to python path to enable relative imports
import os
import sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# object detection tools and helper functions
import misc.objdet_tools as tools


# compute various performance measures to assess object detection
def measure_detection_performance(detections, labels, labels_valid, min_iou=0.5):

     # find best detection for each valid label
    true_positives = 0 # no. of correctly detected objects
    center_devs = []
    ious = []
    for label, valid in zip(labels, labels_valid):
        matches_lab_det = []
        if valid:
            ####### ID_S4_EX1 START #######
            print("student task ID_S4_EX1 ")

            ## step 1 : extract the four corners of the current label bounding-box
            lbox = label.box

            label_corners = tools.compute_box_corners(lbox.center_x, lbox.center_y,
                                                      lbox.width, lbox.length, lbox.heading)
            print("label_corners ", label_corners)
            ## step 2 : loop over all detected objects (done)
            for detection in detections:
                ## step 3 : extract the four corners of the current detection
                # Ignore height
                class_id, x, y, z, _, w, l, yaw = detection
                #print(class_id, x, y, z, _, w, l, yaw)
                # Set yaw to zero
                detection_corners = tools.compute_box_corners(x, y, w, l, 0)
                print("Detection corners ", detection_corners)
                ## step 4 : computer the center distance between label and detection bounding-box in x, y, and z
                distance = np.array([lbox.center_x, lbox.center_y, lbox.center_z]) - np.array([x, y, z])
                ## step 5 : compute the intersection over union (IOU) between label and detection bounding-box
                label_poly = Polygon(label_corners)
                detecion_poly = Polygon(detection_corners)

                intersection = detecion_poly.intersection(label_poly)
                union = detecion_poly.union(label_poly)
                iou = intersection.area / union.area
                #print(iou)
                ## step 6 : if IOU exceeds min_iou threshold, store [iou,dist_x, dist_y, dist_z] in matches_lab_det and increase the TP count
                if iou > min_iou:
                    dist_x, dist_y, dist_z = distance
                    matches_lab_det.append([iou,dist_x, dist_y, dist_z ])
            #######
            ####### ID_S4_EX1 END #######

        # find best match and compute metrics
        if matches_lab_det:
            # retrieve entry with max iou in case of multiple candidates
            best_match = max(matches_lab_det,key=itemgetter(1))
            ious.append(best_match[0])
            center_devs.append(best_match[1:])


    ####### ID_S4_EX2 START #######
    #######
    print("student task ID_S4_EX2")

    # compute positives and negatives for precision/recall

    ## step 1 : compute the total number of positives present in the scene
    all_positives = labels_valid.sum()

    ## step 2 : compute the number of false negatives
    false_negatives = all_positives - true_positives

    ## step 3 : compute the number of false positives
    false_positives = len(detections) - true_positives

    #######
    ####### ID_S4_EX2 END #######

    pos_negs = [all_positives, true_positives, false_negatives, false_positives]
    det_performance = [ious, center_devs, pos_negs]
    print(det_performance)
    return det_performance


# evaluate object detection performance based on all frames
def compute_performance_stats(det_performance_all):
    print(det_performance_all)
    ####### ID_S4_EX3 START #######
    #######
    print('student task ID_S4_EX3')

    # extract the total number of positives, true positives, false negatives and false positives
    # format of det_performance_all is [ious, center_devs, pos_negs]
    # extract elements

    ious = []
    center_devs = []
    pos_negs = []
    for item in det_performance_all:
        ious.append(item[0])
        center_devs.append(item[1])
        pos_negs.append(item[2])
        pos_negs_arr = np.asarray(pos_negs)

    print(ious)
    print(center_devs)
    positives = sum(pos_negs_arr[:,0])
    true_positives = sum(pos_negs_arr[:,1])
    false_negatives = sum(pos_negs_arr[:,2])
    false_positives = sum(pos_negs_arr[:,3])
    print("TP = " + str(true_positives) + ", FP = " + str(false_positives) + ", FN = " + str(false_negatives))

    # compute precision
    # When an object is detected, what are the chances of it being real?
    precision = true_positives / (true_positives + false_positives)

    # compute recall
    # What are the chances of a real object being detected?
    recall = true_positives / (true_positives + false_negatives)

    print("precision = " + str(precision) + ", recall = " + str(recall) )

    #######
    ####### ID_S4_EX3 END #######
    print('precision = ' + str(precision) + ", recall = " + str(recall))

    # serialize intersection-over-union and deviations in x,y,z
    ious_all = [element for tupl in ious for element in tupl]
    devs_x_all = []
    devs_y_all = []
    devs_z_all = []
    for tuple in center_devs:
        for elem in tuple:
            dev_x, dev_y, dev_z = elem
            devs_x_all.append(dev_x)
            devs_y_all.append(dev_y)
            devs_z_all.append(dev_z)


    # compute statistics
    stdev__ious = np.std(ious_all)
    mean__ious = np.mean(ious_all)

    stdev__devx = np.std(devs_x_all)
    mean__devx = np.mean(devs_x_all)

    stdev__devy = np.std(devs_y_all)
    mean__devy = np.mean(devs_y_all)

    stdev__devz = np.std(devs_z_all)
    mean__devz = np.mean(devs_z_all)
    #std_dev_x = np.std(devs_x)

    # plot results
    data = [precision, recall, ious_all, devs_x_all, devs_y_all, devs_z_all]
    titles = ['detection precision', 'detection recall', 'intersection over union', 'position errors in X', 'position errors in Y', 'position error in Z']
    textboxes = ['', '', '',
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_x_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_x_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_y_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_y_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), ))),
                 '\n'.join((r'$\mathrm{mean}=%.4f$' % (np.mean(devs_z_all), ), r'$\mathrm{sigma}=%.4f$' % (np.std(devs_z_all), ), r'$\mathrm{n}=%.0f$' % (len(devs_x_all), )))]

    f, a = plt.subplots(2, 3)
    a = a.ravel()
    num_bins = 20
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    for idx, ax in enumerate(a):
        ax.hist(data[idx], num_bins)
        ax.set_title(titles[idx])
        if textboxes[idx]:
            ax.text(0.05, 0.95, textboxes[idx], transform=ax.transAxes, fontsize=10,
                    verticalalignment='top', bbox=props)
    plt.tight_layout()
    plt.show()

