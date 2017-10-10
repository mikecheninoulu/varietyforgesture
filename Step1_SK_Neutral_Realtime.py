#-------------------------------------------------------------------------------
# Name:        Starting Kit for ChaLearn LAP 2014 Track3
# Purpose:     Show basic functionality of provided code
#
# Author:      Xavier Baro
# Author:      Di Wu: stevenwudi@gmail.com
# Created:     24/03/2014
# Copyright:   (c) Chalearn LAP 2014
# Licence:     GPL3
#-------------------------------------------------------------------------------
import sys, os,random,numpy,zipfile
from shutil import copyfile
import matplotlib.pyplot as plt

from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_Realtime
from utils import Extract_feature_UNnormalized


# Data folder (Training data);
print("Extracting the training files")
data=os.path.join("/research/imag/personal/haoyu/finaldata/Train")  
# Get the list of training samples
samples=os.listdir(data)
used_joints = ['ElbowLeft', 'WristLeft', 'ShoulderLeft','HandLeft',
                'ElbowRight', 'WristRight','ShoulderRight','HandRight',
                'Head','SpineMid','SpineBase']
njoints = len(used_joints)
STATE_NO = 10
count = 0

class_count = 7

# pre-allocating the memory
Feature_all =  numpy.zeros(shape=(200000, (njoints*(njoints-1)/2 + njoints**2)*3),dtype=numpy.float32)
Targets = numpy.zeros( shape=(200000, STATE_NO*class_count+1), dtype=numpy.uint8)

# Access to each sample
for file_count, file in enumerate(sorted(samples)):
    #if not file.endswith(".zip"):
    #    continue;        
    if file_count<650: 
        print("\t Processing file " + file)
        # Create the object to access the sample
        smp=GestureSample(os.path.join(data,file))
        # ###############################################
        # USE Ground Truth information to learn the model
        # ###############################################
        # Get the list of actions for this frame
        gesturesList=smp.getGestures()
        # Iterate for each action in this sample
        # Then we also choose 5 frame before and after the ground true data:
        seg_length = 5
        for i in range(len(gesturesList) - 2):
            
                
            # Get the gesture ID, and start and end frames for the gesture
            gestureID1, startFrame1, endFrame1 = gesturesList[i + 1]
            # This part is to extract action data

            # Get the gesture ID, and start and end frames for the gesture
            gestureID2, startFrame2, endFrame2 = gesturesList[i + 2]
            # This part is to extract action data
  
            Skeleton_matrix = numpy.zeros(shape=(5, len(used_joints)*3))
            HipCentre_matrix = numpy.zeros(shape=(5, 3))
            frame_num = 0 
            if startFrame2 - endFrame1 > 5:
                gestureID = gestureID1
                startFrame = startFrame1
                endFrame = endFrame1       
                ## extract first 5 frames
                if startFrame-seg_length > 0:
                    Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, startFrame-seg_length+1, startFrame)              
                    if not valid_skel:
                        print "No detected Skeleton of the gesture: ", gestureID
                    else:
                        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)
                        begin_frame = count
                        end_frame = count+seg_length-1
                        Feature_all[begin_frame:end_frame,:] = Feature
                        Targets[begin_frame:end_frame, -1] = 1
                        count=count+seg_length-1
#                         print(end_frame)

                ## extract last 5 frames
                if endFrame+seg_length < smp.getNumFrames():
                    Skeleton_matrix, valid_skel = Extract_feature_UNnormalized(smp, used_joints, endFrame, endFrame+seg_length-1)              
                    if not valid_skel:
                        print "No detected Skeletonof the gesture: ", gestureID
                    else:
                        Feature = Extract_feature_Realtime(Skeleton_matrix, njoints)
                        begin_frame = count
                        end_frame = count+seg_length-1
                        Feature_all[begin_frame:end_frame,:] = Feature
                        Targets[begin_frame:end_frame, -1] = 1
                        count=count+seg_length-1
#                         print(end_frame)
        # ###############################################
        del smp

# save the skeleton file:
Feature_all_new = Feature_all[0:end_frame, :]
Targets_all_new = Targets[0:end_frame, :]
import cPickle as pickle
Feature_train = { "Feature_all_neutral": Feature_all_new, "Targets_all_new": Targets_all_new }
pickle.dump( Feature_train, open( "Feature_all_neutral_realtime.pkl", "wb" ) )

import scipy.io as sio
sio.savemat('Feature_all_neutral_realtime.mat', Feature_train)









