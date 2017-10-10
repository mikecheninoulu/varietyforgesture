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
from numpy import log
from shutil import copyfile
import matplotlib.pyplot as plt
import cv2
from ChalearnLAPEvaluation import evalGesture,exportGT_Gesture
from ChalearnLAPSample import GestureSample
from utils import IsLeftDominant
from utils import Extract_feature_Realtime
from utils import Extract_feature_UNnormalized
from utils import normalize
from utils import imdisplay
from utils import viterbi_colab_clean
from utils import createSubmisionFile
import time
import cPickle
import numpy
import pickle
import scipy.io as sio  


myList = [1, 8, 3, 4, 5]
a = sorted(range(len(myList)),key=lambda x:myList[x])
print a 