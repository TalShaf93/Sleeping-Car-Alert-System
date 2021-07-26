##########################################################
#                     Imported Libraries                 #
##########################################################
import requests
import cv2
import numpy as np
from scipy.spatial import distance

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from playsound import playsound
import argparse
import imutils
import time
import dlib
import cv2
import os
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class Lib:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()  # face detector
        # predictor of 68 facial features
        self.predictor = dlib.shape_predictor(
            './shape_predictor_68_face_landmarks.dat')

        df = pd.read_csv('./data.csv')   # read training data csv
        X = df.drop(columns=['Label', 'Video', 'Frame', 'EAR_N',
                    'MAR_N', 'CIR_N', 'ME_N', 'HEAD_N'])  # features
        Y = df['Label']  # labels
        self.model = RandomForestClassifier()
        self.model.fit(X, Y)

    @staticmethod
    def eye_aspect_ratio(eye):  # returns eye aspect ratio
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    @staticmethod
    def mouth_aspect_ratio(mouth):  # returns mouth aspect ratio
        A = distance.euclidean(mouth[14], mouth[18])
        C = distance.euclidean(mouth[12], mouth[16])
        mar = (A) / (C)
        return mar

    @staticmethod
    def circularity(eye):   # returns eye circularity
        A = distance.euclidean(eye[1], eye[4])
        radius = A/2.0
        Area = math.pi * (radius ** 2)
        p = 0
        for i in range(5):
            p += distance.euclidean(eye[i], eye[i+1])
            if i == 4:
                p += distance.euclidean(eye[5], eye[0])
        return 4 * math.pi * Area / (p**2)

    @staticmethod
    def mouth_over_eye(eye):    # returns ratio of mouth and eye
        ear = Lib.eye_aspect_ratio(eye)
        mar = Lib.mouth_aspect_ratio(eye)
        mouth_eye = mar/ear
        return mouth_eye

    @staticmethod
    def head_drop(nose):    # returns head tilt
        A = distance.euclidean(nose[0], nose[35])
        B = distance.euclidean(nose[1], nose[34])
        C = distance.euclidean(nose[2], nose[33])
        return (A+B+C)/3

    async def predict(self, filename):
        frame = cv2.imread(filename)
        i = 0
        probabilities = []
        if frame is not None:
            # convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            i += 1
            rects = self.detector(gray, 0)   # detect a face
            seq = 0
            for rect in rects:  # run on all faces found
                # get 68 points representing the face
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)   # convert to np array
                features = []
                nose = shape[32:68]
                eye = shape[36:68]
                ear = self.eye_aspect_ratio(eye)
                mar = self.mouth_aspect_ratio(eye)
                cir = self.circularity(eye)
                mouth_eye = self.mouth_over_eye(eye)
                head = self.head_drop(nose)
                # add this frame's 5 features to the list
                features.append([ear, mar, cir, mouth_eye, head])
                # print(features)
                predictions = self.model.predict_proba(
                    features)  # make prediction on frame
                print(predictions)
                # probability of prediction
                probabilities.append(predictions[0][0])
                if len(probabilities) == 6:  # keep 5 last frames
                    probabilities.pop(0)
                # sum the probabilities of last 5 frames
                seq = sum(probabilities)
            return seq
        return None
