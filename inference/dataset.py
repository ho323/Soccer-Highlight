from torch.utils.data import Dataset

import numpy as np
import random
# import pandas as pd
import os
import time
import ffmpy

# TF2 GPU memory growth 설정 (전체 VRAM 점유 방지)
try:
    import tensorflow as tf
    for _gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass

from tqdm import tqdm
# import utils

import torch

import logging
import json

from SoccerNet.Downloader import SoccerNetDownloader
from Features.VideoFeatureExtractor import VideoFeatureExtractor, PCAReducer

# 스크립트 파일 위치 기준 경로 설정
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SoccerNetClipsTesting(Dataset):
    def __init__(self, path, features="ResNET_PCA512.npy", 
                framerate=2, chunk_size=240, receptive_field=80):
        self.path = path
        self.chunk_size = chunk_size
        self.receptive_field = receptive_field
        self.framerate = framerate
        self.num_classes = 17
        self.num_detections =15

        # 경로 설정
        self.outputs_dir = os.path.join(_SCRIPT_DIR, "outputs")
        self.features_dir = os.path.join(_SCRIPT_DIR, "Features")
        os.makedirs(self.outputs_dir, exist_ok=True)

        video_lq_path = os.path.join(self.outputs_dir, "videoLQ.mkv")
        features_path = os.path.join(self.outputs_dir, "features.npy")
        features_pca_path = os.path.join(self.outputs_dir, "features_PCA.npy")
        pca_file = os.path.join(self.features_dir, "pca_512_TF2.pkl")
        scaler_file = os.path.join(self.features_dir, "average_512_TF2.pkl")

        #Changing video format to 
        ff = ffmpy.FFmpeg(
             inputs={self.path: ""},
             outputs={video_lq_path: '-y -r 25 -vf scale=-1:224 -max_muxing_queue_size 9999'})
        print(ff.cmd)
        ff.run()

        print("Initializing feature extractor")
        myFeatureExtractor = VideoFeatureExtractor(
            feature="ResNET",
            back_end="TF2",
            transform="crop",
            grabber="opencv",
            FPS=self.framerate)

        print("Extracting frames")
        myFeatureExtractor.extractFeatures(path_video_input=video_lq_path,
                                           path_features_output=features_path,
                                           overwrite=True)

        print("Initializing PCA reducer")
        myPCAReducer = PCAReducer(pca_file=pca_file,
                                  scaler_file=scaler_file)

        print("Reducing with PCA")
        myPCAReducer.reduceFeatures(input_features=features_path,
                                    output_features=features_pca_path,
                                    overwrite=True)



    def __getitem__(self, index):
        # Load features
        features_pca_path = os.path.join(self.outputs_dir, "features_PCA.npy")
        feat_half1 = np.load(features_pca_path)
        print("Shape half 1: ", feat_half1.shape)
        size = feat_half1.shape[0]

        def feats2clip(feats, stride, clip_length):

            idx = torch.arange(start=0, end=feats.shape[0]-1, step=stride)
            idxs = []
            for i in torch.arange(0, clip_length):
                idxs.append(idx+i)
            idx = torch.stack(idxs, dim=1)

            idx = idx.clamp(0, feats.shape[0]-1)
            idx[-1] = torch.arange(clip_length)+feats.shape[0]-clip_length

            return feats[idx,:]
            

        feat_half1 = feats2clip(torch.from_numpy(feat_half1), 
                        stride=self.chunk_size-self.receptive_field, 
                        clip_length=self.chunk_size)
                                  
        return feat_half1, size

    def __len__(self):
        return 1