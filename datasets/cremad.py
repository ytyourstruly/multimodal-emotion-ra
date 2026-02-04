# -*- coding: utf-8 -*-
"""
CREMA-D Dataset Loader - Using Preprocessed Files
Uses preprocessed audio (croppad) and video (facecroppad) files
Compatible with RAVDESS format for seamless integration
"""

import torch
import torch.utils.data as data
from PIL import Image
import functools
import numpy as np
import librosa
import cv2
import os


def video_loader_npy(video_path):
    """
    Load video from preprocessed NPY file
    Args:
        video_path: path to .npy file
    Returns:
        list of PIL Images
    """
    video = np.load(video_path)
    video_data = []
    for i in range(video.shape[0]):
        video_data.append(Image.fromarray(video[i, :, :, :]))
    return video_data


def get_default_video_loader():
    return functools.partial(video_loader_npy)


def load_audio(audiofile, sr=22050):
    """Load preprocessed audio file using librosa"""
    y, _ = librosa.load(audiofile, sr=sr)
    return y, sr


def get_mfccs(y, sr, n_mfcc=10):
    """Extract MFCC features from audio"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def make_dataset(subset, annotation_path):
    """
    Create dataset from annotation file
    Uses preprocessed files (_croppad for audio, _facecroppad for video)
    
    Args:
        subset: 'training', 'validation', or 'testing'
        annotation_path: path to annotation TXT file (RAVDESS format)
        root_path: root directory of CREMA-D dataset (not used, paths in annotation are absolute)
    Returns:
        list of sample dictionaries
    """
    with open(annotation_path, 'r') as f:
        annots = f.readlines()
    
    dataset = []
    for line in annots:
        # Format: video_path;audio_path;label;split
        parts = line.strip().split(';')
        if len(parts) != 4:
            continue
        
        video_path, audio_path, label, split = parts
        
        if split.strip() != subset:
            continue
        
        # Convert to preprocessed file paths
        # video_path: .../VideoFlash/XXXX_YYY_EMO_ZZ.flv -> .../VideoFlash/XXXX_YYY_EMO_ZZ_facecroppad.npy
        # audio_path: .../AudioWAV/XXXX_YYY_EMO_ZZ.wav -> .../AudioWAV/XXXX_YYY_EMO_ZZ_croppad.wav
        
        video_preprocessed = video_path
        audio_preprocessed = audio_path
        
        # Check if preprocessed files exist
        if not os.path.exists(video_preprocessed):
            print(f"Warning: Preprocessed video not found: {video_preprocessed}")
            continue
        
        if not os.path.exists(audio_preprocessed):
            print(f"Warning: Preprocessed audio not found: {audio_preprocessed}")
            continue
        
        sample = {
            'video_path': video_preprocessed,
            'audio_path': audio_preprocessed,
            'label': int(label) - 1  # Convert to 0-indexed
            # 'gender': get_gender_from_filename(filename=os.path.basename(video_path))
        }
        dataset.append(sample)
    
    return dataset

import pandas as pd

# Load once at startup, not on every call
# demographics = pd.read_csv('/home/yeskendir/Downloads/crema-d-mirror-main/VideoDemographics.csv')
# GENDER_MAP = {
#     row['ActorID']: 0 if row['Sex'] == 'Male' else 1
#     for _, row in demographics.iterrows()
# }

# def get_gender_from_filename(filename):
#     actor_id = int(filename.split("_")[0])
#     # if actor_id == 1001:
#     #     print("Debug: ActorID is 1001")
#     return GENDER_MAP[actor_id]  # 0=male, 1=female

class CREMAD(data.Dataset):
    """
    CREMA-D Dataset for audiovisual emotion recognition
    Uses preprocessed files created by preprocessing scripts
    Compatible with RAVDESS format
    """
    
    def __init__(self,
                 annotation_path,
                 subset,
                 spatial_transform=None,
                 audio_transform=None,
                 data_type='audiovisual',
                 sr=22050,
                 n_mfcc=10,
                 get_loader=get_default_video_loader):
        """
        Args:
            annotation_path: path to annotation TXT file (RAVDESS format)
            subset: 'training', 'validation', or 'testing'
            root_path: not used (for compatibility)
            spatial_transform: transformations for video frames
            audio_transform: transformations for audio
            data_type: 'video', 'audio', or 'audiovisual'
            sr: audio sample rate
            n_mfcc: number of MFCC coefficients
            get_loader: video loader function
        """
        self.data = make_dataset(subset, annotation_path)
        self.spatial_transform = spatial_transform
        self.audio_transform = audio_transform
        self.loader = get_loader()
        self.data_type = data_type
        self.sr = sr
        self.n_mfcc = n_mfcc
        
        if len(self.data) == 0:
            print(f"Warning: No data found for subset '{subset}' in {annotation_path}")
            print(f"Make sure preprocessing has been completed!")
        else:
            print(f"Loaded {len(self.data)} samples for {subset} from CREMA-D")

    def __getitem__(self, index):
        """Get a sample from the dataset"""
        target = self.data[index]['label']
        # gender = self.data[index]['gender']
        # gender = torch.tensor(gender, dtype=torch.long)
        # Load video data
        if self.data_type == 'video' or self.data_type == 'audiovisual':
            path = self.data[index]['video_path']
            
            try:
                clip = self.loader(path)
                
                if self.spatial_transform is not None:
                    self.spatial_transform.randomize_parameters()
                    clip = [self.spatial_transform(img) for img in clip]
                
                clip = torch.stack(clip, 0).permute(1, 0, 2, 3)  # (C, T, H, W)
            
            except Exception as e:
                print(f"Error loading video {path}: {e}")
                # Return dummy video
                clip = torch.zeros(3, 15, 224, 224)
            
            if self.data_type == 'video':
                return clip, target
        
        # Load audio data
        if self.data_type == 'audio' or self.data_type == 'audiovisual':
            path = self.data[index]['audio_path']
            
            try:
                y, sr = load_audio(path, sr=self.sr)
                
                if self.audio_transform is not None:
                    self.audio_transform.randomize_parameters()
                    y = self.audio_transform(y)
                
                mfcc = get_mfccs(y, sr, n_mfcc=self.n_mfcc)
                audio_features = mfcc
            
            except Exception as e:
                print(f"Error loading audio {path}: {e}")
                # Return dummy audio features
                audio_features = np.zeros((self.n_mfcc, 100))
            
            if self.data_type == 'audio':
                return audio_features, target
        
        # Return audiovisual data
        if self.data_type == 'audiovisual':
            return audio_features, clip, target #  gender

    def __len__(self):
        return len(self.data)