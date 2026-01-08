import sys
import os
import cv2
import torch
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from PIL import Image
from facenet_pytorch import MTCNN
import tqdm
from typing import Tuple, Optional, List, Dict
import warnings
import subprocess
from torch.autograd import Variable
from scipy.ndimage import gaussian_filter1d
from torchview import draw_graph
from opts import parse_opts
opt = parse_opts()
# from moviepy.editor import VideoFileClip

# def extract_audio_without_ffmpeg(video_path):


#     audio_path = video_path + ".wav"
#     video_clip = VideoFileClip(video_path)
#     audio_clip = video_clip.audio
#     audio_clip.write_audiofile(audio_path, codec='pcm_s16le')
#     return audio_path

# Assuming 'models' module is correctly configured
from models import multimodalcnn


#############################################
# CRITICAL: EXACT TRAINING CONFIGURATION
#############################################

class TrainingConfig:
    """Configuration that MUST match training preprocessing exactly"""
    
    # Model paths
    # VIDEO_PATH = "D:/Yeskendir_files/downloads/test_vid1.mp4"
    MODEL_PATH = opt.model_path
    EFFICIENTFACE_PATH = opt.pretrain_path
    MEDIA_PATH = opt.media_path
    # MODELS_DIR = '/content/drive/MyDrive/inference_ravdess'
    
    # CRITICAL TRAINING PARAMETERS - DO NOT CHANGE
    WINDOW_SEC = 3.6        # Exact duration used in training
    FRAME_COUNT = 15        # Exact number of frames
    INPUT_FPS = 30          # Expected video FPS from training
    SR = 22050              # Audio sample rate
    N_MFCC = 10            # Number of MFCC coefficients
    NUM_EMOTIONS = 8        # Number of emotion classes
    
    # Face detection parameters from training
    MTCNN_IMAGE_SIZE = (720, 1280)  # CRITICAL: Match training exactly
    FACE_OUTPUT_SIZE = (224, 224)   # Output face crop size
    
    # Emotion labels
    EMOTIONS = ["calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
#############################################
# EXACT TRAINING PREPROCESSING FUNCTIONS
#############################################

def select_distributed(m: int, n: int) -> List[int]:
    """
    CRITICAL: This is the EXACT function used in training to select frame indices.
    Selects m frames distributed evenly across n total frames.
    
    Args:
        m: Number of frames to select (15)
        n: Total number of frames available
        
    Returns:
        List of frame indices to extract
    """
    return [i*n//m + n//(2*m) for i in range(m)]

class TrainingAlignedInference:
    """
    Inference class that EXACTLY matches the training preprocessing pipeline.
    Any deviation from training preprocessing will produce incorrect results.
    """
    
    def __init__(self, config: TrainingConfig = None):
        """Initialize with training-aligned configuration"""
        self.config = config or TrainingConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 
        # if not check_ffmpeg_installed():
        #     print("not")
        #     return 0
        # Add models directory to path
        # sys.path.append(self.config.MODELS_DIR)
        
        # CRITICAL: Initialize MTCNN with EXACT training parameters
        print(f"Initializing MTCNN with training size: {self.config.MTCNN_IMAGE_SIZE}")
        self.mtcnn = MTCNN(
            image_size=self.config.MTCNN_IMAGE_SIZE,
            device=self.device
        )
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load the trained model"""
        print("Loading model...")
        self.model = multimodalcnn.MultiModalCNN(
            self.config.NUM_EMOTIONS,
            fusion='ia',
            seq_length=self.config.FRAME_COUNT,
            pretr_ef=self.config.EFFICIENTFACE_PATH,
            num_heads=1
        )
        
        
        checkpoint = torch.load(self.config.MODEL_PATH, map_location=self.device, weights_only=False)
        if 'state_dict' in checkpoint:
            checkpoint = checkpoint['state_dict']
        # torch.save(self.model, "D:/Yeskendir_files/model.pt")
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded. Expects: {self.config.WINDOW_SEC}s @ {self.config.INPUT_FPS}fps = {self.config.FRAME_COUNT} frames")
        
    def preprocess_audio_like_training(self, media_path: str, start_sec: float = 0.0) -> np.ndarray:
        """
        Process audio EXACTLY like in training (extract_audios.py).
        Handles both audio files (.wav) and video files (.mp4, .avi).
        
        Args:
            media_path: Path to audio or video file
            start_sec: Start time in seconds
            
        Returns:
            Preprocessed audio array
        """
        # Check if input is video or audio file
        file_ext = os.path.splitext(media_path)[1].lower()
        is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if is_video:
            # For video files, librosa can extract audio directly
            print(f"Extracting audio from video: {os.path.basename(media_path)}")
            try:
                # total_duration = librosa.get_duration(path=media_path)
    
        #         if start_sec >= total_duration:
        #             print(f"Skipping {os.path.basename(media_path)}: Start time {start_sec}s exceeds file duration {total_duration}s.")
        #         # Handle this case (continue loop, return None, etc.)
        # # return None 
        #             raise ValueError("Offset out of bounds")
                # librosa handles video files automatically via ffmpeg
                y, sr = librosa.core.load(media_path, sr=self.config.SR, 
                                    offset=start_sec, 
                                    duration=self.config.WINDOW_SEC)
        #         if len(y) == 0 or np.abs(y).sum() == 0:
        #             print(f"Warning: Librosa returned silence for {media_path}. Trying fallback...")
        
        # # Fallback: Try loading the whole file without offset/duration first
        # # This often fixes FFmpeg seeking issues in video containers
        #             y_full, _ = librosa.load(media_path, sr=self.config.SR)
                    
        #             # Manually slice the array
        #             start_sample = int(start_sec * self.config.SR)
        #             end_sample = start_sample + int(self.config.WINDOW_SEC * self.config.SR)
                    
        #             if start_sample < len(y_full):
        #                 y = y_full[start_sample:end_sample]
        #             else:
        #                 y = np.array([]) # Still empty
            except Exception as e:
                print(f"Error extracting audio from video: {e}")
                print("Make sure ffmpeg is installed: pip install ffmpeg-python")
                raise
        else:
            # For audio files, load directly
            y, sr = librosa.load(media_path, sr=self.config.SR, 
                                offset=start_sec, 
                                duration=self.config.WINDOW_SEC)
        target_length = int(self.config.SR * self.config.WINDOW_SEC)
        if len(y) == 0:
            print(f"Error: Could not extract any audio from {media_path}")
            # Handle empty audio (skip or fill with zeros)
            y = np.zeros(target_length)
        
        elif len(y) < target_length:
            # Pad with zeros at the end (efficiently)
            # print(len(y))
            y = librosa.util.fix_length(y, size=target_length)
            
        elif len(y) > target_length:
            # Center Crop
            remain = len(y) - target_length
            start = remain // 2
            y = y[start : start + target_length]
        return y
    
    def extract_mfcc_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features exactly as in training.
        
        Args:
            y: Audio signal
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(y=y, sr=self.config.SR, n_mfcc=self.config.N_MFCC)
        return mfcc
    
    def preprocess_video_like_training(self, video_path: str, start_sec: float = 0.0) -> np.ndarray:
        """
        Process video EXACTLY like in training (extract_faces.py).
        
        Args:
            video_path: Path to video file
            start_sec: Start time in seconds
            
        Returns:
            Array of face crops [15, 224, 224, 3]
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Warning if FPS doesn't match training
        if abs(fps - self.config.INPUT_FPS) > 1:
            warnings.warn(f"Video FPS ({fps}) differs from training FPS ({self.config.INPUT_FPS}). "
                         f"Results may be unreliable.")
        
        # Calculate frame range for 3.6 seconds
        start_frame = int(start_sec * fps)
        total_frames_needed = int(self.config.WINDOW_SEC * fps)
        
        # Get frames to select using EXACT training distribution
        frames_to_select = select_distributed(self.config.FRAME_COUNT, total_frames_needed)
        frames_to_select = [f + start_frame for f in frames_to_select]  # Adjust for start offset
        
        # Set video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        numpy_video = []
        current_frame = start_frame
        frames_to_select_set = set(frames_to_select)
        
        print(f"Extracting frames: {frames_to_select[:3]}...{frames_to_select[-3:]}")
        
        for _ in range(total_frames_needed):
            ret, im = cap.read()
            
            if not ret:
                # If video ends, pad with black frames
                if current_frame in frames_to_select_set:
                    numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
                current_frame += 1
                continue
            
            if current_frame in frames_to_select_set:
                # Process this frame EXACTLY like training
                
                # Convert to RGB for MTCNN (training uses RGB)
                im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                im_rgb_tensor = torch.tensor(im_rgb).to(self.device)
                
                # Detect face with same MTCNN configuration
                bbox, _ = self.mtcnn.detect(im_rgb_tensor)
                
                if bbox is not None and len(bbox) > 0:
                    # Use first detected face
                    bbox = bbox[0]
                    bbox = [round(x) for x in bbox]
                    x1, y1, x2, y2 = bbox
                    
                    # Ensure valid crop coordinates
                    h, w = im.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        # Crop face from BGR image
                        face_crop = im[y1:y2, x1:x2, :]
                    else:
                        face_crop = im  # Use full frame if crop invalid
                else:
                    # No face detected, use full frame
                    face_crop = im
                
                # Resize to expected size and convert to RGB for model
                face_crop = cv2.resize(face_crop, self.config.FACE_OUTPUT_SIZE)
                face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                numpy_video.append(face_crop)
            
            current_frame += 1
        
        cap.release()
        
        # Pad with black frames if needed (same as training)
        while len(numpy_video) < self.config.FRAME_COUNT:
            numpy_video.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        # Ensure we have exactly 15 frames
        numpy_video = numpy_video[:self.config.FRAME_COUNT]
        
        return np.array(numpy_video)
    
    def inference_single_window(self, media_path: str, start_sec: float = 0.0) -> Dict:
        """
        Run inference on a single 3.6-second window.
        
        Args:
            media_path: Path to video or audio+video file
            start_sec: Start time in seconds
            
        Returns:
            Dictionary with emotion predictions
        """
        # Verify file exists
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")
        
        print(f"\nProcessing window at {start_sec}s - {start_sec + self.config.WINDOW_SEC}s")
        print(f"Media file: {os.path.basename(media_path)}")
        
        # Preprocess audio exactly like training (handles both audio and video files)
        print("Extracting and preprocessing audio...")
        audio = self.preprocess_audio_like_training(media_path, start_sec)
        mfcc = self.extract_mfcc_features(audio)
        
        # Preprocess video exactly like training
        print("Extracting and preprocessing video frames...")
        video_frames = self.preprocess_video_like_training(media_path, start_sec)
    
        print(f"Preprocessed shapes - Audio MFCC: {mfcc.shape}, Video frames: {video_frames.shape}")
        
        # Prepare tensors
        audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # 2. Prepare Visual Tensor
        # Input 'video_frames' is numpy: (15, 224, 224, 3) -> [Time, Height, Width, Channel]
        
        # Normalize to 0-1 range (Standard PyTorch practice usually matches Training DataLoaders)
        # If your training data was 0-255, remove the '/ 255.0'
        video_tensor = torch.tensor(video_frames, dtype=torch.float32).to(self.device) / 255.0
        
        # Move Channel to dim 1: 
        # (15, 224, 224, 3) -> (15, 3, 224, 224) -> [Time, Channel, Height, Width]
        video_tensor = video_tensor.permute(0, 3, 1, 2)
        
        # 3. Flatten for Model Input
        # The model expects a flattened batch of images: (Batch * Time, Channel, Height, Width)
        # Since Batch=1, Batch*Time is just Time (15).
        # Current shape is already (15, 3, 224, 224). We just need to enforce it.
        
        # Note: We SKIP the .permute(0,2,1,3,4) seen in training because our 
        # starting tensor is already in the target order (Time, Channel, H, W).
        video_tensor_flat = video_tensor

        print(f"Final Input Shapes -> Audio: {audio_tensor.shape}, Visual: {video_tensor_flat.shape}")
        
        # 4. Forward Pass
        with torch.no_grad():
            # model_graph = draw_graph(self.model, input_size=(audio_tensor.shape,video_tensor_flat.shape), device="cuda", save_graph=True, filename="graph", expand_nested=False, depth=1)
            # graph_viz_object = model_graph.visual_graph

            # graph_viz_object.render(filename='model_graph', format='png', cleanup=True)

            output = self.model(audio_tensor, video_tensor_flat)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]
        neutral = probabilities[0]
        calm = probabilities[1]
        merged_calm = neutral + calm

        # Rebuild new probability array (7 classes)
        probabilities = np.array([
            merged_calm,          # calm
            probabilities[2],     # happy
            probabilities[3],     # sad
            probabilities[4],     # angry
            probabilities[5],     # fearful
            probabilities[6],     # disgust
            probabilities[7]      # surprised
        ], dtype=np.float32)

        # Normalize (safety)
        probabilities = probabilities / probabilities.sum()
        # Handle NaN values
        if np.isnan(probabilities).any():
            print("Warning: NaN values detected in predictions, using uniform distribution")
            probabilities = np.ones(self.config.NUM_EMOTIONS) / self.config.NUM_EMOTIONS
        
        # Get results
        dominant_idx = np.argmax(probabilities)
        dominant_emotion = self.config.EMOTIONS[dominant_idx]
        confidence = probabilities[dominant_idx]
        
        return {
            'probabilities': probabilities,
            'emotion_labels': self.config.EMOTIONS,
            'dominant_emotion': dominant_emotion,
            'confidence': confidence,
            'start_time': start_sec,
            'duration': self.config.WINDOW_SEC,
            'media_file': os.path.basename(media_path)
        }
    
    # def inference_sliding_window(self, media_path: str, step_sec: float = 1.0,
    #                            plot_results: bool = True) -> Dict:
    #     """
    #     Run sliding window inference across entire video.
        
    #     Args:
    #         media_path: Path to video or audio+video file
    #         step_sec: Step size in seconds
    #         plot_results: Whether to plot results
            
    #     Returns:
    #         Dictionary with temporal emotion analysis
    #     """
    #     # Verify file exists
    #     if not os.path.exists(media_path):
    #         raise FileNotFoundError(f"Media file not found: {media_path}")
        
    #     # Get video duration
    #     cap = cv2.VideoCapture(media_path)
    #     fps = cap.get(cv2.CAP_PROP_FPS)
    #     total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #     duration_sec = total_frames / fps
    #     cap.release()
        
    #     print(f"\nMedia file info:")
    #     print(f"  File: {os.path.basename(media_path)}")
    #     print(f"  Duration: {duration_sec:.2f}s")
    #     print(f"  FPS: {fps}")
    #     print(f"  Total frames: {total_frames}")
        
    #     # Check if video is long enough
    #     if duration_sec < self.config.WINDOW_SEC:
    #         raise ValueError(f"Media ({duration_sec:.2f}s) shorter than window ({self.config.WINDOW_SEC}s)")
    #     audio_path = None
    #     file_ext = os.path.splitext(media_path)[1].lower()
    #     is_video = file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    #     # if is_video:
    #     #     audio_path = extract_audio_without_ffmpeg(media_path)
    #     # Pre-load audio for efficiency (handles both audio and video files)
    #     print("\nPre-loading audio track...")
    #     try:
    #         # total_duration = librosa.get_duration(path=media_path)
    #         full_audio, _ = librosa.core.load(media_path, sr=self.config.SR)
    #         print(f"  Audio loaded: {len(full_audio)/self.config.SR:.2f}s at {self.config.SR}Hz")
    #     except Exception as e:
    #         print(f"Error loading audio: {e}")
    #         print("Ensure ffmpeg is installed for video file audio extraction")
    #         raise
        
    #     # Calculate windows
    #     t_starts = np.arange(0.0, duration_sec - self.config.WINDOW_SEC + 0.001, step_sec)
        
    #     times = []
    #     probs_list = []
    #     dominant_emotions = []
        
    #     print(f"\nProcessing {len(t_starts)} windows with step={step_sec}s")
    #     for i in range(4):
    #         MODEL_PATH = os.listdir(self.config.MODEL_PATH)
    #         checkpoint = torch.load(MODEL_PATH+str(i+1), map_location=self.device, weights_only=False)
    #         if 'state_dict' in checkpoint:
    #             checkpoint = checkpoint['state_dict']
    #         # torch.save(self.model, "D:/Yeskendir_files/model.pt")
    #         self.model.load_state_dict(checkpoint)
    #         self.model.to(self.device)
    #         self.model.eval()
    #         for t_start in tqdm.tqdm(t_starts, desc="Processing windows"):
    #             # Extract audio segment
    #             start_sample = int(t_start * self.config.SR)
    #             end_sample = int((t_start + self.config.WINDOW_SEC) * self.config.SR)
    #             # print("Start:")
    #             # print(start_sample)
    #             # print("End:")
    #             # print(end_sample)
    #             audio_segment = full_audio[start_sample:end_sample]
    #             # print("Audio segment before:")
    #             # print(len(audio_segment))
    #             # Apply training preprocessing to audio segment
    #             target_length = int(self.config.SR * self.config.WINDOW_SEC)
    #             if len(audio_segment) < target_length:
    #                 audio_segment = np.array(list(audio_segment) + [0] * (target_length - len(audio_segment)))
    #             elif len(audio_segment) > target_length:
    #                 remain = len(audio_segment) - target_length
    #                 audio_segment = audio_segment[remain//2:-(remain - remain//2)]
    #             # print("Audio segment afbefore:")
    #             # print(len(audio_segment))
    #             # Extract MFCC
    #             mfcc = self.extract_mfcc_features(audio_segment)
                
    #             # Extract video frames
    #             video_frames = self.preprocess_video_like_training(media_path, t_start)
                
    #             # Prepare tensors
    #             audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
            
    #             # 2. Prepare Visual Tensor
    #             # Input 'video_frames' is numpy: (15, 224, 224, 3) -> [Time, Height, Width, Channel]
                
    #             # Normalize to 0-1 range (Standard PyTorch practice usually matches Training DataLoaders)
    #             # If your training data was 0-255, remove the '/ 255.0'
    #             video_tensor = torch.tensor(video_frames, dtype=torch.float32).to(self.device) / 255.0
                
    #             # Move Channel to dim 1: 
    #             # (15, 224, 224, 3) -> (15, 3, 224, 224) -> [Time, Channel, Height, Width]
    #             video_tensor = video_tensor.permute(0, 3, 1, 2)
                
    #             # 3. Flatten for Model Input
    #             # The model expects a flattened batch of images: (Batch * Time, Channel, Height, Width)
    #             # Since Batch=1, Batch*Time is just Time (15).
    #             # Current shape is already (15, 3, 224, 224). We just need to enforce it.
                
    #             # Note: We SKIP the .permute(0,2,1,3,4) seen in training because our 
    #             # starting tensor is already in the target order (Time, Channel, H, W).
    #             video_tensor_flat = video_tensor
                
    #             # Forward pass
    #             with torch.no_grad():
    #                 output = self.model(audio_tensor, video_tensor_flat)
    #                 # dot = make_dot(output, params = dict(self.model.named_parameters()))
    #                 # dot.render("graph", format="png", view=True)
    #                 prob = torch.softmax(output, dim=1).cpu().numpy()[0]
    #                 # merge neutral (0) + calm (1)
    #             merged_calm = prob[0] + prob[1]

            
    #             probabilities = np.array([
    #                 merged_calm,          # calm
    #                 prob[2],
    #                 prob[3],
    #                 prob[4],
    #                 prob[5],
    #                 prob[6],
    #                 prob[7]
    #             ], dtype=np.float32)

    #             probabilities = probabilities / probabilities.sum()
    #             probs_list.append(probabilities)
    #             times.append(t_start)
    #             dominant_emotions.append(self.config.EMOTIONS[np.argmax(probabilities)])
            
    #         probs_array = np.stack(probs_list)
            
    #         # Plot if requested
    #         if plot_results:
    #             self.plot_emotions(np.array(times), probs_array)
            
    #         return {
    #             'times': np.array(times),
    #             'probabilities': probs_array,
    #             'emotion_labels': self.config.EMOTIONS,
    #             'dominant_emotions': dominant_emotions,
    #             'window_size': self.config.WINDOW_SEC,
    #             'step_size': step_sec,
    #             'video_duration': duration_sec
    #         }
    def inference_sliding_window(
        self,
        media_path: str,
        step_sec: float = 1.0,
        plot_results: bool = False,
        model_name: str = "Model"
    ) -> Dict:

        # --- Validate media ---
        if not os.path.exists(media_path):
            raise FileNotFoundError(f"Media file not found: {media_path}")

        # Get video metadata
        cap = cv2.VideoCapture(media_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps
        cap.release()

        # Check minimum length
        if duration_sec < self.config.WINDOW_SEC:
            raise ValueError(f"Media ({duration_sec:.2f}s) shorter than window ({self.config.WINDOW_SEC}s)")

        # Load audio
        full_audio, _ = librosa.core.load(media_path, sr=self.config.SR)

        # Generate window start times
        t_starts = np.arange(0.0, duration_sec - self.config.WINDOW_SEC + 1e-3, step_sec)

        times = []
        probs_list = []

        # --- sliding window inference ---
        for t_start in tqdm.tqdm(t_starts, desc=f"[{model_name}] Windows"):

            start_sample = int(t_start * self.config.SR)
            end_sample = int((t_start + self.config.WINDOW_SEC) * self.config.SR)
            audio_segment = full_audio[start_sample:end_sample]

            # Pad or trim
            target_len = int(self.config.SR * self.config.WINDOW_SEC)
            if len(audio_segment) < target_len:
                audio_segment = np.pad(audio_segment, (0, target_len - len(audio_segment)))
            elif len(audio_segment) > target_len:
                extra = len(audio_segment) - target_len
                audio_segment = audio_segment[extra // 2: -(extra - extra // 2)]

            mfcc = self.extract_mfcc_features(audio_segment)

            # Video frames
            video_frames = self.preprocess_video_like_training(media_path, t_start)

            # Prepare tensors
            audio_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(self.device)
            video_tensor = torch.tensor(video_frames, dtype=torch.float32).to(self.device) / 255.0
            video_tensor = video_tensor.permute(0, 3, 1, 2)  # (15,3,224,224)

            # Forward pass
            with torch.no_grad():
                output = self.model(audio_tensor, video_tensor)
                prob = torch.softmax(output, dim=1).cpu().numpy()[0]

            # Merge neutral+calm
            merged_calm = prob[0] + prob[1]

            probabilities = np.array([
                merged_calm,
                prob[2], prob[3], prob[4],
                prob[5], prob[6], prob[7]
            ], dtype=np.float32)

            probabilities /= probabilities.sum()

            probs_list.append(probabilities)
            times.append(t_start)

        return {
            "model_name": model_name,
            "times": np.array(times),
            "probabilities": np.stack(probs_list),
            "emotion_labels": self.config.EMOTIONS,
            "duration": duration_sec,
        }
    def plot_unified_models_grid(self, results):
        """
        results: list of dicts returned by inference_sliding_window()
        """

        emotions = results[0]["emotion_labels"]
        num_emotions = len(emotions)

        plt.figure(figsize=(18, 3 * num_emotions))

        for e_idx in range(num_emotions):

            plt.subplot(num_emotions, 1, e_idx + 1)
            emotion_name = emotions[e_idx]

            for r in results:
                probs = r["probabilities"][:, e_idx]
                timeline = r["times"]

                # Smooth
                probs_smooth = gaussian_filter1d(probs, sigma=1.0)

                plt.plot(
                    timeline,
                    probs_smooth,
                    linewidth=2,
                    label=r["model_name"]
                )

            plt.title(f"{emotion_name}", fontsize=14)
            plt.ylabel("Probability")
            plt.grid(True, alpha=0.3)

            if e_idx == 0:
                plt.legend(fontsize=9)

        plt.xlabel("Time (s)")
        plt.tight_layout()
        plt.show()

    def plot_emotions(self, times: np.ndarray, probs: np.ndarray, save_path: str = None):
        # --- 1. Create evenly spaced timeline (0 → duration) ---
        video_duration = times[-1] + self.config.WINDOW_SEC
        timeline = np.linspace(0, video_duration, len(times))
        print(timeline)
        # --- 2. Smooth probabilities for each emotion ---
        smooth_probs = np.zeros_like(probs)
        for i in range(probs.shape[1]):
            smooth_probs[:, i] = gaussian_filter1d(probs[:, i], sigma=1.0)
        """Plot emotion probabilities over time"""
        plt.figure(figsize=(14, 8))
        EMOTION_COLORS = {
            "calm": "#808080",
            "happy": "#FFFF00",
            "sad": "#0000FF",
            "angry": "#FF0000",
            "fearful": "#BB00FF",
            "disgust": "#008000",
            "surprised": "#00FFFF"
        }


        plt.figure(figsize=(14, 8))

        for i, label in enumerate(self.config.EMOTIONS):
            color = EMOTION_COLORS.get(label.lower(), None)
            plt.plot(
                timeline,
                smooth_probs[:, i],
                label=label.capitalize(),
                linewidth=2,
                color=color
            )
        
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('Emotion Probability', fontsize=12)
        plt.title('Emotion Analysis Over Time', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])
        
        # Add minute markers
        max_time = times[-1]
        for minute in range(1, int(max_time / 60) + 1):
            plt.axvline(x=minute * 60, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()

#############################################
# VERIFICATION FUNCTIONS
#############################################

def verify_preprocessing_alignment():
    """
    Verify that preprocessing matches training expectations.
    This function helps debug preprocessing issues.
    """
    print("\n" + "="*60)
    print("PREPROCESSING ALIGNMENT VERIFICATION")
    print("="*60)
    
    config = TrainingConfig()
    
    print("\nExpected Training Configuration:")
    print(f"  Window Duration: {config.WINDOW_SEC}s")
    print(f"  Frame Count: {config.FRAME_COUNT}")
    print(f"  Input FPS: {config.INPUT_FPS}")
    print(f"  Audio SR: {config.SR} Hz")
    print(f"  MFCC Coefficients: {config.N_MFCC}")
    print(f"  MTCNN Size: {config.MTCNN_IMAGE_SIZE}")
    print(f"  Face Output Size: {config.FACE_OUTPUT_SIZE}")
    
    # Test frame selection
    print("\nFrame Selection Test (108 total frames @ 30fps):")
    frames = select_distributed(15, 108)
    print(f"  Selected frames: {frames}")
    print(f"  Frame spacing: ~{108/15:.1f} frames")
    
    print("\nCritical Notes:")
    print("  ✓ Audio must be padded with zeros if < 3.6s")
    print("  ✓ Audio must be cropped from both sides if > 3.6s")
    print("  ✓ Frames must use select_distributed() function")
    print("  ✓ MTCNN must use (720, 1280) image size")
    print("  ✓ Missing frames must be padded with black (zeros)")
    
    print("="*60)

#############################################
# MAIN USAGE
#############################################

if __name__ == "__main__":
    # First, verify preprocessing alignment
    verify_preprocessing_alignment()
    
    # Configuration
    config = TrainingConfig()
    
    # You can specify either a video file or an audio+video file
    # The script will automatically extract audio from video files
    
    # Initialize inference engine with training-aligned preprocessing
    print("\n" + "="*60)
    print("INITIALIZING TRAINING-ALIGNED INFERENCE")
    print("="*60)
    
    engine = TrainingAlignedInference(config)
    
    # Example 1: Single window inference
    print("\n" + "="*60)
    print("SINGLE WINDOW INFERENCE")
    print("="*60)
    
    result = engine.inference_single_window(config.MEDIA_PATH, start_sec=0.0)
    
    print(f"\nResults for: {result['media_file']}")
    print(f"  Window: {result['start_time']}s - {result['start_time'] + result['duration']}s")
    print(f"  Dominant Emotion: {result['dominant_emotion'].upper()}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"\nAll Emotions:")
    for emotion, prob in zip(result['emotion_labels'], result['probabilities']):
        bar = '█' * int(prob * 20)
        print(f"  {emotion.capitalize():10s}: {prob:6.2%} {bar}")
    
    # # Example 2: Sliding window inference
    # print("\n" + "="*60)
    # print("SLIDING WINDOW INFERENCE")
    # print("="*60)
    
    # try:
    #     sliding_results = engine.inference_sliding_window(
    #         config.MEDIA_PATH,
    #         step_sec=1.0,
    #         plot_results=True
    #     )
        
    #     # Summary statistics
    #     mean_probs = np.mean(sliding_results['probabilities'], axis=0)
    #     print(f"\nOverall Emotion Distribution:")
    #     sorted_indices = np.argsort(mean_probs)[::-1]
    #     for idx in sorted_indices:
    #         emotion = sliding_results['emotion_labels'][idx]
    #         prob = mean_probs[idx]
    #         bar = '█' * int(prob * 20)
    #         print(f"  {emotion.capitalize():10s}: {prob:6.2%} {bar}")
        
    #     # Find peak moments for each emotion
    #     print(f"\nPeak Moments for Each Emotion:")
    #     for i, emotion in enumerate(sliding_results['emotion_labels']):
    #         peak_idx = np.argmax(sliding_results['probabilities'][:, i])
    #         peak_time = sliding_results['times'][peak_idx]
    #         peak_prob = sliding_results['probabilities'][peak_idx, i]
    #         if peak_prob > 0.2:  # Only show significant peaks
    #             print(f"  {emotion.capitalize():10s}: {peak_time:6.1f}s (confidence: {peak_prob:.2%})")
                
    # except Exception as e:
    #     print(f"\n❌ Error during sliding window inference: {e}")
    #     if "ffmpeg" in str(e).lower():
    #         print("\nThis appears to be an audio extraction issue.")
    #         # install_ffmpeg_instructions()
    
    # print("\n" + "="*60)
    # print("INFERENCE COMPLETE")
    # print("="*60)
