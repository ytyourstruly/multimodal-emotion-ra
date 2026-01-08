# -*- coding: utf-8 -*-
"""
CREMA-D Video Preprocessing
- Extracts frames from FLV videos
- Detects and crops faces using MTCNN
- Saves cropped videos as .avi and .npy files
"""

import os
import numpy as np          
import cv2
from tqdm import tqdm
import torch
from facenet_pytorch import MTCNN


def preprocess_cremad_video(root_path, save_frames=15, input_fps=30, save_length=3.6, 
                            save_avi=False, output_size=(224, 224)):
    """
    Preprocess CREMA-D video files with face detection
    
    Args:
        root_path: path to CREMA-D root directory
        save_frames: number of frames to save (default: 15)
        input_fps: input video FPS (default: 30)
        save_length: target video length in seconds (default: 3.6)
        save_avi: whether to save .avi files (default: True)
        output_size: output frame size (default: (224, 224))
    """
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize MTCNN for face detection
    # Note: CREMA-D videos are typically smaller resolution than RAVDESS
    mtcnn = MTCNN(device=device)
    
    video_folder = os.path.join(root_path, 'VideoFlash')
    
    if not os.path.exists(video_folder):
        raise ValueError(f"VideoFlash folder not found at {video_folder}")
    
    # Get all FLV files
    video_files = [f for f in os.listdir(video_folder) if f.endswith('.flv')]
    
    # Filter out already processed files
    video_files = [f for f in video_files if 'facecroppad' not in f]
    
    print(f"\n{'='*60}")
    print(f"CREMA-D Video Preprocessing")
    print(f"{'='*60}")
    print(f"Found {len(video_files)} video files to preprocess")
    print(f"Output frames: {save_frames}")
    print(f"Video length: {save_length} seconds")
    print(f"Output size: {output_size}")
    print(f"Save AVI: {save_avi}")
    print(f"{'='*60}\n")
    
    failed_videos = []
    processed_count = 0
    skipped_count = 0
    
    # Helper function for frame selection
    select_distributed = lambda m, n: [i*n//m + n//(2*m) for i in range(m)]
    
    for filename in tqdm(video_files, desc="Processing videos"):
        output_base = os.path.join(video_folder, filename[:-4] + '_facecroppad')
        
        # Skip if already processed
        if os.path.exists(output_base + '.npy'):
            skipped_count += 1
            continue
        
        try:
            input_path = os.path.join(video_folder, filename)
            cap = cv2.VideoCapture(input_path)
            
            # Count total frames
            total_frames = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                total_frames += 1
            
            if total_frames == 0:
                print(f"\nWarning: No frames in {filename}")
                failed_videos.append((filename, "no frames"))
                continue
            
            # Reset video capture
            cap = cv2.VideoCapture(input_path)
            
            # Calculate skip at beginning if video is longer than save_length
            target_total_frames = int(save_length * input_fps)
            skip_begin = 0
            
            if target_total_frames < total_frames:
                skip_begin = int((total_frames - target_total_frames) // 2)
                for _ in range(skip_begin):
                    cap.read()
                total_frames = target_total_frames
            
            # Select frames to save (uniformly distributed)
            frames_to_select = select_distributed(save_frames, total_frames)
            save_fps = save_frames // (total_frames // input_fps) if total_frames >= input_fps else save_frames
            
            # Initialize video writer if needed
            if save_avi:
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                out = cv2.VideoWriter(
                    output_base + '.avi',
                    fourcc,
                    save_fps,
                    output_size
                )
            
            numpy_video = []
            frame_ctr = 0
            faces_detected = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if this frame should be saved
                if frame_ctr not in frames_to_select:
                    frame_ctr += 1
                    continue
                else:
                    frames_to_select.remove(frame_ctr)
                    frame_ctr += 1
                
                # Convert BGR to RGB for MTCNN
                try:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                except Exception as e:
                    print(f"\nError converting frame {frame_ctr} in {filename}: {e}")
                    failed_videos.append((filename, f"conversion error at frame {frame_ctr}"))
                    break
                
                # Detect face
                frame_tensor = torch.from_numpy(frame_rgb).to(device)
                boxes, _ = mtcnn.detect(frame_tensor)
                
                # Crop face if detected
                if boxes is not None and len(boxes) > 0:
                    # Use the first detected face
                    bbox = boxes[0]
                    x1, y1, x2, y2 = [int(coord) for coord in bbox]
                    
                    # Ensure coordinates are within frame bounds
                    h, w = frame.shape[:2]
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    
                    # Crop face
                    face = frame[y1:y2, x1:x2, :]
                    faces_detected += 1
                else:
                    # No face detected, use full frame
                    face = frame
                
                # Resize to output size
                face_resized = cv2.resize(face, output_size)
                
                # Save frame
                if save_avi:
                    out.write(face_resized)
                numpy_video.append(face_resized)
            
            cap.release()
            
            # Pad with black frames if necessary
            while len(numpy_video) < save_frames:
                black_frame = np.zeros((*output_size, 3), dtype=np.uint8)
                if save_avi:
                    out.write(black_frame)
                numpy_video.append(black_frame)
            
            if save_avi:
                out.release()
            
            # Save as numpy array
            np.save(output_base + '.npy', np.array(numpy_video))
            
            # Verify frame count
            if len(numpy_video) != save_frames:
                print(f"\nWarning: {filename} has {len(numpy_video)} frames instead of {save_frames}")
                failed_videos.append((filename, f"wrong frame count: {len(numpy_video)}"))
            
            # Warn if few faces detected
            if faces_detected < save_frames * 0.5:
                print(f"\nWarning: Only {faces_detected}/{save_frames} frames had detected faces in {filename}")
            
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            failed_videos.append((filename, str(e)))
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Video Preprocessing Complete!")
    print(f"{'='*60}")
    print(f"Processed: {processed_count} files")
    print(f"Skipped (already processed): {skipped_count} files")
    print(f"Failed: {len(failed_videos)} files")
    print(f"Total: {len(video_files)} files")
    
    if failed_videos:
        print(f"\nFailed videos:")
        for vid, reason in failed_videos:
            print(f"  - {vid}: {reason}")
    
    # Verify processed files
    processed_npy = [f for f in os.listdir(video_folder) if 'facecroppad.npy' in f]
    if save_avi:
        processed_avi = [f for f in os.listdir(video_folder) if 'facecroppad.avi' in f]
        print(f"\nVerification:")
        print(f"  - NPY files: {len(processed_npy)}")
        print(f"  - AVI files: {len(processed_avi)}")
    else:
        print(f"\nVerification: Found {len(processed_npy)} processed NPY files")
    
    print(f"{'='*60}\n")
    
    # Save failed videos list
    if failed_videos:
        failed_path = os.path.join(root_path, 'failed_videos.txt')
        with open(failed_path, 'w') as f:
            for vid, reason in failed_videos:
                f.write(f"{vid}: {reason}\n")
        print(f"Failed videos list saved to: {failed_path}")
    
    return processed_count, failed_videos


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CREMA-D video files')
    parser.add_argument(
        '--cremad_path',
        type=str,
        required=True,
        help='Path to CREMA-D root directory'
    )
    parser.add_argument(
        '--save_frames',
        type=int,
        default=15,
        help='Number of frames to extract (default: 15)'
    )
    parser.add_argument(
        '--input_fps',
        type=int,
        default=30,
        help='Input video FPS (default: 30)'
    )
    parser.add_argument(
        '--save_length',
        type=float,
        default=3.6,
        help='Target video length in seconds (default: 3.6)'
    )
    parser.add_argument(
        '--no_avi',
        action='store_true',
        help='Do not save AVI files (only NPY)'
    )
    parser.add_argument(
        '--output_size',
        type=int,
        nargs=2,
        default=[224, 224],
        help='Output frame size (default: 224 224)'
    )
    
    args = parser.parse_args()
    
    preprocess_cremad_video(
        args.cremad_path,
        save_frames=args.save_frames,
        input_fps=args.input_fps,
        save_length=args.save_length,
        save_avi=not args.no_avi,
        output_size=tuple(args.output_size)
    )