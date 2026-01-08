#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CREMA-D Complete Preprocessing Pipeline
Runs all preprocessing steps in correct order:
1. Audio preprocessing (crop/pad to 3.6s)
2. Video preprocessing (face detection and cropping)
3. Create annotation files for k-fold cross-validation

This should be run ONCE before training, just like RAVDESS preprocessing
"""

import os
import sys
import argparse
from cremad_audio_preprocess import preprocess_cremad_audio
from cremad_video_preprocess import preprocess_cremad_video
from preprocess_cremad import prepare_cremad_kfold_annotations


def verify_cremad_structure(cremad_path):
    """Verify CREMA-D dataset has correct structure"""
    required_folders = ['VideoFlash', 'AudioWAV']
    required_files = ['VideoDemographics.csv', 'summaryTable.csv']
    
    print(f"Verifying CREMA-D structure at: {cremad_path}")
    
    # Check folders
    for folder in required_folders:
        folder_path = os.path.join(cremad_path, folder)
        if not os.path.exists(folder_path):
            raise ValueError(f"Missing required folder: {folder}")
        print(f"  ✓ Found {folder}/")
    
    # Check files
    for file in required_files:
        file_path = os.path.join(cremad_path, file)
        if not os.path.exists(file_path):
            raise ValueError(f"Missing required file: {file}")
        print(f"  ✓ Found {file}")
    
    print("Dataset structure verified!\n")


def count_files(cremad_path, pattern):
    """Count files matching pattern"""
    video_folder = os.path.join(cremad_path, 'VideoFlash')
    audio_folder = os.path.join(cremad_path, 'AudioWAV')
    
    if 'flv' in pattern:
        files = [f for f in os.listdir(video_folder) if pattern in f]
    else:
        files = [f for f in os.listdir(audio_folder) if pattern in f]
    
    return len(files)


def full_preprocess_pipeline(cremad_path, n_folds=5, skip_audio=False, 
                            skip_video=False, skip_annotations=False,
                            audio_params=None, video_params=None):
    """
    Run complete CREMA-D preprocessing pipeline
    
    Args:
        cremad_path: path to CREMA-D root directory
        n_folds: number of folds for cross-validation
        skip_audio: skip audio preprocessing
        skip_video: skip video preprocessing
        skip_annotations: skip annotation creation
        audio_params: dict of audio preprocessing parameters
        video_params: dict of video preprocessing parameters
    """
    print("\n" + "="*70)
    print("CREMA-D COMPLETE PREPROCESSING PIPELINE")
    print("="*70 + "\n")
    
    # Verify structure
    try:
        verify_cremad_structure(cremad_path)
    except ValueError as e:
        print(f"Error: {e}")
        print("\nPlease ensure your CREMA-D dataset has the correct structure:")
        print("  CREMA-D/")
        print("    ├── VideoFlash/")
        print("    ├── AudioWAV/")
        print("    ├── VideoDemographics.csv")
        print("    └── summaryTable.csv")
        return False
    
    # Default parameters
    if audio_params is None:
        audio_params = {'target_time': 3.6, 'sr': 22050}
    if video_params is None:
        video_params = {
            'save_frames': 15,
            'input_fps': 30,
            'save_length': 3.6,
            'save_avi': False,  # Changed to False - AVI not needed for training
            'output_size': (224, 224)
        }
    
    # Count original files
    original_videos = count_files(cremad_path, '.flv')
    original_audios = count_files(cremad_path, '.wav')
    print(f"Original files found:")
    print(f"  - Videos (.flv): {original_videos}")
    print(f"  - Audios (.wav): {original_audios}\n")
    
    success = True
    
    # Step 1: Audio Preprocessing
    if not skip_audio:
        print("="*70)
        print("STEP 1/3: Audio Preprocessing")
        print("="*70 + "\n")
        
        try:
            processed = preprocess_cremad_audio(cremad_path, **audio_params)
            print(f"\n✓ Audio preprocessing completed: {processed} files processed\n")
        except Exception as e:
            print(f"\n✗ Audio preprocessing failed: {e}\n")
            success = False
            return False
    else:
        print("Skipping audio preprocessing (--skip_audio flag)\n")
    
    # Step 2: Video Preprocessing
    if not skip_video:
        print("="*70)
        print("STEP 2/3: Video Preprocessing (This may take a while...)")
        print("="*70 + "\n")
        
        try:
            processed, failed = preprocess_cremad_video(cremad_path, **video_params)
            print(f"\n✓ Video preprocessing completed: {processed} files processed")
            if failed:
                print(f"  Warning: {len(failed)} files failed (see failed_videos.txt)")
            print()
        except Exception as e:
            print(f"\n✗ Video preprocessing failed: {e}\n")
            success = False
            return False
    else:
        print("Skipping video preprocessing (--skip_video flag)\n")
    
    # Step 3: Create Annotations
    if not skip_annotations:
        print("="*70)
        print("STEP 3/3: Creating Annotation Files")
        print("="*70 + "\n")
        
        annotation_dir = os.path.join(cremad_path, 'cremad_annotations')
        
        try:
            annotation_paths = prepare_cremad_kfold_annotations(
                cremad_path,
                annotation_dir,
                n_folds=n_folds
            )
            print(f"\n✓ Annotation files created: {len(annotation_paths)} folds\n")
        except Exception as e:
            print(f"\n✗ Annotation creation failed: {e}\n")
            success = False
            return False
    else:
        print("Skipping annotation creation (--skip_annotations flag)\n")
    
    # Final Summary
    print("="*70)
    print("PREPROCESSING COMPLETE!")
    print("="*70)
    
    if not skip_audio:
        processed_audios = count_files(cremad_path, 'croppad.wav')
        print(f"\nProcessed Audio Files: {processed_audios}")
    
    if not skip_video:
        processed_videos = count_files(cremad_path, 'facecroppad.npy')
        print(f"Processed Video Files: {processed_videos}")
    
    if not skip_annotations:
        annotation_dir = os.path.join(cremad_path, 'cremad_annotations')
        if os.path.exists(annotation_dir):
            annotations = [f for f in os.listdir(annotation_dir) if f.endswith('.txt')]
            print(f"Annotation Files: {len(annotations)}")
            print(f"Location: {annotation_dir}")
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Update your main training script to use CREMA-D:")
    print("   python main.py --dataset CREMAD --cremad_path <path_to_cremad>")
    print("\n2. The preprocessed files are now ready for training!")
    print("\n3. Annotation files are in: cremad_annotations/")
    print("   - annotations_fold1.txt")
    print("   - annotations_fold2.txt")
    print("   - ...")
    print("\n" + "="*70 + "\n")
    
    return success


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Complete CREMA-D Preprocessing Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full preprocessing pipeline
  python preprocess_cremad_full.py --cremad_path D:/Datasets/CREMA-D

  # Run only audio preprocessing
  python preprocess_cremad_full.py --cremad_path D:/Datasets/CREMA-D --skip_video --skip_annotations

  # Run only video preprocessing
  python preprocess_cremad_full.py --cremad_path D:/Datasets/CREMA-D --skip_audio --skip_annotations

  # Run only annotation creation (after audio/video preprocessing)
  python preprocess_cremad_full.py --cremad_path D:/Datasets/CREMA-D --skip_audio --skip_video

  # Custom parameters
  python preprocess_cremad_full.py --cremad_path D:/Datasets/CREMA-D --n_folds 10 --save_frames 20
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--cremad_path',
        type=str,
        required=True,
        help='Path to CREMA-D root directory'
    )
    
    # Pipeline control
    parser.add_argument(
        '--n_folds',
        type=int,
        default=5,
        help='Number of folds for cross-validation (default: 5)'
    )
    
    parser.add_argument(
        '--skip_audio',
        action='store_true',
        help='Skip audio preprocessing step'
    )
    
    parser.add_argument(
        '--skip_video',
        action='store_true',
        help='Skip video preprocessing step'
    )
    
    parser.add_argument(
        '--skip_annotations',
        action='store_true',
        help='Skip annotation file creation'
    )
    
    # Audio parameters
    parser.add_argument(
        '--target_time',
        type=float,
        default=3.6,
        help='Target audio length in seconds (default: 3.6)'
    )
    
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=22050,
        help='Audio sample rate (default: 22050)'
    )
    
    # Video parameters
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
    
    # Prepare parameters
    audio_params = {
        'target_time': args.target_time,
        'sr': args.sample_rate
    }
    
    video_params = {
        'save_frames': args.save_frames,
        'input_fps': args.input_fps,
        'save_length': args.save_length,
        'save_avi': not args.no_avi,
        'output_size': tuple(args.output_size)
    }
    
    # Run pipeline
    success = full_preprocess_pipeline(
        args.cremad_path,
        n_folds=args.n_folds,
        skip_audio=args.skip_audio,
        skip_video=args.skip_video,
        skip_annotations=args.skip_annotations,
        audio_params=audio_params,
        video_params=video_params
    )
    
    sys.exit(0 if success else 1)