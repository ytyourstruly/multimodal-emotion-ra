"""
CREMA-D Dataset Preprocessor
Creates annotation files in the same format as RAVDESS for seamless integration
Can be called directly from main.py or run standalone
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split


def verify_cremad_files(root_path):
    """
    Verify CREMA-D files exist and are valid
    Returns list of valid filenames
    """
    # Files to exclude (as specified in original CremaDataset)
    problematic_files = [
        "1076_MTI_NEU_XX",
        "1076_MTI_SAD_XX",
        "1064_TIE_SAD_XX",
        "1064_IEO_DIS_MD",
        "1047_IEO_SAD_LO",
        "1047_IEO_FEA_LO"
    ]
    
    video_folder = os.path.join(root_path, "VideoFlash")
    audio_folder = os.path.join(root_path, "AudioWAV")
    
    if not os.path.exists(video_folder) or not os.path.exists(audio_folder):
        raise ValueError(f"CREMA-D folders not found in {root_path}")
    
    # Get all video files
    video_files = [f.replace('.flv', '') for f in os.listdir(video_folder) if f.endswith('.flv')]
    
    valid_files = []
    for filename in video_files:
        if filename in problematic_files:
            continue
            
        video_path = os.path.join(video_folder, filename + '.flv')
        audio_path = os.path.join(audio_folder, filename + '.wav')
        
        if os.path.exists(video_path) and os.path.exists(audio_path):
            valid_files.append(filename)
        else:
            print(f"Warning: Missing files for {filename}")
    
    print(f"Found {len(valid_files)} valid CREMA-D samples")
    return valid_files


def prepare_cremad_kfold_annotations(root_path, output_dir, n_folds=5, random_state=42):
    """
    Prepare CREMA-D annotation files with k-fold cross-validation
    Creates annotation files compatible with RAVDESS format
    
    Args:
        root_path: path to CREMA-D root directory
        output_dir: directory to save fold annotation files
        n_folds: number of folds (default: 5)
        random_state: random seed for reproducibility
        
    Returns:
        List of annotation file paths
    """
    print(f"\n{'='*60}")
    print(f"CREMA-D Preprocessing - Creating {n_folds}-Fold Annotations")
    print(f"{'='*60}\n")
    
    # Verify files
    valid_files = verify_cremad_files(root_path)
    
    if len(valid_files) == 0:
        raise ValueError("No valid CREMA-D files found!")
    
    # Create dataframe with file information
    data = []
    for filename in valid_files:
        parts = filename.split('_')
        actor_id = int(parts[0])
        emotion = parts[2]  # ANG, DIS, FEA, HAP, NEU, SAD
        
        data.append({
            'filename': filename,
            'actor_id': actor_id,
            'emotion': emotion
        })
    
    df = pd.DataFrame(data)
    
    # Emotion mapping (CREMA-D has 6 emotions, RAVDESS has 8)
    # We'll map CREMA-D emotions to RAVDESS label indices for consistency
    emotion_to_label = {
        "NEU": 0,  # neutral
        "HAP": 2,  # happy
        "SAD": 3,  # sad
        "ANG": 4,  # angry
        "FEA": 5,  # fearful
        "DIS": 6,  # disgust
    }
    
    df['label'] = df['emotion'].map(emotion_to_label)
    
    # Print emotion distribution
    print("Emotion Distribution:")
    print(df['emotion'].value_counts().sort_index())
    print(f"\nTotal samples: {len(df)}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create k-fold splits using StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    annotation_paths = []
    
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(df, df['emotion'])):
        fold_num = fold_idx + 1
        print(f"\nProcessing Fold {fold_num}...")
        
        # Get train+val and test sets
        train_val_df = df.iloc[train_val_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        # Further split train_val into train and validation (85-15 split)
        train_idx, val_idx = train_test_split(
            range(len(train_val_df)),
            test_size=0.15,
            stratify=train_val_df['emotion'],
            random_state=random_state
        )
        
        train_df = train_val_df.iloc[train_idx].copy()
        val_df = train_val_df.iloc[val_idx].copy()
        
        # Print split statistics
        print(f"  Training:   {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Testing:    {len(test_df)} samples")
        
        # Create annotation file in RAVDESS format
        # Format: video_path;audio_path;label;split
        annotation_file = os.path.join(output_dir, f'annotations_fold{fold_num}.txt')
        
        with open(annotation_file, 'w') as f:
            # Write training samples
            for _, row in train_df.iterrows():
                video_path = os.path.join(root_path, 'VideoFlash', row['filename'] + '.flv')
                audio_path = os.path.join(root_path, 'AudioWAV', row['filename'] + '.wav')
                f.write(f"{video_path};{audio_path};{row['label'] + 1};training\n")
            
            # Write validation samples
            for _, row in val_df.iterrows():
                video_path = os.path.join(root_path, 'VideoFlash', row['filename'] + '.flv')
                audio_path = os.path.join(root_path, 'AudioWAV', row['filename'] + '.wav')
                f.write(f"{video_path};{audio_path};{row['label'] + 1};validation\n")
            
            # Write test samples
            for _, row in test_df.iterrows():
                video_path = os.path.join(root_path, 'VideoFlash', row['filename'] + '.flv')
                audio_path = os.path.join(root_path, 'AudioWAV', row['filename'] + '.wav')
                f.write(f"{video_path};{audio_path};{row['label'] + 1};testing\n")
        
        annotation_paths.append(annotation_file)
        print(f"  Saved: {annotation_file}")
    
    # Save emotion mapping info
    mapping_file = os.path.join(output_dir, 'emotion_mapping.txt')
    with open(mapping_file, 'w') as f:
        f.write("CREMA-D Emotion Mapping to RAVDESS Label Indices:\n")
        f.write("=" * 50 + "\n")
        for emotion, label in sorted(emotion_to_label.items(), key=lambda x: x[1]):
            f.write(f"{emotion} -> Label {label + 1}\n")
        f.write("\nNote: CREMA-D has 6 emotions, RAVDESS has 8 emotions\n")
        f.write("Labels 1 (calm) and 7 (surprise) are not present in CREMA-D\n")
    
    print(f"\n{'='*60}")
    print(f"Preprocessing Complete!")
    print(f"Created {n_folds} annotation files in: {output_dir}")
    print(f"Emotion mapping saved to: {mapping_file}")
    print(f"{'='*60}\n")
    
    return annotation_paths


def prepare_cremad_single_split(root_path, output_path, test_size=0.15, val_size=0.15, random_state=42):
    """
    Prepare single train/val/test split for CREMA-D
    Alternative to k-fold for faster experimentation
    """
    print(f"\n{'='*60}")
    print(f"CREMA-D Preprocessing - Creating Single Split")
    print(f"{'='*60}\n")
    
    # Verify files
    valid_files = verify_cremad_files(root_path)
    
    # Create dataframe
    data = []
    for filename in valid_files:
        parts = filename.split('_')
        actor_id = int(parts[0])
        emotion = parts[2]
        
        data.append({
            'filename': filename,
            'actor_id': actor_id,
            'emotion': emotion
        })
    
    df = pd.DataFrame(data)
    
    # Emotion mapping
    emotion_to_label = {
        "NEU": 0, "HAP": 2, "SAD": 3,
        "ANG": 4, "FEA": 5, "DIS": 6,
    }
    
    df['label'] = df['emotion'].map(emotion_to_label)
    
    # Create splits
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['emotion'], random_state=random_state
    )
    
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_size/(1-test_size), 
        stratify=train_val_df['emotion'], random_state=random_state
    )
    
    print(f"Training:   {len(train_df)} samples")
    print(f"Validation: {len(val_df)} samples")
    print(f"Testing:    {len(test_df)} samples")
    
    # Save annotation file
    with open(output_path, 'w') as f:
        for _, row in train_df.iterrows():
            video_path = os.path.join(root_path, 'VideoFlash', row['filename'] + '.flv')
            audio_path = os.path.join(root_path, 'AudioWAV', row['filename'] + '.wav')
            f.write(f"{video_path};{audio_path};{row['label'] + 1};training\n")
        
        for _, row in val_df.iterrows():
            video_path = os.path.join(root_path, 'VideoFlash', row['filename'] + '.flv')
            audio_path = os.path.join(root_path, 'AudioWAV', row['filename'] + '.wav')
            f.write(f"{video_path};{audio_path};{row['label'] + 1};validation\n")
        
        for _, row in test_df.iterrows():
            video_path = os.path.join(root_path, 'VideoFlash', row['filename'] + '.flv')
            audio_path = os.path.join(root_path, 'AudioWAV', row['filename'] + '.wav')
            f.write(f"{video_path};{audio_path};{row['label'] + 1};testing\n")
    
    print(f"\nAnnotation file saved to: {output_path}")
    print(f"{'='*60}\n")
    
    return output_path


if __name__ == '__main__':
    """
    Standalone execution
    Run this script directly to preprocess CREMA-D dataset
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CREMA-D Dataset')
    parser.add_argument('--cremad_path', type=str, required=True,
                        help='Path to CREMA-D root directory')
    parser.add_argument('--output_dir', type=str, default='cremad_annotations',
                        help='Output directory for annotation files')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--single_split', action='store_true',
                        help='Create single split instead of k-fold')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    if args.single_split:
        output_file = os.path.join(args.output_dir, 'annotations.txt')
        os.makedirs(args.output_dir, exist_ok=True)
        prepare_cremad_single_split(
            args.cremad_path,
            output_file,
            random_state=args.random_seed
        )
    else:
        prepare_cremad_kfold_annotations(
            args.cremad_path,
            args.output_dir,
            n_folds=args.n_folds,
            random_state=args.random_seed
        )