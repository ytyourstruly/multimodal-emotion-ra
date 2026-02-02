# -*- coding: utf-8 -*-
"""
CREMA-D Audio Preprocessing
Ensures all audio files are of the same length (3.6 seconds)
- If length < 3.6s: pad with zeros at the end
- If length > 3.6s: equally crop from both sides
"""

import librosa
import os
import soundfile as sf
import numpy as np
from tqdm import tqdm


def preprocess_cremad_audio(root_path, target_time=3.6, sr=22050):
    """
    Preprocess CREMA-D audio files
    
    Args:
        root_path: path to CREMA-D root directory
        target_time: target length in seconds (default: 3.6)
        sr: sample rate (default: 22050)
    """
    audio_folder = os.path.join(root_path, 'AudioWAV')
    
    if not os.path.exists(audio_folder):
        raise ValueError(f"AudioWAV folder not found at {audio_folder}")
    
    # Get all wav files
    audio_files = [f for f in os.listdir(audio_folder) if f.endswith('.wav')]
    
    # Filter out already processed files
    audio_files = [f for f in audio_files if 'croppad' not in f]
    
    print(f"Found {len(audio_files)} audio files to preprocess")
    print(f"Target length: {target_time} seconds")
    print(f"Sample rate: {sr} Hz")
    print(f"Processing...\n")
    
    target_length = int(sr * target_time)
    processed_count = 0
    skipped_count = 0
    
    for audiofile in tqdm(audio_files, desc="Processing audio"):
        input_path = os.path.join(audio_folder, audiofile)
        output_path = os.path.join(audio_folder, audiofile[:-4] + '_croppad.wav')
        
        # Skip if already processed
        if os.path.exists(output_path):
            skipped_count += 1
            continue
        
        try:
            # Load audio
            # y, file_sr = librosa.load(input_path, sr=sr)
            y, _ = librosa.load(input_path, sr=sr)

            # remove silence
            y, _ = librosa.effects.trim(y, top_db=20)

            # normalize amplitude
            y = librosa.util.normalize(y)

            # fix length
            y = librosa.util.fix_length(y, size=target_length)

            sf.write(output_path, y, sr)

            # Save processed audio
            sf.write(output_path, y, sr)
            processed_count += 1
            
        except Exception as e:
            print(f"\nError processing {audiofile}: {repr(e)}")
            break
    
    print(f"\n{'='*60}")
    print(f"Audio Preprocessing Complete!")
    print(f"Processed: {processed_count} files")
    print(f"Skipped (already processed): {skipped_count} files")
    print(f"Total: {len(audio_files)} files")
    print(f"{'='*60}\n")
    
    # Verify processed files
    processed_files = [f for f in os.listdir(audio_folder) if 'croppad.wav' in f]
    print(f"Verification: Found {len(processed_files)} processed audio files")
    
    return processed_count


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess CREMA-D audio files')
    parser.add_argument(
        '--cremad_path',
        type=str,
        default = '/home/yeskendir/Downloads/crema-d-mirror-main',
        help='Path to CREMA-D root directory'
    )
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
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"CREMA-D Audio Preprocessing")
    print(f"{'='*60}\n")
    print(f"Input directory: {args.cremad_path}")
    print(f"Target length: {args.target_time}s")
    print(f"Sample rate: {args.sample_rate} Hz\n")
    
    preprocess_cremad_audio(
        args.cremad_path,
        target_time=args.target_time,
        sr=args.sample_rate
    )
