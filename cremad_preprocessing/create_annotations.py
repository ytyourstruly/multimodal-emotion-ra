# -*- coding: utf-8 -*-
"""
create_annotations_cremad.py
-----------------------------
Generates per-fold annotation files for the CREMA-D dataset.

Expected directory layout:
    ROOT/
        AudioWAV/       ← flat folder of .wav files
            1001_DFA_ANG_XX.wav
            1001_DFA_ANG_XX_croppad.wav   (if you have croppad versions)
            ...
        VideoFlash/     ← flat folder of .npy files (extracted frames)
            1001_DFA_ANG_XX_croppad.npy
            ...

Pairing logic:
    The .npy file is the anchor.  Its corresponding .wav is found by
    stripping '_croppad' (if present) from the base name and switching
    the extension to .wav.  Adjust WAV_CROPPAD below if your .wav files
    also carry the _croppad suffix.

Key guarantees:
    - Actor-exclusive splits: no actor appears in more than one of
      {train, validation, test} within any fold.
    - Two modes via SUBSET_MODE:
        False → full 5-fold CV over all 91 actors.
        True  → small subset (20 actors, 4 folds) for quick feature testing.
"""

import random
import numpy as np
import os
import glob

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

ROOT = '/home/yeskendir/Downloads/crema-d-mirror-main'          # <-- change to your root

# Set to True if your .wav files also have _croppad in the name.
# False  → 1001_DFA_ANG_XX_croppad.npy  pairs with  1001_DFA_ANG_XX.wav
# True   → 1001_DFA_ANG_XX_croppad.npy  pairs with  1001_DFA_ANG_XX_croppad.wav
WAV_CROPPAD = True

# Flip to True for a small actor subset (fast iteration while testing features)
SUBSET_MODE = True

RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

EMOTION_TO_LABEL = {
    'NEU': 0,   # Neutral
    'HAP': 1,   # Happy
    'SAD': 2,   # Sad
    'FEA': 3,   # Fear
    'DIS': 4,   # Disgust
    'ANG': 5,   # Anger
}

ALL_ACTOR_IDS = list(range(1001, 1092))       # 91 actors: 1001-1091

# --- subset config (only used when SUBSET_MODE == True) ---
SUBSET_ACTOR_IDS = list(range(1001, 1021))    # 20 actors
SUBSET_N_FOLDS   = 4                          # 4 folds → test≈5, val≈5, train≈10


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_npy_files(root):
    """Return sorted list of all .npy paths inside root/VideoFlash/."""
    folder = os.path.join(root, 'VideoFlash')
    return sorted(glob.glob(os.path.join(folder, '*.npy')))


def npy_to_wav(npy_path, root, wav_croppad=False):
    """
    Given a .npy path, return the matching .wav path inside root/AudioWAV/.

    Example (wav_croppad=False):
        VideoFlash/1001_DFA_ANG_XX_croppad.npy  →  AudioWAV/1001_DFA_ANG_XX.wav
    Example (wav_croppad=True):
        VideoFlash/1001_DFA_ANG_XX_croppad.npy  →  AudioWAV/1001_DFA_ANG_XX_croppad.wav
    """
    basename = os.path.basename(npy_path)          # 1001_DFA_ANG_XX_croppad.npy

    # strip .npy
    name = basename.replace('.npy', '')            # 1001_DFA_ANG_XX_croppad

        # remove _croppad suffix if present
    name = name.replace('_facecroppad', '_croppad')        # 1001_DFA_ANG_XX

    wav_name = name + '.wav'
    return os.path.join(root, 'AudioWAV', wav_name)


def actor_id(filepath):
    """Extract actor ID (int) from filename: first token before '_'."""
    return int(os.path.basename(filepath).split('_')[0])


def emotion_label(filepath):
    """Extract emotion label (int) from filename: third token."""
    parts = os.path.basename(filepath).split('_')
    return EMOTION_TO_LABEL[parts[2]]


def create_folds(actor_ids, n_folds, seed):
    """
    Shuffle actors, split into n_folds chunks, then build cyclic
    test / val / train assignment for each fold.

    Returns list of n_folds entries, each = [test_ids, val_ids, train_ids].
    """
    ids = list(actor_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)

    chunks = [list(c) for c in np.array_split(ids, n_folds)]

    folds = []
    for i in range(n_folds):
        test  = chunks[i]
        val   = chunks[(i + 1) % n_folds]
        train = []
        for j in range(n_folds):
            if j != i and j != (i + 1) % n_folds:
                train.extend(chunks[j])
        folds.append([test, val, train])
    return folds


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    # --- choose mode ---
    if SUBSET_MODE:
        actor_ids = SUBSET_ACTOR_IDS
        n_folds   = SUBSET_N_FOLDS
        print(f"[SUBSET MODE] {len(actor_ids)} actors, {n_folds} folds")
    else:
        actor_ids = ALL_ACTOR_IDS
        n_folds   = 5
        print(f"[FULL MODE]   {len(actor_ids)} actors, {n_folds} folds")

    actor_set = set(actor_ids)

    # --- discover .npy files ---
    npy_files = get_npy_files(ROOT)
    print(f"Found {len(npy_files)} .npy files in VideoFlash/")
    if not npy_files:
        print("ERROR: no .npy files found. Check ROOT path.")
        return

    # --- filter to actors we care about ---
    npy_files = [f for f in npy_files if actor_id(f) in actor_set]
    print(f"After actor filter: {len(npy_files)} .npy files")

    # --- sanity check: verify every .npy has a matching .wav ---
    missing_wavs = []
    for npy in npy_files:
        wav = npy_to_wav(npy, ROOT, WAV_CROPPAD)
        if not os.path.isfile(wav):
            missing_wavs.append((npy, wav))
    if missing_wavs:
        print(f"\nWARNING: {len(missing_wavs)} .npy files have no matching .wav:")
        for npy, wav in missing_wavs[:10]:   # show first 10
            print(f"  {os.path.basename(npy)}  →  expected: {wav}")
        if len(missing_wavs) > 10:
            print(f"  ... and {len(missing_wavs)-10} more.")
        print("These samples will still be written; fix WAV_CROPPAD if paths look wrong.\n")

    # --- build folds ---
    folds = create_folds(actor_ids, n_folds, RANDOM_SEED)

    print("\n--- Fold Actor Assignments ---")
    for i, (test, val, train) in enumerate(folds):
        print(f"  Fold {i+1}:  test({len(test)})={sorted(test)}  "
              f"val({len(val)})={sorted(val)}  "
              f"train({len(train)})={sorted(train)}")
    print()

    # --- write annotation files ---
    for fold_idx, (test_ids, val_ids, train_ids) in enumerate(folds):
        test_set  = set(test_ids)
        val_set   = set(val_ids)
        train_set = set(train_ids)

        annotation_file = f'cremad_preprocessing/annotations_croppad_fold{fold_idx + 1}.txt'

        # fresh file every run
        if os.path.exists(annotation_file):
            os.remove(annotation_file)

        counts = {'training': 0, 'validation': 0, 'testing': 0}

        with open(annotation_file, 'w') as f:
            for npy in npy_files:                       # already sorted
                aid   = actor_id(npy)
                wav   = npy_to_wav(npy, ROOT, WAV_CROPPAD)
                label = emotion_label(npy)

                if   aid in train_set:  split = 'training'
                elif aid in val_set:    split = 'validation'
                elif aid in test_set:   split = 'testing'
                else:                   continue

                f.write(f"{npy};{wav};{label};{split}\n")
                counts[split] += 1

        total = sum(counts.values())
        print(f"Fold {fold_idx+1} → {annotation_file}  |  "
              f"train={counts['training']}  val={counts['validation']}  "
              f"test={counts['testing']}  total={total}")


if __name__ == '__main__':
    main()