"""
Stage data/splitv2 (+ data/component_*) into the input_data/reference_data
layout expected by bundle/ingestion_program/ingestion.py and
bundle/scoring_program/scoring.py, ready to be zipped and uploaded to
Codabench interactively (one dataset per phase, per input/reference slot).

Source files under data/ are never modified -- everything here is either a
hardlink (for large arrays copied byte-for-byte) or a freshly written file
(for reshaped labels / derived test_conditions.npy). Safe to re-run: it
always removes and recreates the target files.

Output:
  bundle/feedback_phase/input_data/     (Development task, phase 1 test set)
  bundle/feedback_phase/reference_data/
  bundle/final_phase/input_data/        (Final task, phase 2 test set)
  bundle/final_phase/reference_data/

These paths match the commented-out input_data/reference_data entries
already in competition.yaml and the patterns already in .gitignore.
"""

import json
import os
import shutil

import numpy as np

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLIT_DIR  = os.path.join(ROOT, 'data', 'splitv2')
COMP_LABELS = os.path.join(ROOT, 'data', 'component_labels_unique.npy')
COMP_MAP    = os.path.join(ROOT, 'data', 'component_map.json')
BUNDLE_DIR  = os.path.join(ROOT, 'bundle')

NWALLP = 260774
COL_MINF, COL_AOA, COL_PI = 6, 7, 8  # -> test_conditions.npy column order

PHASES = {
    'feedback_phase': 'phase1',  # Development task
    'final_phase':     'phase2',  # Final task
}


def link_or_copy(src, dst):
    """ Hardlink src -> dst (same filesystem, zero extra disk, instant).
    Falls back to a real copy if hardlinking isn't possible. """
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def save_as_column(src_path, dst_path):
    """ ingestion.py / scoring.py index labels as arr[:, 0], but the raw
    label files are 1-D. Reshape to (N, 1) once into the staged copy. """
    arr = np.load(src_path)
    np.save(dst_path, arr.reshape(-1, 1))


def save_conditions(src_data_path, dst_path):
    """ Extract the tiny (n_cases, 3) [Minf, AoA, Pi] summary scoring.py
    needs, instead of duplicating the full multi-GB test_data.npy into
    reference_data. """
    X = np.load(src_data_path, mmap_mode='r')
    conditions = np.array(X[::NWALLP, [COL_MINF, COL_AOA, COL_PI]])
    np.save(dst_path, conditions)


def main():
    os.makedirs(BUNDLE_DIR, exist_ok=True)

    for bundle_phase, split_phase in PHASES.items():
        input_dir     = os.path.join(BUNDLE_DIR, bundle_phase, 'input_data')
        reference_dir = os.path.join(BUNDLE_DIR, bundle_phase, 'reference_data')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(reference_dir, exist_ok=True)

        print(f'--- {bundle_phase} ({split_phase}) ---')

        # --- input_data (participant-visible: train + test features) ---
        link_or_copy(
            os.path.join(SPLIT_DIR, 'train_data.npy'),
            os.path.join(input_dir, 'train_data.npy'))
        save_as_column(
            os.path.join(SPLIT_DIR, 'train_labels.npy'),
            os.path.join(input_dir, 'train_labels.npy'))
        link_or_copy(
            os.path.join(SPLIT_DIR, f'test_{split_phase}_data.npy'),
            os.path.join(input_dir, 'test_data.npy'))
        link_or_copy(COMP_LABELS, os.path.join(input_dir, 'component_labels_unique.npy'))
        link_or_copy(COMP_MAP, os.path.join(input_dir, 'component_map.json'))
        print(f'  [OK] input_data written to {input_dir}')

        # --- reference_data (scoring-only: labels, weights, conditions) ---
        save_as_column(
            os.path.join(SPLIT_DIR, f'test_{split_phase}_labels.npy'),
            os.path.join(reference_dir, 'test_labels.npy'))
        link_or_copy(
            os.path.join(SPLIT_DIR, f'test_{split_phase}_weights.npy'),
            os.path.join(reference_dir, 'test_weights.npy'))
        save_conditions(
            os.path.join(SPLIT_DIR, f'test_{split_phase}_data.npy'),
            os.path.join(reference_dir, 'test_conditions.npy'))
        link_or_copy(COMP_LABELS, os.path.join(reference_dir, 'component_labels_unique.npy'))
        link_or_copy(COMP_MAP, os.path.join(reference_dir, 'component_map.json'))
        print(f'  [OK] reference_data written to {reference_dir}')


if __name__ == '__main__':
    main()
