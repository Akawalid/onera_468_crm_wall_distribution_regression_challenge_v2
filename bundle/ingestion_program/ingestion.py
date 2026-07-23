# ingestion.py
# 1. Imports participant's model
# 2. Trains it on training data
# 3. Saves predictions for the scoring program

import json
import os
import sys
import traceback
import numpy as np
from datetime import datetime as dt

# --- Paths ---
# Codabench invokes this program as:
#   ingestion.py $input $output $program $submission_program
# Fall back to local defaults when run without arguments.
if len(sys.argv) == 1:
    input_dir      = '/app/input_data/'
    output_dir     = '/app/output/'
    program_dir    = '/app/program'
    submission_dir = '/app/ingested_program'
else:
    input_dir      = os.path.abspath(sys.argv[1])
    output_dir     = os.path.abspath(sys.argv[2])
    program_dir    = os.path.abspath(sys.argv[3])
    submission_dir = os.path.abspath(sys.argv[4])

# Register paths once, at module level
for p in (output_dir, program_dir, submission_dir):
    if p not in sys.path:
        sys.path.append(p)


def get_data():
    """Load X_train, y_train, and X_test from disk."""
    paths = {
        'train':  os.path.join(input_dir, 'train_data.npy'),
        'labels': os.path.join(input_dir, 'train_labels.npy'),
        'test':   os.path.join(input_dir, 'test_data.npy'),
    }
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f'Missing {name} file: {path}')

    X_train = np.load(paths['train'])
    y_train = np.load(paths['labels'])[:, 0]
    X_test  = np.load(paths['test'])

    return X_train, y_train, X_test


def validate_predictions(y_pred, expected_len):
    """Raise if predictions are malformed, wrong length, or non-finite."""
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(f'predict() must return np.ndarray, got {type(y_pred)}')
    if y_pred.ndim == 0 or y_pred.shape[0] != expected_len:
        raise ValueError(
            f'predict() returned shape {y_pred.shape}, '
            f'expected first dim = {expected_len}'
        )
    if not np.all(np.isfinite(y_pred)):
        n_bad = np.sum(~np.isfinite(y_pred))
        raise ValueError(f'predict() returned {n_bad} NaN/Inf values.')


def print_bar():
    print('-' * 40)


def write_metadata(extra):
    """Best-effort metadata.json write; never raises."""
    try:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
            json.dump(extra, f, indent=2)
    except Exception as e:
        print(f'[ERR] Could not write metadata.json: {e}')


def run():
    """Run the full ingestion pipeline. Raises on any failure; the caller
    is responsible for catching, reporting, and writing failure metadata."""
    print_bar()
    print('Ingestion program. ONERA 468 CRM challenge rho.')

    # Make sure the output dir exists up front, so a failure right after
    # this point can still leave failure metadata behind for the scorer.
    os.makedirs(output_dir, exist_ok=True)

    # --- Load model ---
    print_bar()
    print('Importing participant model...')
    try:
        from model import Model
        model = Model()
        print('[OK] Model initialised.')
    except Exception as e:
        print(f'[ERR] Could not import/initialise Model: {e}')
        raise

    start_time = dt.now()

    # --- Load data ---
    print_bar()
    print('Loading data...')
    X_train, y_train, X_test = get_data()
    print(f'  X_train : {X_train.shape}  dtype={X_train.dtype}')
    print(f'  y_train : {y_train.shape}  dtype={y_train.dtype}')
    print(f'  X_test  : {X_test.shape}  dtype={X_test.dtype}')

    # --- Train ---
    print_bar()
    print(f'Training on {len(X_train)} samples...')
    train_start = dt.now()
    try:
        model.fit(X_train, y_train)
    except MemoryError:
        print('[ERR] Out of memory during fit().')
        raise
    except Exception as e:
        print(f'[ERR] fit() raised: {e}')
        raise

    elapsed = (dt.now() - train_start).total_seconds()
    print(f'[OK] Training done in {elapsed:.2f}s.')

    # --- Predict ---
    print_bar()
    print(f'Running inference on {len(X_test)} samples...')
    try:
        y_pred = model.predict(X_test)
    except MemoryError:
        print('[ERR] Out of memory during predict().')
        raise
    except Exception as e:
        print(f'[ERR] predict() raised: {e}')
        raise

    validate_predictions(y_pred, expected_len=len(X_test))
    print(f'[OK] Predictions shape: {y_pred.shape}')

    # --- Save outputs ---
    y_pred = y_pred.reshape(-1, 1).astype(np.float32)
    np.save(os.path.join(output_dir, 'Yhat.npy'), y_pred)
    print(f'[OK] Yhat.npy saved: {y_pred.shape}')

    # --- Save metadata ---
    duration = (dt.now() - start_time).total_seconds()
    write_metadata({
        'duration':      duration,
        'X_train_shape': list(X_train.shape),
        'X_test_shape':  list(X_test.shape),
        'y_pred_shape':  list(y_pred.shape),
        'success':       True,
    })
    print(f'[OK] Total duration: {duration:.2f}s')
    print('Ingestion complete. Handing off to scorer.')
    print_bar()


def main():
    try:
        run()
    except Exception as e:
        # Catch-all: never let a raw traceback be the only thing the user
        # sees. Log full details for organizers, leave clean failure
        # metadata for the scorer, and exit with a non-zero status.
        print_bar()
        print(f'[FATAL] Ingestion failed: {e}')
        traceback.print_exc()
        write_metadata({
            'duration': -1,
            'success':  False,
            'error':    str(e),
        })
        sys.exit(1)


if __name__ == '__main__':
    main()
