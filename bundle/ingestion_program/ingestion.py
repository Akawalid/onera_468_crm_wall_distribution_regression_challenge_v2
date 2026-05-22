# The ingestion program is the program that:
# 1. Take participant's code submission
# 2. Train the given model on the training data
# 3. Make predictions on the test data, and save them to forward them to the scoring program

# Imports
import json
import os
import sys
import subprocess
import importlib
import resource
import threading
import numpy as np
from datetime import datetime as dt

# Paths
input_dir      = '/app/input_data/'
output_dir     = '/app/output/'
program_dir    = '/app/program'
submission_dir = '/app/ingested_program'


sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)


def check_and_install_dependencies(submission_dir):
    """
    Installs missing dependencies from requirements.txt only if not already present.
    """
    req_path = os.path.join(submission_dir, 'requirements.txt')

    if not os.path.exists(req_path):
        print('[*] No requirements.txt found. Using default environment.')
        return

    print('[*] Checking requirements.txt for missing libraries...')
    with open(req_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    #====================== time can be factorized here, but checking the libraries in the environement
    # instead of doing it with exceptions =============================================================
    to_install = []
    for req in requirements:
        package_name = req.split('==')[0].split('>=')[0].split('>')[0].strip().replace('-', '_')
        try:
            importlib.import_module(package_name)
        except ImportError:
            to_install.append(req)

    if to_install:
        print(f'[*] Installing missing dependencies: {", ".join(to_install)}...')
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-cache-dir', *to_install])
            print('[OK] Installation complete.')
        except Exception as e:
            print(f'[!] Pip installation failed: {e}')
    else:
        print('[OK] All requirements are already met. Skipping installation.')


def get_data():
    """ Get X_train, y_train and X_test. """
    train_path = os.path.join(input_dir, 'X9_train_fl32.npy')
    label_path = os.path.join(input_dir, 'rho_fl32.npy')
    test_path  = os.path.join(input_dir, 'X9_test_fl32.npy')

    if not os.path.exists(train_path):
        raise FileNotFoundError(f'Training data not found: {train_path}')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f'Training labels not found: {label_path}')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f'Test data not found: {test_path}')

    X_train = np.load(train_path)
    y_train = np.load(label_path)[:, 0]
    X_test  = np.load(test_path)
    return X_train, y_train, X_test


def print_bar():
    """ Display a bar ('----------') """
    print('-' * 10)


#To be reviewed, we can maybe use OOP conepts to accelerate the process
def is_oom(e):
    """ Check if an exception looks like an out-of-memory error. """
    msg = str(e).lower()
    return any(k in msg for k in ('memory', 'alloc', 'oom', 'out of mem', 'cannot allocate'))


def check_memory_usage():
    """ Return current process RAM usage in GB. """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def get_total_memory_gb():
    """ Get total system RAM in GB from /proc/meminfo. """
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    return int(line.split()[1]) / 1e6
    except:
        pass
    return float('nan')


def memory_monitor(interval=30):
    """ Background thread that logs memory usage every 30s. """
    total_gb = get_total_memory_gb()
    while True:
        used = check_memory_usage()
        print(f'[MEM] RAM usage: {used:.2f} GB / {total_gb:.2f} GB total', flush=True)
        threading.Event().wait(interval)


def main():
    """ The ingestion program. """
    print_bar()
    print('Ingestion program - ONERA 468 CRM challenge rho.')

    monitor = threading.Thread(target=memory_monitor, daemon=True)
    monitor.start()

    check_and_install_dependencies(submission_dir)

    print_bar()
    print('Importing model.')
    sys.path.append(submission_dir)
    try:
        from model import Model
        model = Model()
        print('[OK] Model initialized successfully.')
    except ImportError as e:
        print(f'[-] Could not import Model from submission: {e}')
        raise

    start_time = dt.now()

    print_bar()
    print('Reading data.')
    X_train, y_train, X_test = get_data()
    print(f'X_train : {X_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'X_test  : {X_test.shape}')

    print_bar()
    if X_train is not None and y_train is not None:
        print(f'Training the model on {len(X_train)} samples.')
        try:
            model.fit(X_train, y_train)
        except MemoryError:
            print('[ERR] OOM during fit: not enough RAM to train on this data.')
            raise
        except Exception as e:
            if is_oom(e):
                print(f'[ERR] OOM during fit (caught as generic exception): {e}')
            else:
                print(f'[ERR] fit() failed: {e}')
            raise
    else:
        print('[!] Skipping fit: training data missing.')

    print_bar()
    print(f'Making predictions on {len(X_test)} test samples.')
    try:
        y_pred = model.predict(X_test)
    except MemoryError:
        print('[ERR] OOM during predict: not enough RAM to run inference.')
        raise
    except Exception as e:
        if is_oom(e):
            print(f'[ERR] OOM during predict (caught as generic exception): {e}')
        else:
            print(f'[ERR] predict() failed: {e}')
        raise
    print(f'y_pred shape : {y_pred.shape}')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    y_pred = y_pred.reshape(-1, 1).astype(np.float32)
    np.save(os.path.join(output_dir, 'Yhat.npy'), y_pred)
    print(f'[OK] Yhat.npy saved : {y_pred.shape}')

    end_time = dt.now()
    duration = end_time - start_time
    print(f'[OK] Total duration: {duration}')

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({'duration': duration.total_seconds()}, f)

    print('Ingestion program finished. Moving on to scoring.')
    print_bar()


if __name__ == '__main__':
    main()