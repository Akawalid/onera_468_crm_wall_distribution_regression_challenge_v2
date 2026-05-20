# The ingestion program is the program that:
# 1. Take participant's code submission
# 2. Train the given model on the training data
# 3. Make predictions on the test data, and save them to forward them to the scoring program
# Imports
import json
import os
import sys
import time
import numpy as np

# Paths
input_dir      = '/app/input_data/'   # Data
output_dir     = '/app/output/'       # For the predictions
program_dir    = '/app/program'
submission_dir = '/app/ingested_program' # The code submitted

sys.path.append(output_dir)
sys.path.append(program_dir)
sys.path.append(submission_dir)


def get_data():
    """ Get X_train, y_train and X_test.
    """
    X_train = np.load(os.path.join(input_dir, 'X9_train_fl32.npy'))
    y_train = np.load(os.path.join(input_dir, 'rho_fl32.npy'))[:, 0]
    X_test  = np.load(os.path.join(input_dir, 'X9_test_fl32.npy'))
    return X_train, y_train, X_test


def print_bar():
    """ Display a bar ('----------')
    """
    print('-' * 10)


def main():
    """ The ingestion program.
    """
    print_bar()
    print('Ingestion program - ONERA 468 CRM challenge rho.')

    from model import Model # The model submitted by the participant

    start = time.time()

    print_bar()
    # Read data
    print('Reading data.')
    X_train, y_train, X_test = get_data()
    print(f'X_train : {X_train.shape}')
    print(f'y_train : {y_train.shape}')
    print(f'X_test  : {X_test.shape}')

    # Initialize model
    print_bar()
    print('Initializing the model.')
    model = Model()

    # Train model
    print_bar()
    print('Training the model.')
    model.fit(X_train, y_train)

    # Make predictions
    print_bar()
    print('Making predictions.')
    y_pred = model.predict(X_test)
    print(f'y_pred shape : {y_pred.shape}')

    # Save predictions
    y_pred = y_pred.reshape(-1, 1).astype(np.float32)
    np.save(os.path.join(output_dir, 'Yhat.npy'), y_pred)
    print(f'Yhat.npy saved : {y_pred.shape}')

    # End
    duration = time.time() - start
    print(f'Completed. Total duration: {duration:.2f}s')

    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:
        json.dump({'duration': duration}, f)

    print('Ingestion program finished. Moving on to scoring.')
    print_bar()


if __name__ == '__main__':
    main()