"""
Automated data download: fetches feedback_phase / final_phase input_data
straight from Codabench with `wget`, unzips them into the layout the rest
of kit_utils expects (`<phase>/input_data/...`), and deletes the
downloaded zip afterwards. Skips re-downloading if the target folder is
already populated.
"""

import os
import subprocess
import zipfile

# Codabench's stable per-dataset download endpoint -- it 302-redirects to a
# freshly-signed URL each time, so these links don't expire even though the
# redirect target does.
DATASET_URLS = {
    'feedback_phase': 'https://www.codabench.org/datasets/download/25e0fd03-1f60-4d60-8bc1-718f05f5ea9d/',
    'final_phase':    'https://www.codabench.org/datasets/download/338d9248-42cb-4dbb-bf40-1efb0413250c/',
}


def download_input_data(phase, dest_root='..', force=False):
    """ Download + unzip `phase`'s input_data from Codabench into
    `<dest_root>/<phase>/input_data/`, then delete the downloaded zip.
    Skips the download if that folder is already populated, unless
    force=True. Returns the input_data directory path. """
    if phase not in DATASET_URLS:
        raise ValueError(f'Unknown phase {phase!r}, expected one of {list(DATASET_URLS)}')

    input_dir = os.path.join(dest_root, phase, 'input_data')
    if not force and os.path.isdir(input_dir) and os.listdir(input_dir):
        print(f'[skip] {input_dir} already populated.')
        return input_dir

    os.makedirs(input_dir, exist_ok=True)
    zip_path = os.path.join(dest_root, f'{phase}_input_data.zip')

    print(f'[*] Downloading {phase} input_data from Codabench...')
    subprocess.run(['wget', '-q', '--show-progress', '-O', zip_path, DATASET_URLS[phase]], check=True)

    print(f'[*] Extracting to {input_dir} ...')
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith('__MACOSX/') or name.endswith('/'):
                continue  # skip macOS zip junk and directory entries
            zf.extract(name, input_dir)

    print(f'[*] Removing {zip_path}')
    os.remove(zip_path)

    return input_dir


def download_all(dest_root='..', force=False):
    """ Download both phases; returns {phase: input_data_dir}. """
    return {phase: download_input_data(phase, dest_root=dest_root, force=force) for phase in DATASET_URLS}
