# ONERA 468 CRM - Wall distribution regression challenge: Extra-packages

The Codabench environment provides the following packages by default:

- `Cython==3.2.8`
- `numpy==2.5.1`
- `matplotlib==3.11.0`
- `seaborn==0.13.2`
- `scipy==1.18.0`
- `scikit-learn==1.9.0`
- `pandas==3.0.3`
- `PyYAML==6.0.3`
- `imutils==0.5.4`
- `opencv-python==5.0.0.93`
- `torch==2.12.1`
- `tensorflow==2.21.0`
- `tqdm==4.68.4`
- `psutil==7.2.2`
- `h5py==3.14.0`
- `jupyter==1.1.1`

If your model requires packages beyond this list, you need to include them in a folder that should be names `python_packages` on the same level as your `model.py` file in your submission.
The `using_extra_packages/` folder in the starting kit provides ready-to-use examples for two workflows:

### With conda

1. Create a fresh environment from the provided `environement.yml`:
```bash
   conda env create -f environement.yml
   conda activate codabench-env
```
2. Create a `python_packages/` folder next to your `model.py`:
```bash
   mkdir python_packages
```
3. Install your package into that folder:
```bash
   pip install --target python_packages --no-deps lightgbm
```
4. Add the following lines at the top of your `model.py`:
```python
   import sys
   from pathlib import Path
   sys.path.append(str(Path(__file__).parent / "python_packages"))
```
5. Zip everything together and submit:
```bash
   cd PATH_TO_YOUR_FOLDER && zip -r ../submission.zip .
```

See `using_extra_packages/submission_example/conda_tuto.txt` for the full walkthrough and `using_extra_packages/submission_example/model.py` for a working example.

### With uv

1. Set up the environment from the provided `requirements.txt`:
```bash
   uv venv --python 3.12
   source .venv/bin/activate
   uv pip install -r requirements.txt
```
2. Follow steps 2–5 from the conda workflow above, replacing the install command with:
```bash
   uv pip install --target python_packages --no-deps lightgbm
```

See `using_extra_packages/submission_example/uv_tuto.txt` for the full walkthrough and `using_extra_packages/submission_example/model.py` for a working example.

## How to submit

Your submission must be a zip file containing a single file named `model.py`. This file must define a `Model` class with the following interface:

```python
class Model:
    def fit(self, X, y):
        ...
    def predict(self, X):
        ...
```

The `fit()` method receives `X_train` of shape [np×n_train, 9] and `Y_train` of shape [np×n_train], and the `predict()` method receives `X_test` of shape [np×n_test, 9] and must return a numpy array of shape [np×n_test].

To submit, compress your `model.py` file into a zip file and upload it in the **My Submissions** tab:

```bash
cd PATH_TO_YOUR_FOLDER && zip -r ../submission.zip .
```

A template `model.py` is available in the `using_extra_packages/submission_example/` folder of the starting kit.