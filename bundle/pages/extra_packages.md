# ONERA 468 CRM - Wall distribution regression challenge: Extra-packages

The Codabench environment provides the following packages by default:

- `Cython==3.0.12`
- `numpy==1.26.4`
- `scipy==1.11.4`
- `scikit-learn==1.5.1`
- `pandas==2.2.3`
- `pyyaml==6.0.2`
- `imutils==0.5.4`
- `numba==0.61.2`
- `threadpoolctl==3.6.0`
- `matplotlib==3.8.4`
- `psutil==7.0.0`

If your model requires packages beyond this list, you need to include them in a folder that should be names `python_packages` on the same level as your `model.py` file in your submission.
The `using_extra_packages/` folder in the starting kit provides ready-to-use examples for two workflows:

### With conda

1. Create a fresh environment from the provided `environment.yml`:
```bash
   conda env create -f environment.yml
   conda activate codabench-env
```
2. Create a `python_packages/` folder next to your `model.py`:
```bash
   mkdir python_packages
```
3. Install your package into that folder:
```bash
   pip install --target python_packages lightgbm
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

See `using_extra_packages/conda_submission_example.py` for a full working example.

### With uv

1. Set up the environment from the `codalab-env/` folder:
```bash
   cd codalab-env
   uv sync
   uv activate
```
2. Follow steps 2–5 from the conda workflow above, replacing the install command with:
```bash
   uv pip install --target python_packages lightgbm
```

See `using_extra_packages/uv_submission_example.py` for a full working example.

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

A template `model.py` is available in the `solution/` folder of the starting kit.