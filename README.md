# Weighted Random Survival Forest

## Installation Guide

### Step 1: Install dependencies
```
$ conda install -c sebp scikit-survival
$ pip install numpy pandas lifelines scikit-learn cvxpy cython tqdm
```

### Step 2: Compile Cython functions
```
$ chmod +x compile_cython.sh
$ ./compile_cython.sh
```

## Usage

Modify `test_survival_forest.py` to choose datasets and parameter grid to test Weighted Random Survival Forest on.

```
$ python3 test_survival_forest.py
```