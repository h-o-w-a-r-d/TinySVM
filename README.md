# TinySVM 🚀

A lightweight, dependency-free (NumPy only), and educational implementation of Support Vector Machines (SVM) in pure Python.

**TinySVM** implements both Classification (SVC) and Regression (SVR) with a simplified SMO (Sequential Minimal Optimization) and Coordinate Descent algorithm. It serves as a great learning resource or a drop-in replacement for heavy libraries when you only need basic SVM functionality.

## ✨ Features

- **Zero Dependencies**: Only requires `numpy`. No `scikit-learn` or `scipy` needed.
- **Full-featured**:
  - Binary & Multi-class Classification (One-vs-Rest).
  - Regression (Single & Multi-output).
  - Kernel Support: Linear & RBF (Gaussian).
- **Production Ready-ish**:
  - Built-in **Auto-Scaling** (StandardScaler).
  - Probability estimates (`predict_proba`).
  - Scikit-learn compatible API (`fit`, `predict`, `score`).
- **Tiny**: Less than 300 lines of code.

## 📦 Installation

Just copy `tinysvm.py` to your project folder. Yes, it's that simple!

Dependencies:
```bash
pip install numpy
```

## ⚡ Quick Start

### Classification (XOR Problem)

```python
from tinysvm import TinySVM
import numpy as np

# XOR data (Linear Separable? No.)
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]

# Initialize with RBF kernel and Auto-Scaling
clf = TinySVM(mode='classification', kernel='rbf', gamma=2.0, C=10.0, scaling=True)
clf.fit(X, y)

print(f"Prediction: {clf.predict([[0, 1]])}") # Output: [1]
print(f"Accuracy:   {clf.score(X, y)}")       # Output: 1.0
```

### Regression

```python
# Simple Linear Regression
X = [[1], [2], [3], [4], [5]]
y = [3, 5, 7, 9, 11] # y = 2x + 1

reg = TinySVM(mode='regression', kernel='linear', C=50.0)
reg.fit(X, y)

print(f"Prediction for x=6: {reg.predict([[6]])}") # Should be close to 13
```

## ⚙️ API Reference

### `TinySVM(mode, C, kernel, gamma, scaling, ...)`

- `mode`: `'classification'` or `'regression'`.
- `C`: Regularization parameter (default `1.0`).
- `kernel`: `'rbf'` (default) or `'linear'`.
- `gamma`: Kernel coefficient for RBF.
- `scaling`: Boolean. If `True` (default), automatically scales data using Z-score.

### Methods
- `fit(X, y)`: Train the model.
- `predict(X)`: Predict class or value.
- `predict_proba(X)`: (Classification only) Estimate class probabilities.
- `score(X, y)`: Returns Accuracy (Classification) or R² (Regression).
- `save(path) / load(path)`: Save/Load model state.

## 📜 License

MIT License. Feel free to use it in your own projects!