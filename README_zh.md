# TinySVM 🚀

[English Version](README.md)

一個輕量級、零依賴（僅需 NumPy）、以教育為目的純 Python 支持向量機（SVM）實現。

**TinySVM** 同時實現了分類（SVC）與回歸（SVR），並採用簡化的 SMO（序列最小優化）與坐標下降算法。它可以作為極佳的學習資源，或者在僅需基礎 SVM 功能時，替代笨重的第三方庫。

## ✨ 特性

- **零依賴**：僅需 `numpy`。不需要 `scikit-learn` 或 `scipy`。
- **功能完備**：
  - 二元與多元分類（One-vs-Rest）。
  - 回歸（單輸出與多輸出）。
  - 核函數支持：線性（Linear）與 RBF（高斯核）。
- **準生產環境可用**：
  - 內置**自動縮放**（StandardScaler）。
  - 概率估計（`predict_proba`）。
  - 兼容 Scikit-learn 的 API（`fit`、`predict`、`score`）。
- **極簡**：代碼量少於 300 行。

## 📦 安裝

只需將 `tinysvm.py` 複製到您的項目文件夾中。沒錯，就是這麼簡單！

依賴項：
```bash
pip install numpy

```

## ⚡ 快速上手

### 分類（XOR 問題）

```python
from tinysvm import TinySVM
import numpy as np

# XOR 數據（線性可分？不。）
X = [[0, 0], [1, 1], [1, 0], [0, 1]]
y = [0, 0, 1, 1]

# 使用 RBF 核與自動縮放初始化
clf = TinySVM(mode='classification', kernel='rbf', gamma=2.0, C=10.0, scaling=True)
clf.fit(X, y)

print(f"預測結果: {clf.predict([[0, 1]])}") # 輸出: [1]
print(f"準確率:   {clf.score(X, y)}")       # 輸出: 1.0

```

### 回歸

```python
# 簡單線性回歸
X = [[1], [2], [3], [4], [5]]
y = [3, 5, 7, 9, 11] # y = 2x + 1

reg = TinySVM(mode='regression', kernel='linear', C=50.0)
reg.fit(X, y)

print(f"x=6 的預測結果: {reg.predict([[6]])}") # 結果應接近 13

```

## ⚙️ API 參考

### `TinySVM(mode, C, kernel, gamma, scaling, ...)`

* `mode`: `'classification'`（分類）或 `'regression'`（回歸）。
* `C`: 正則化參數（默認 `1.0`）。
* `kernel`: `'rbf'`（默認）或 `'linear'`（線性）。
* `gamma`: RBF 核函數系數。
* `scaling`: 布爾值。若為 `True`（默認），自動使用 Z-score 進行數據縮放。

### 方法

* `fit(X, y)`: 訓練模型。
* `predict(X)`: 預測類別或數值。
* `predict_proba(X)`: （僅限分類）估計類別概率。
* `score(X, y)`: 返回準確率（分類）或 R² 分數（回歸）。
* `save(path) / load(path)`: 保存/加載模型狀態。

## 📜 開源協議

MIT 協議。歡迎在您的項目中自由使用！
