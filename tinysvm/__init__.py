import numpy as np
import pickle
import copy
import warnings

# --- 工具類 ---

class TinyScaler:
    """
    簡易的 StandardScaler (Z-score Normalization)。
    SVM 對特徵的尺度非常敏感，因此這是必須的組件。
    """
    def __init__(self):
        self.mean = None
        self.scale = None

    def fit(self, X):
        X = np.array(X, dtype=np.float32)
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        # 避免除以 0：如果某特徵方差為 0，則設其 scale 為 1
        self.scale[self.scale < 1e-8] = 1.0
        return self

    def transform(self, X):
        if self.mean is None or self.scale is None:
            raise ValueError("Scaler has not been fitted yet.")
        X = np.array(X, dtype=np.float32)
        return (X - self.mean) / self.scale

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# --- 核心求解器 ---

class BaseSVM:
    """SVM 基礎類，處理核函數與基礎參數"""
    def __init__(self, C=1.0, kernel='rbf', gamma=0.5, tol=1e-3, max_iter=100, epsilon=0.1):
        self.C = C
        self.kernel_type = kernel
        self.gamma = gamma
        self.tol = tol
        self.max_iter = max_iter
        self.epsilon = epsilon  # 僅用於 SVR
        
        # 模型參數
        self.coef = None        # 對偶係數 (alpha * y 或 beta)
        self.b = 0.0            # 偏置
        self.support_vectors = None
        self.weights = None     # 線性核權重 (w)

    def _compute_kernel(self, X1, X2):
        """向量化核函數計算"""
        if self.kernel_type == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel_type == 'rbf':
            # RBF Kernel: exp(-gamma * ||x1 - x2||^2)
            if X2.ndim == 1: X2 = X2[np.newaxis, :]
            
            # 使用 (a-b)^2 = a^2 + b^2 - 2ab 加速計算
            sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                       np.sum(X2**2, axis=1) - \
                       2 * np.dot(X1, X2.T)
            
            # 數值穩定性截斷
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def _prune(self, X):
        """剪枝：移除係數接近 0 的支持向量以壓縮模型 (僅限 Dual 模式使用)"""
        if self.coef is None: return

        # 篩選非零係數 (支持向量)
        mask = np.abs(self.coef) > 1e-5
        self.support_vectors = X[mask]
        self.coef = self.coef[mask]
        
        # 線性核優化：如果使用線性核，可以坍縮成單一權重向量 w
        if self.kernel_type == 'linear':
            # Regression: w = sum(coef * sv)
            # Classification: w = sum(alpha * y * sv)
            if len(self.coef) > 0:
                self.weights = np.dot(self.coef, self.support_vectors)
            else:
                self.weights = np.zeros(X.shape[1])
            
            # 釋放原始支持向量以節省內存
            self.support_vectors = None 
            self.coef = None

    def _predict_raw(self, X):
        """計算原始決策值 (wx + b 或 sum(coef * K) + b)"""
        X = np.array(X, dtype=np.float32)
        if X.ndim == 1: X = X[np.newaxis, :]
        
        if self.weights is not None:
            # 線性加速模式 (Primal 或 Pruned Dual)
            return np.dot(X, self.weights) + self.b
        else:
            # 核函數模式 (Dual)
            if self.support_vectors is None or len(self.support_vectors) == 0:
                return np.zeros(X.shape[0]) + self.b
            
            K = self._compute_kernel(self.support_vectors, X)
            return np.dot(self.coef, K) + self.b

class TinySVC(BaseSVM):
    """二元分類器 (Support Vector Classification)"""
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        
        # --- 優化分支：如果是線性核，使用快速的原始形式求解 (Primal Solver) ---
        if self.kernel_type == 'linear':
            self._fit_linear_primal(X, y)
            return
        
        # --- 默認分支：對偶形式求解 (Dual Solver) - 適用於 RBF ---
        n_samples = X.shape[0]
        self.coef = np.zeros(n_samples) 
        self.b = 0.0
        
        learning_rate = 0.01
        
        print(f"Warning: Training with {self.kernel_type} kernel on {n_samples} samples might be slow.")
        
        for iter_i in range(self.max_iter):
            alpha_changed = 0
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for i in indices:
                # 這裡的運算量非常大 O(N)
                K_i = self._compute_kernel(X, X[i:i+1]).flatten() 
                pred = np.dot((self.coef * y), K_i) + self.b
                
                error = pred.item() - y[i]
                
                if (y[i] * error < -self.tol and self.coef[i] < self.C) or \
                   (y[i] * error > self.tol and self.coef[i] > 0):
                    
                    grad = error * y[i]
                    self.coef[i] -= learning_rate * grad
                    self.coef[i] = np.clip(self.coef[i], 0, self.C)
                    self.b -= learning_rate * error * 0.5
                    alpha_changed += 1
            
            if alpha_changed == 0:
                break
                
        self.coef = self.coef * y
        self._prune(X)

    def _fit_linear_primal(self, X, y):
        """
        針對線性核的特化求解器 (Primal SGD / Pegasos 變體)。
        複雜度從 O(N^2) 降低到 O(N * features)，速度極快。
        目標函數: min 0.5*||w||^2 + C * sum(max(0, 1 - y(wx+b)))
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.b = 0.0
        
        # 初始學習率
        eta0 = 0.01
        # Lambda 是正則化強度的倒數，對應 SVM 的 C
        # 這裡做一個簡單轉換，讓 C 的行為跟 sklearn 類似
        # 一般來說 lambda = 1 / (n_samples * C)
        lambda_param = 1.0 / (self.C * n_samples) if self.C > 0 else 0.0
        
        for epoch in range(self.max_iter):
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            # 學習率衰減
            eta = eta0 / (1 + lambda_param * epoch * 10)
            
            count_updates = 0
            for i in indices:
                decision = np.dot(X[i], self.weights) + self.b
                
                # Hinge Loss 判斷: y * f(x) < 1 代表分類錯誤或在邊界內
                if y[i] * decision < 1.0:
                    # 梯度下降更新
                    # w = w + eta * (y * x - 2 * lambda * w) (包含 L2 正則化)
                    # 這裡簡化係數以確保數值穩定
                    self.weights = (1 - eta * lambda_param) * self.weights + (eta * self.C) * y[i] * X[i]
                    self.b += (eta * self.C) * y[i] # Bias 通常不加正則化
                    count_updates += 1
                else:
                    # 就算分類正確，也要做正則化衰減
                    self.weights = (1 - eta * lambda_param) * self.weights
            
            # 簡易收斂檢測
            if count_updates == 0:
                break

    def predict(self, X):
        decision = self._predict_raw(X)
        return np.sign(decision).astype(int)

class TinySVR(BaseSVM):
    """回歸器 (Epsilon-Support Vector Regression)"""
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        n_samples = X.shape[0]
        
        # SVR 尚未實作 Primal Linear Solver，因此保持原樣
        # 如果需要跑大量數據的回歸，建議也需要針對 Linear 進行類似優化
        self.coef = np.zeros(n_samples) 
        self.b = np.mean(y)
        
        kernel_diag = np.diag(self._compute_kernel(X, X))
        
        for iter_idx in range(self.max_iter * 2):
            max_change = 0
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in indices:
                pred = np.dot(self.coef, self._compute_kernel(X, X[i:i+1])) + self.b
                residual = y[i] - pred.item()
                
                if abs(residual) < self.epsilon:
                    continue
                
                denom = kernel_diag[i]
                if denom < 0.01: denom = 0.01
                
                step = (residual - np.sign(residual) * self.epsilon) / denom
                step = np.clip(step, -self.C, self.C)

                new_coef = self.coef[i] + step * 0.1
                new_coef = np.clip(new_coef, -self.C, self.C)
                
                change = abs(new_coef - self.coef[i])
                if change > max_change:
                    max_change = change
                    
                self.coef[i] = new_coef
                self.b += step * 0.05

            if max_change < self.tol:
                break
                
        self._prune(X)

    def predict(self, X):
        return self._predict_raw(X)

# --- 高級包裝器 ---

class TinySVM:
    """
    全能包裝器：自動處理縮放、分類、回歸、多分類、多輸出。
    建議使用者直接使用此類。
    """
    def __init__(self, mode='classification', C=1.0, kernel='rbf', gamma=0.5, scaling=True, **kwargs):
        self.mode = mode
        self.scaling = scaling
        self.params = {'C': C, 'kernel': kernel, 'gamma': gamma, **kwargs}
        
        self.models = []       # 子模型列表
        self.scaler = TinyScaler() if scaling else None
        self.is_multi_output = False
        self.classes = None    # 僅分類用
        self._fitted = False

    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        if self.scaling:
            X = self.scaler.fit_transform(X)

        if self.mode == 'regression' and y.ndim > 1 and y.shape[1] > 1:
            self.is_multi_output = True
            n_outputs = y.shape[1]
            self.models = []
            for i in range(n_outputs):
                model = TinySVR(**self.params)
                model.fit(X, y[:, i])
                self.models.append(model)
        
        elif self.mode == 'classification':
            self.classes = np.unique(y)
            n_classes = len(self.classes)
            
            if n_classes > 2:
                self.models = []
                for cls in self.classes:
                    y_binary = np.where(y == cls, 1, -1)
                    model = TinySVC(**self.params)
                    model.fit(X, y_binary)
                    self.models.append(model)
            else:
                y_mapped = np.where(y == self.classes[0], -1, 1)
                model = TinySVC(**self.params)
                model.fit(X, y_mapped)
                self.models = [model]

        elif self.mode == 'regression':
            model = TinySVR(**self.params)
            model.fit(X, y)
            self.models = [model]
            
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self._fitted = True
        return self

    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
            
        X = np.array(X)
        if self.scaling:
            X = self.scaler.transform(X)

        if self.is_multi_output:
            preds = [m.predict(X) for m in self.models]
            return np.column_stack(preds)
        
        if self.mode == 'classification':
            if len(self.classes) > 2:
                decision_values = np.column_stack([m._predict_raw(X) for m in self.models])
                indices = np.argmax(decision_values, axis=1)
                return self.classes[indices]
            else:
                raw = self.models[0].predict(X)
                return np.where(raw == -1, self.classes[0], self.classes[1])

        return self.models[0].predict(X)

    def decision_function(self, X):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
        
        X = np.array(X)
        if self.scaling:
            X = self.scaler.transform(X)
            
        if self.mode == 'classification':
            if len(self.classes) > 2:
                return np.column_stack([m._predict_raw(X) for m in self.models])
            else:
                return self.models[0]._predict_raw(X)
        else:
            raise NotImplementedError("Decision function not available for regression")

    def predict_proba(self, X):
        if self.mode != 'classification':
            raise AttributeError("predict_proba only available for classification")
            
        decisions = self.decision_function(X)
        
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if len(self.classes) > 2:
            probs = sigmoid(decisions)
            row_sums = probs.sum(axis=1, keepdims=True)
            return probs / row_sums
        else:
            prob_pos = sigmoid(decisions)
            prob_neg = 1 - prob_pos
            return np.column_stack([prob_neg, prob_pos])

    def score(self, X, y):
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
            
        y_pred = self.predict(X)
        y = np.array(y)
        
        if self.mode == 'classification':
            return np.mean(y_pred == y)
        else:
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - np.mean(y)) ** 2).sum() + 1e-10
            return 1 - (ss_res / ss_tot)

    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)
        print(f"Model loaded from {filepath}")