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
        self.weights = None     # 線性核權重 (prune 後生成)

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
            
            # 數值穩定性截斷，防止負數開根號（雖這裡是算距離平方，但浮點誤差可能導致負極小值）
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-self.gamma * sq_dists)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")

    def _prune(self, X):
        """剪枝：移除係數接近 0 的支持向量以壓縮模型"""
        # 篩選非零係數 (支持向量)
        mask = np.abs(self.coef) > 1e-5
        self.support_vectors = X[mask]
        self.coef = self.coef[mask]
        
        # 線性核優化：如果使用線性核，可以坍縮成單一權重向量 w
        if self.kernel_type == 'linear':
            # Regression: w = sum(coef * sv)
            # Classification: w = sum(alpha * y * sv) (fit 中 coef 已經是 alpha*y)
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
            # 線性加速模式
            return np.dot(X, self.weights) + self.b
        else:
            # 核函數模式
            if self.support_vectors is None or len(self.support_vectors) == 0:
                return np.zeros(X.shape[0]) + self.b
            
            K = self._compute_kernel(self.support_vectors, X)
            # coef shape: (n_sv,), K shape: (n_sv, n_samples)
            return np.dot(self.coef, K) + self.b

class TinySVC(BaseSVM):
    """二元分類器 (Support Vector Classification)"""
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        n_samples = X.shape[0]
        
        # 初始化對偶係數 alpha
        self.coef = np.zeros(n_samples) 
        self.b = 0.0
        
        # 隨機梯度下降 / 坐標下降 (Pegasos / Simplified SMO 混合體)
        learning_rate = 0.01
        
        for iter_i in range(self.max_iter):
            alpha_changed = 0
            # 隨機打亂順序以避免循環震盪
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            
            for i in indices:
                # 計算當前預測值 f(x_i)
                K_i = self._compute_kernel(X, X[i:i+1]).flatten() 
                pred = np.dot((self.coef * y), K_i) + self.b
                
                # [FIX] 使用 .item() 避免 NumPy DeprecationWarning
                error = pred.item() - y[i]
                
                # 違反 KKT 條件才進行更新
                if (y[i] * error < -self.tol and self.coef[i] < self.C) or \
                   (y[i] * error > self.tol and self.coef[i] > 0):
                    
                    # 計算梯度：Loss 對 alpha 的梯度
                    grad = error * y[i]
                    
                    # 更新 alpha
                    self.coef[i] -= learning_rate * grad
                    
                    # 投影到 [0, C]
                    self.coef[i] = np.clip(self.coef[i], 0, self.C)
                    
                    # 更新 Bias (簡單移動平均)
                    self.b -= learning_rate * error * 0.5
                    
                    alpha_changed += 1
            
            # 如果收斂了 (幾乎沒有 alpha 改變)，提前停止
            if alpha_changed == 0:
                break
                
        # 轉換 alpha 為 alpha * y 存儲，方便推理時直接做點積
        self.coef = self.coef * y
        self._prune(X)

    def predict(self, X):
        decision = self._predict_raw(X)
        return np.sign(decision).astype(int)

class TinySVR(BaseSVM):
    """回歸器 (Epsilon-Support Vector Regression)"""
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        n_samples = X.shape[0]
        
        # SVR 的對偶係數 beta = (alpha - alpha*)
        # 範圍是 [-C, C]
        self.coef = np.zeros(n_samples) 
        self.b = np.mean(y) # 初始 b 設為均值
        
        # 預計算核矩陣對角線 (用於計算步長)
        kernel_diag = np.diag(self._compute_kernel(X, X))
        
        # 坐標下降法 (Dual Coordinate Descent)
        for iter_idx in range(self.max_iter * 2):
            max_change = 0
            indices = np.arange(n_samples)
            np.random.shuffle(indices)

            for i in indices:
                # 計算當前殘差
                pred = np.dot(self.coef, self._compute_kernel(X, X[i:i+1])) + self.b
                
                # [FIX] 使用 .item() 獲取標量
                residual = y[i] - pred.item()
                
                # 根據 Epsilon-Insensitive Loss 更新
                if abs(residual) < self.epsilon:
                    continue
                
                # 計算步長: delta = residual / K_ii
                # [FIX] 數值穩定性修復：
                # 當使用 Linear Kernel + Scaler 時，數據點可能接近 0，導致 K_ii = 0。
                # 這裡強制分母最小為 0.01，防止除以零爆炸。
                denom = kernel_diag[i]
                if denom < 0.01: denom = 0.01
                
                step = (residual - np.sign(residual) * self.epsilon) / denom
                
                # [FIX] 限制單步更新幅度，防止震盪
                step = np.clip(step, -self.C, self.C)

                new_coef = self.coef[i] + step * 0.1 # 學習率衰減
                
                # 截斷 [-C, C]
                new_coef = np.clip(new_coef, -self.C, self.C)
                
                change = abs(new_coef - self.coef[i])
                if change > max_change:
                    max_change = change
                    
                self.coef[i] = new_coef
                
                # 更新 Bias (跟隨殘差移動)
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
        
        # 1. 自動特徵縮放
        if self.scaling:
            X = self.scaler.fit_transform(X)

        # 2. 模式檢測與訓練
        
        # --- 多輸出回歸 (y 是矩陣) ---
        if self.mode == 'regression' and y.ndim > 1 and y.shape[1] > 1:
            self.is_multi_output = True
            n_outputs = y.shape[1]
            self.models = []
            for i in range(n_outputs):
                model = TinySVR(**self.params)
                model.fit(X, y[:, i])
                self.models.append(model)
        
        # --- 分類 (二元或多類) ---
        elif self.mode == 'classification':
            self.classes = np.unique(y)
            n_classes = len(self.classes)
            
            if n_classes > 2:
                # 多分類 (One-vs-Rest)
                self.models = []
                for cls in self.classes:
                    # 創建二元標籤: 當前類為 1，其他為 -1
                    y_binary = np.where(y == cls, 1, -1)
                    model = TinySVC(**self.params)
                    model.fit(X, y_binary)
                    self.models.append(model)
            else:
                # 二分類
                # 映射標籤為 {-1, 1}
                y_mapped = np.where(y == self.classes[0], -1, 1)
                model = TinySVC(**self.params)
                model.fit(X, y_mapped)
                self.models = [model]

        # --- 普通單輸出回歸 ---
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
        # 預測時必須應用相同的縮放
        if self.scaling:
            X = self.scaler.transform(X)

        # 多輸出回歸
        if self.is_multi_output:
            preds = [m.predict(X) for m in self.models]
            return np.column_stack(preds)
        
        # 分類
        if self.mode == 'classification':
            if len(self.classes) > 2:
                # 多分類 (OvR): 取決策值最大的類別
                decision_values = np.column_stack([m._predict_raw(X) for m in self.models])
                indices = np.argmax(decision_values, axis=1)
                return self.classes[indices]
            else:
                # 二分類
                raw = self.models[0].predict(X)
                # 映射回原始類別
                return np.where(raw == -1, self.classes[0], self.classes[1])

        # 普通回歸
        return self.models[0].predict(X)

    def decision_function(self, X):
        """獲取原始決策值 (距離超平面的距離)"""
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
        """
        輸出機率估計 (使用 Sigmoid 函數近似)
        注意：這不是精確的 Platt Scaling，但在小樣本下足夠有效。
        """
        if self.mode != 'classification':
            raise AttributeError("predict_proba only available for classification")
            
        decisions = self.decision_function(X)
        
        # Sigmoid 函數: 1 / (1 + e^-x)
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        if len(self.classes) > 2:
            # 多分類：對每個 OvR 結果做 Sigmoid，然後歸一化
            probs = sigmoid(decisions)
            # 歸一化使機率和為 1
            row_sums = probs.sum(axis=1, keepdims=True)
            return probs / row_sums
        else:
            # 二分類
            # decisions > 0 代表 class[1], < 0 代表 class[0]
            prob_pos = sigmoid(decisions) # class[1] 的機率
            prob_neg = 1 - prob_pos       # class[0] 的機率
            return np.column_stack([prob_neg, prob_pos])

    def score(self, X, y):
        """
        計算模型得分
        分類：Accuracy
        回歸：R^2 Score
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted yet.")
            
        y_pred = self.predict(X)
        y = np.array(y)
        
        if self.mode == 'classification':
            return np.mean(y_pred == y)
        else:
            # R^2 Score
            ss_res = ((y - y_pred) ** 2).sum()
            ss_tot = ((y - np.mean(y)) ** 2).sum() + 1e-10 # 防止除零
            return 1 - (ss_res / ss_tot)

    def save(self, filepath):
        """保存模型狀態"""
        with open(filepath, 'wb') as f:
            pickle.dump(self.__dict__, f)
        print(f"Model saved to {filepath}")

    def load(self, filepath):
        """加載模型狀態"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.__dict__.update(data)
        print(f"Model loaded from {filepath}")
