import unittest
import numpy as np
import os
from tinysvm import TinySVM

class TestTinySVM(unittest.TestCase):
    
    def test_classification_xor(self):
        """測試 XOR 問題 (非線性分類)"""
        # XOR data
        X = [[0, 0], [1, 1], [1, 0], [0, 1]]
        y = [0, 0, 1, 1]
        
        # 使用 RBF 核
        clf = TinySVM(mode='classification', kernel='rbf', gamma=2.0, C=10.0, scaling=True)
        clf.fit(X, y)
        
        # 驗證預測
        pred = clf.predict([[0, 1]])
        self.assertEqual(pred[0], 1)
        
        # 驗證準確率
        score = clf.score(X, y)
        self.assertEqual(score, 1.0)

    def test_linear_regression(self):
        """測試簡單線性回歸"""
        X = [[1], [2], [3], [4], [5]]
        y = [3, 5, 7, 9, 11] # y = 2x + 1

        reg = TinySVM(mode='regression', kernel='linear', C=50.0)
        reg.fit(X, y)
        
        # 預測 x=6, 應該接近 13
        pred = reg.predict([[6]])
        self.assertTrue(12.5 < pred[0] < 13.5, f"Prediction {pred[0]} is not close to 13")

    def test_save_load(self):
        """測試模型保存與加載"""
        X = [[1], [2], [3]]
        y = [1, 2, 3]
        model = TinySVM(mode='regression', kernel='linear')
        model.fit(X, y)
        
        filename = "test_model.pkl"
        model.save(filename)
        
        new_model = TinySVM()
        new_model.load(filename)
        
        self.assertTrue(new_model._fitted)
        self.assertTrue(np.allclose(new_model.predict([[2]]), model.predict([[2]])))
        
        # 清理文件
        if os.path.exists(filename):
            os.remove(filename)

if __name__ == '__main__':
    unittest.main()
