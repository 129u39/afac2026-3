"""V5 组件测试：特征筛选、损失函数、树模型。"""

import pytest
import numpy as np
import torch


class TestFeatureSelection:
    """测试特征筛选模块。"""

    def test_variance_selector(self):
        """测试低方差特征过滤。"""
        from features.variance_selector import VarianceSelector

        # 创建测试数据：第0列方差极低
        features = np.array([
            [1.0, 10.0, 100.0],
            [1.0, 20.0, 200.0],
            [1.0, 30.0, 300.0],
            [1.0, 40.0, 400.0],
        ])

        selector = VarianceSelector(threshold=0.01)
        selected = selector.fit_transform(features)

        # 第0列应该被过滤掉
        assert selected.shape[1] == 2
        print(f"VarianceSelector: {features.shape} -> {selected.shape}")

    def test_lgb_selector(self):
        """测试 LightGBM 特征筛选。"""
        from features.lgb_selector import LGBSelector

        # 创建测试数据
        np.random.seed(42)
        features = np.random.randn(100, 50)
        labels = np.random.randint(0, 3, 100)

        selector = LGBSelector(n_top_features=10)
        selected = selector.fit_transform(features, labels)

        assert selected.shape == (100, 10)
        assert len(selector.get_selected_indices()) == 10
        print(f"LGBSelector: {features.shape} -> {selected.shape}")

    def test_xgb_selector(self):
        """测试 XGBoost 特征筛选。"""
        from features.xgb_selector import XGBSelector

        np.random.seed(42)
        features = np.random.randn(100, 50)
        labels = np.random.randint(0, 3, 100)

        selector = XGBSelector(n_top_features=10)
        selected = selector.fit_transform(features, labels)

        assert selected.shape == (100, 10)
        assert len(selector.get_selected_indices()) == 10
        print(f"XGBSelector: {features.shape} -> {selected.shape}")

    def test_feature_cache(self):
        """测试特征缓存。"""
        import tempfile
        import os
        from features.feature_cache import FeatureCache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = FeatureCache(cache_dir=tmpdir)

            features = np.random.randn(10, 5)
            cache.save("test", features)

            assert cache.exists("test")
            loaded = cache.load("test")
            assert np.allclose(features, loaded)
            print("FeatureCache: save/load OK")


class TestLossFunctions:
    """测试损失函数。"""

    def test_focal_loss(self):
        """测试 Focal Loss。"""
        from losses.focal_loss import FocalLoss

        loss_fn = FocalLoss(gamma=2.0)

        # 模拟预测和标签
        inputs = torch.randn(8, 10)
        targets = torch.randint(0, 10, (8,))

        loss = loss_fn(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print(f"FocalLoss: {loss.item():.4f}")

    def test_weighted_ce(self):
        """测试加权交叉熵。"""
        from losses.weighted_ce import WeightedCrossEntropyLoss

        # 模拟类别分布
        class_counts = np.array([100, 500, 1000, 50, 200])
        loss_fn = WeightedCrossEntropyLoss(class_counts)

        inputs = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))

        loss = loss_fn(inputs, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)
        print(f"WeightedCE: {loss.item():.4f}")


class TestTreeModels:
    """测试树模型。"""

    def test_lightgbm_model(self):
        """测试 LightGBM 模型。"""
        from models.lightgbm_model import LightGBMModel

        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(20, 20)

        model = LightGBMModel(n_estimators=50, max_depth=3)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert len(preds) == 20
        assert proba.shape == (20, 3)
        print(f"LightGBM: predict {len(preds)} samples, proba shape {proba.shape}")

    def test_xgboost_model(self):
        """测试 XGBoost 模型。"""
        from models.xgboost_model import XGBoostModel

        np.random.seed(42)
        X_train = np.random.randn(100, 20)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(20, 20)

        model = XGBoostModel(n_estimators=50, max_depth=3)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        assert len(preds) == 20
        assert proba.shape == (20, 3)
        print(f"XGBoost: predict {len(preds)} samples, proba shape {proba.shape}")


class TestIntegration:
    """集成测试：特征筛选 + 树模型。"""

    def test_feature_selection_pipeline(self):
        """测试特征筛选流水线。"""
        from features.variance_selector import VarianceSelector
        from features.lgb_selector import LGBSelector

        np.random.seed(42)
        features = np.random.randn(200, 100)
        labels = np.random.randint(0, 5, 200)

        # 第一步：低方差过滤
        var_selector = VarianceSelector(threshold=1e-5)
        features_filtered = var_selector.fit_transform(features)
        print(f"Step 1 - VarianceThreshold: {features.shape} -> {features_filtered.shape}")

        # 第二步：LightGBM 特征筛选
        lgb_selector = LGBSelector(n_top_features=20)
        features_selected = lgb_selector.fit_transform(features_filtered, labels)
        print(f"Step 2 - LGBSelector: {features_filtered.shape} -> {features_selected.shape}")

        # 第三步：LightGBM 分类
        from models.lightgbm_model import LightGBMModel
        model = LightGBMModel(n_estimators=100)
        model.fit(features_selected, labels)

        preds = model.predict(features_selected)
        accuracy = (preds == labels).mean()
        print(f"Step 3 - LightGBM: accuracy={accuracy:.4f}")

        assert accuracy > 0.5  # 随机数据，5类应该>20%
