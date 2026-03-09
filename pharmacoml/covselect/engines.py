"""
ML engine abstraction — 8 pluggable backends for covariate screening.

Classical methods (6): XGBoost, LightGBM, CatBoost, Random Forest,
                       Elastic Net, Lasso
Deep learning (2):    TabNet, MLP

All engines support SHAP values for functional form detection, and all
implement a common ``predict`` method so downstream filtering can score
incremental utility consistently across backends.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from importlib.util import find_spec
import numpy as np
import shap
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler


class BaseEngine(ABC):
    """Abstract base for all ML engines."""

    @abstractmethod
    def fit(self, X, y): ...

    @abstractmethod
    def predict(self, X): ...

    @abstractmethod
    def shap_values(self, X): ...

    @abstractmethod
    def feature_importances(self, feature_names): ...

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Engine", "")


# ─────────────────────────────────────────────────────────────
# Tree-based engines (gradient boosting variants)
# ─────────────────────────────────────────────────────────────

class XGBoostEngine(BaseEngine):
    """XGBoost gradient boosted trees."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        from xgboost import XGBRegressor
        self._model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=1,
            random_state=self.random_state, verbosity=0)
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.TreeExplainer(self._model).shap_values(X)

    def feature_importances(self, feature_names):
        return dict(zip(feature_names, self._model.feature_importances_))


class LightGBMEngine(BaseEngine):
    """LightGBM — histogram-based gradient boosting.

    Uses leaf-wise tree growth (vs XGBoost's level-wise), often faster
    and can find different splits. Provides independent signal from XGBoost.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        self._model = LGBMRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1,
            n_jobs=1,
            random_state=self.random_state, verbose=-1, force_col_wise=True)
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.TreeExplainer(self._model).shap_values(X)

    def feature_importances(self, feature_names):
        return dict(zip(feature_names, self._model.feature_importances_))


class CatBoostEngine(BaseEngine):
    """CatBoost — gradient boosting with native categorical support.

    Uses ordered boosting (reduces prediction shift), symmetric trees,
    and handles categoricals without one-hot encoding. Popular in pharma
    for handling mixed covariate types (continuous + categorical).
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        from catboost import CatBoostRegressor, Pool
        self._model = CatBoostRegressor(
            iterations=200, depth=4, learning_rate=0.1,
            subsample=0.8, rsm=0.8,  # rsm = column sampling ratio
            random_seed=self.random_state, verbose=0,
            allow_writing_files=False)
        self._model.fit(X, y)
        self._train_pool = Pool(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        from catboost import Pool
        pool = Pool(X)
        sv = self._model.get_feature_importance(data=pool, type="ShapValues")
        return sv[:, :-1]  # last column is base value

    def _shap_fallback(self, X):
        """Fallback to standard SHAP TreeExplainer if native fails."""
        return shap.TreeExplainer(self._model).shap_values(X)

    def feature_importances(self, feature_names):
        imp = self._model.get_feature_importance()
        # CatBoost returns raw importance, normalize to sum to 1
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


# ─────────────────────────────────────────────────────────────
# Ensemble tree engine
# ─────────────────────────────────────────────────────────────

class RandomForestEngine(BaseEngine):
    """Random Forest regression — bagged decision trees."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        self._model = RandomForestRegressor(
            n_estimators=300, min_samples_leaf=5,
            random_state=self.random_state, n_jobs=1)
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.TreeExplainer(self._model).shap_values(X)

    def feature_importances(self, feature_names):
        return dict(zip(feature_names, self._model.feature_importances_))


class ExtraTreesEngine(BaseEngine):
    """Extra Trees regression — randomized bagged trees."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        self._model = ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=4,
            random_state=self.random_state,
            n_jobs=1,
        )
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.TreeExplainer(self._model).shap_values(X)

    def feature_importances(self, feature_names):
        return dict(zip(feature_names, self._model.feature_importances_))


class GradientBoostingEngine(BaseEngine):
    """Scikit-learn gradient boosting for an all-local baseline."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        self._model = GradientBoostingRegressor(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.85,
            random_state=self.random_state,
        )
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.TreeExplainer(self._model).shap_values(X)

    def feature_importances(self, feature_names):
        return dict(zip(feature_names, self._model.feature_importances_))


# ─────────────────────────────────────────────────────────────
# Regularized linear engines
# ─────────────────────────────────────────────────────────────

class ElasticNetEngine(BaseEngine):
    """Elastic Net — L1+L2 regularized regression (mixed penalty)."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        self._model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1.0],
            cv=5, random_state=self.random_state, max_iter=5000)
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.LinearExplainer(self._model, X).shap_values(X)

    def feature_importances(self, feature_names):
        imp = np.abs(self._model.coef_)
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


class LassoEngine(BaseEngine):
    """Lasso — pure L1 regularized regression (sparse solutions).

    The classic sparsity-inducing method. Drives irrelevant covariate
    coefficients to exactly zero. Provides a different sparsity profile
    than Elastic Net, especially with correlated covariates (Lasso tends
    to pick one from a correlated group, Elastic Net keeps both).
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        self._model = LassoCV(
            cv=5, random_state=self.random_state, max_iter=5000)
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.LinearExplainer(self._model, X).shap_values(X)

    def feature_importances(self, feature_names):
        imp = np.abs(self._model.coef_)
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


class AdaptiveLassoEngine(BaseEngine):
    """Adaptive LASSO using ridge-initialized feature weights."""

    def __init__(self, random_state=42, gamma: float = 1.0, augmentation_strength: float = 0.0):
        self.random_state = random_state
        self.gamma = gamma
        self.augmentation_strength = augmentation_strength
        self._scaler = None
        self._coef_std = None
        self._intercept = 0.0
        self._feature_names = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float).ravel()

        self._scaler = StandardScaler()
        X_std = self._scaler.fit_transform(X_arr)
        y_mean = float(np.mean(y_arr))
        y_centered = y_arr - y_mean

        ridge = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5, fit_intercept=False)
        ridge.fit(X_std, y_centered)
        beta_init = np.abs(np.asarray(ridge.coef_).ravel())
        weights = 1.0 / np.power(np.maximum(beta_init, 1e-4), self.gamma)

        X_weighted = X_std / weights
        y_fit = y_centered
        if self.augmentation_strength > 0:
            aug = np.sqrt(self.augmentation_strength) * np.eye(X_weighted.shape[1])
            X_weighted = np.vstack([X_weighted, aug])
            y_fit = np.concatenate([y_centered, np.zeros(X_weighted.shape[1])])

        model = LassoCV(
            cv=5,
            random_state=self.random_state,
            max_iter=5000,
            fit_intercept=False,
        )
        model.fit(X_weighted, y_fit)
        theta = np.asarray(model.coef_).ravel()
        self._coef_std = theta / weights
        self._intercept = y_mean

    def predict(self, X):
        X_std = self._scaler.transform(np.asarray(X, dtype=float))
        return np.asarray(self._intercept + X_std @ self._coef_std, dtype=float).ravel()

    def shap_values(self, X):
        X_std = self._scaler.transform(np.asarray(X, dtype=float))
        return X_std * self._coef_std.reshape(1, -1)

    def feature_importances(self, feature_names):
        self._feature_names = list(feature_names)
        imp = np.abs(self._coef_std)
        total = float(imp.sum())
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


class AALassoEngine(AdaptiveLassoEngine):
    """Augmented adaptive LASSO tuned for correlated covariates."""

    def __init__(self, random_state=42, gamma: float = 1.0, augmentation_strength: float = 0.10):
        super().__init__(
            random_state=random_state,
            gamma=gamma,
            augmentation_strength=augmentation_strength,
        )


class RidgeEngine(BaseEngine):
    """Ridge regression baseline for dense linear signal."""

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None

    def fit(self, X, y):
        self._model = RidgeCV(alphas=np.logspace(-4, 4, 41), cv=5)
        self._model.fit(X, y)

    def predict(self, X):
        return np.asarray(self._model.predict(X)).ravel()

    def shap_values(self, X):
        return shap.LinearExplainer(self._model, X).shap_values(X)

    def feature_importances(self, feature_names):
        imp = np.abs(np.asarray(self._model.coef_).ravel())
        total = float(imp.sum())
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


class STGEngine(BaseEngine):
    """Stochastic-gates regressor with a small nonlinear MLP body.

    Gates still sit on the input layer, but the prediction head is now a
    compact MLP so the engine can represent nonlinear covariate effects more
    faithfully than the original linear-only variant.
    """

    def __init__(
        self,
        random_state=42,
        n_epochs: int = 200,
        lr: float = 0.03,
        lambda_sparse: float = 0.01,
        lambda_l2: float = 0.001,
        temperature: float = 0.5,
        hidden_layers: tuple[int, ...] = (16, 8),
        dropout: float = 0.10,
    ):
        self.random_state = random_state
        self.n_epochs = n_epochs
        self.lr = lr
        self.lambda_sparse = lambda_sparse
        self.lambda_l2 = lambda_l2
        self.temperature = temperature
        self.hidden_layers = tuple(hidden_layers)
        self.dropout = dropout
        self._scaler = None
        self._model = None
        self._y_mean = 0.0
        self._y_std = 1.0
        self._input_importance = None
        self._input_signed_importance = None

    def fit(self, X, y):
        import torch
        import torch.nn as nn

        torch.manual_seed(self.random_state or 42)
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).ravel()

        self._scaler = StandardScaler()
        X_std = self._scaler.fit_transform(X_arr).astype(np.float32)
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std() + 1e-6)
        y_scaled = ((y_arr - self._y_mean) / self._y_std).astype(np.float32)

        X_tensor = torch.tensor(X_std)
        y_tensor = torch.tensor(y_scaled)

        class _GateRegressor(nn.Module):
            def __init__(self, n_features: int, hidden_layers: tuple[int, ...], dropout: float):
                super().__init__()
                self.log_alpha = nn.Parameter(torch.zeros(n_features))
                layers: list[nn.Module] = []
                in_features = n_features
                for hidden in hidden_layers:
                    layers.append(nn.Linear(in_features, hidden))
                    layers.append(nn.ReLU())
                    if dropout > 0:
                        layers.append(nn.Dropout(dropout))
                    in_features = hidden
                self.hidden = nn.Sequential(*layers)
                self.output = nn.Linear(in_features, 1)

            def gate_prob(self):
                return torch.sigmoid(self.log_alpha)

            def sampled_gate(self, training: bool, temperature: float):
                if training:
                    noise = torch.rand_like(self.log_alpha).clamp_(1e-6, 1 - 1e-6)
                    logistic = torch.log(noise) - torch.log1p(-noise)
                    return torch.sigmoid((self.log_alpha + logistic) / temperature)
                return self.gate_prob()

            def forward(self, x, training: bool, temperature: float):
                gate = self.sampled_gate(training=training, temperature=temperature)
                hidden = x * gate
                if len(self.hidden):
                    hidden = self.hidden(hidden)
                return self.output(hidden).squeeze(-1)

            def input_statistics(self):
                gate = self.gate_prob()
                if len(self.hidden):
                    first_linear = next(module for module in self.hidden if isinstance(module, nn.Linear))
                    weight = first_linear.weight
                else:
                    weight = self.output.weight
                magnitude = gate * weight.abs().sum(dim=0)
                signed = gate * weight.sum(dim=0)
                return gate, magnitude, signed

        model = _GateRegressor(
            X_std.shape[1],
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        for _ in range(self.n_epochs):
            optimizer.zero_grad()
            pred = model(X_tensor, training=True, temperature=self.temperature)
            mse = torch.mean((pred - y_tensor) ** 2)
            gate_prob = model.gate_prob()
            weight_penalty = torch.zeros(1, dtype=pred.dtype, device=pred.device)
            for name, param in model.named_parameters():
                if "log_alpha" in name:
                    continue
                weight_penalty = weight_penalty + torch.sum(param ** 2)
            penalty = self.lambda_sparse * gate_prob.sum() + self.lambda_l2 * weight_penalty
            loss = mse + penalty
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            model.eval()
            _, magnitude, signed = model.input_statistics()
            self._input_importance = magnitude.detach().cpu().numpy()
            self._input_signed_importance = signed.detach().cpu().numpy()

        self._model = model

    def predict(self, X):
        import torch

        X_std = self._scaler.transform(np.asarray(X, dtype=np.float32)).astype(np.float32)
        with torch.no_grad():
            self._model.eval()
            preds = self._model(torch.tensor(X_std), training=False, temperature=self.temperature)
        preds = preds.detach().cpu().numpy().ravel()
        return np.asarray(self._y_mean + preds * self._y_std, dtype=float).ravel()

    def shap_values(self, X):
        X_std = self._scaler.transform(np.asarray(X, dtype=float))
        signed = np.asarray(self._input_signed_importance, dtype=float).ravel()
        if not np.any(np.abs(signed) > 0):
            signed = np.asarray(self._input_importance, dtype=float).ravel()
        return X_std * signed.reshape(1, -1)

    def feature_importances(self, feature_names):
        imp = np.asarray(self._input_importance, dtype=float).ravel()
        total = float(imp.sum())
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


# ─────────────────────────────────────────────────────────────
# Deep learning engines
# ─────────────────────────────────────────────────────────────

class TabNetEngine(BaseEngine):
    """TabNet — attention-based deep learning for tabular data (Google, AAAI 2021).

    Uses sequential attention to select features at each decision step.
    Provides built-in interpretability via attention masks — no SHAP needed
    for feature importance (though we compute SHAP too for consistency).

    Key advantage over tree methods: learns feature interactions through
    attention, can detect non-linear relationships that trees approximate
    with step functions.
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None
        self._X_train = None

    def fit(self, X, y):
        from pytorch_tabnet.tab_model import TabNetRegressor
        self._model = TabNetRegressor(
            n_d=8, n_a=8,           # width of attention and prediction layers
            n_steps=3,               # number of sequential attention steps
            gamma=1.3,               # coefficient for feature reusage
            n_independent=1,
            n_shared=1,
            lambda_sparse=1e-3,      # sparsity regularization
            seed=self.random_state,
            verbose=0,
            optimizer_params=dict(lr=2e-2),
            scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=None,
        )
        # TabNet requires numpy arrays and reshape for single output
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.float32).reshape(-1, 1)
        self._X_train = X_arr

        self._model.fit(
            X_train=X_arr, y_train=y_arr,
            max_epochs=100,
            patience=15,
            batch_size=min(256, len(X_arr)),
            virtual_batch_size=min(128, len(X_arr)),
            drop_last=False,
        )

    def predict(self, X):
        X_arr = np.array(X, dtype=np.float32)
        return np.asarray(self._model.predict(X_arr)).ravel()

    def shap_values(self, X):
        """Use TabNet's built-in feature importances reshaped as pseudo-SHAP.

        TabNet attention masks provide per-sample feature importance natively.
        We use explain() which returns per-sample attention masks.
        """
        X_arr = np.array(X, dtype=np.float32)
        # explain() returns (M_explain, masks) where M_explain is aggregate
        M_explain, masks = self._model.explain(X_arr)
        # M_explain shape: (n_samples, n_features) — acts like SHAP values
        # Scale by the prediction residual direction to give sign information
        preds = self._model.predict(X_arr).flatten()
        mean_pred = preds.mean()
        signs = np.sign(preds - mean_pred).reshape(-1, 1)
        return M_explain * signs

    def feature_importances(self, feature_names):
        """Use TabNet's global feature importance from attention masks."""
        imp = self._model.feature_importances_
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


class MLPEngine(BaseEngine):
    """Multi-Layer Perceptron — standard feedforward neural network.

    Uses a small architecture appropriate for pharmacometric datasets
    (typically 200-800 subjects, 5-20 covariates). Includes dropout
    and early stopping to prevent overfitting.

    SHAP values computed via KernelExplainer (model-agnostic, slower
    than TreeExplainer but works with any model).

    Note: On typical popPK datasets (<1000 subjects), tree-based methods
    will likely outperform MLP. This engine is included for completeness
    and for larger datasets (e.g., real-world data, pooled analyses).
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self._model = None
        self._scaler = None
        self._X_train_scaled = None

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        # Scale inputs — critical for MLP convergence
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._X_train_scaled = X_scaled

        self._model = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=0.01,              # L2 regularization
            learning_rate="adaptive",
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20,
            random_state=self.random_state,
        )
        self._model.fit(X_scaled, y)

    def predict(self, X):
        X_scaled = self._scaler.transform(X)
        return np.asarray(self._model.predict(X_scaled)).ravel()

    def shap_values(self, X):
        """Compute SHAP values using KernelExplainer.

        Uses a background sample (100 points or full training set if smaller)
        for computational efficiency.
        """
        X_scaled = self._scaler.transform(X)
        # Use subset of training data as background for speed
        n_bg = min(100, len(self._X_train_scaled))
        bg = shap.sample(self._X_train_scaled, n_bg, random_state=self.random_state or 42)
        explainer = shap.KernelExplainer(self._model.predict, bg)
        return explainer.shap_values(X_scaled, nsamples=100)

    def feature_importances(self, feature_names):
        """Approximate importance from first-layer weights.

        Sum of absolute weights from each input to the first hidden layer,
        normalized. This is a fast proxy — SHAP values are more accurate.
        """
        W = self._model.coefs_[0]  # shape: (n_features, hidden_size)
        imp = np.abs(W).sum(axis=1)
        total = imp.sum()
        if total > 0:
            imp = imp / total
        return dict(zip(feature_names, imp))


# ─────────────────────────────────────────────────────────────
# Engine registry
# ─────────────────────────────────────────────────────────────

ENGINE_REGISTRY: dict[str, type[BaseEngine]] = {
    "xgboost": XGBoostEngine,
    "lightgbm": LightGBMEngine,
    "catboost": CatBoostEngine,
    "random_forest": RandomForestEngine,
    "extra_trees": ExtraTreesEngine,
    "gradient_boosting": GradientBoostingEngine,
    "elastic_net": ElasticNetEngine,
    "lasso": LassoEngine,
    "adaptive_lasso": AdaptiveLassoEngine,
    "aalasso": AALassoEngine,
    "ridge": RidgeEngine,
    "stg": STGEngine,
    "tabnet": TabNetEngine,
    "mlp": MLPEngine,
}

# Grouped by category for reporting
ENGINE_CATEGORIES = {
    "Gradient Boosting": ["xgboost", "lightgbm", "catboost", "gradient_boosting"],
    "Ensemble Trees": ["random_forest", "extra_trees"],
    "Regularized Linear": ["elastic_net", "lasso", "adaptive_lasso", "aalasso", "ridge"],
    "Sparse Neural": ["stg"],
    "Deep Learning": ["tabnet", "mlp"],
}
ENGINE_FAMILY = {
    "xgboost": "boosting",
    "lightgbm": "boosting",
    "catboost": "boosting",
    "gradient_boosting": "boosting",
    "random_forest": "bagging",
    "extra_trees": "bagging",
    "elastic_net": "linear",
    "lasso": "linear",
    "adaptive_lasso": "linear",
    "aalasso": "linear",
    "ridge": "linear",
    "stg": "sparse_neural",
    "tabnet": "deep_learning",
    "mlp": "deep_learning",
}

# Ensemble defaults favor classical tabular methods. Deep learning engines
# remain available, but they are opt-in because typical popPK datasets are
# modest in size and easier to overfit.
DEFAULT_ENSEMBLE_METHODS = [
    "xgboost",
    "lightgbm",
    "catboost",
    "random_forest",
    "elastic_net",
    "lasso",
    "aalasso",
]
DEEP_LEARNING_ENSEMBLE_METHODS = ["stg", "tabnet", "mlp"]
FULL_ENSEMBLE_METHODS = DEFAULT_ENSEMBLE_METHODS + DEEP_LEARNING_ENSEMBLE_METHODS


def get_engine(method: str, random_state=42) -> BaseEngine:
    """Create an ML engine by name.

    Available methods: xgboost, lightgbm, catboost, random_forest,
                       elastic_net, lasso, tabnet, mlp
    """
    if method not in ENGINE_REGISTRY:
        available = ", ".join(ENGINE_REGISTRY.keys())
        raise ValueError(f"Unknown method '{method}'. Available: {available}")
    return ENGINE_REGISTRY[method](random_state=random_state)


def list_engines() -> dict[str, list[str]]:
    """Return available engines grouped by category."""
    return ENGINE_CATEGORIES.copy()


def get_engine_family(method: str) -> str:
    """Return the broad model family for a method name."""
    if method not in ENGINE_FAMILY:
        raise ValueError(f"Unknown method '{method}'")
    return ENGINE_FAMILY[method]


def check_engine_availability() -> dict[str, bool]:
    """Check which engines have their dependencies installed.

    Returns dict mapping engine name to availability (True/False).
    Useful for graceful degradation if optional packages not installed.
    """
    _import_checks = {
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "random_forest": "sklearn",
        "extra_trees": "sklearn",
        "gradient_boosting": "sklearn",
        "elastic_net": "sklearn",
        "lasso": "sklearn",
        "adaptive_lasso": "sklearn",
        "aalasso": "sklearn",
        "ridge": "sklearn",
        "stg": "torch",
        "tabnet": "pytorch_tabnet",
        "mlp": "sklearn",
    }
    availability = {}
    for name, pkg in _import_checks.items():
        availability[name] = find_spec(pkg) is not None
    return availability
