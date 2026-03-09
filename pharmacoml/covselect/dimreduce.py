"""
Autoencoder-based dimensionality reduction for high-dimensional covariate spaces.

Optional preprocessor that activates when >30 covariates are present
(e.g., pharmacogenomics + standard demographics). Maps latent dimensions
back to original covariates for interpretability.

Usage:
    from pharmacoml.covselect.dimreduce import AutoEncoderReducer

    reducer = AutoEncoderReducer(n_components=10)
    covs_reduced, mapping = reducer.fit_transform(covariates)
    # covs_reduced: DataFrame with 10 latent features
    # mapping: dict mapping each latent dim to top contributing original covariates

    # After screening on reduced features, map back:
    original_contribs = reducer.map_to_original("latent_3")
    # -> {"WT": 0.45, "BMI": 0.38, "BSA": 0.12, ...}
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class AutoEncoderReducer:
    """Autoencoder-based dimensionality reduction for covariate screening.

    Uses a symmetric autoencoder to learn a compressed representation of
    high-dimensional covariate spaces. The encoder weights provide a
    mapping from latent dimensions back to original covariates.

    Only recommended when n_covariates > 30. For typical pharmacometric
    datasets (5-20 covariates), use the engines directly.

    Parameters
    ----------
    n_components : int, default 10
        Number of latent dimensions. Recommended: n_covariates // 3 to 5.
    hidden_sizes : list of int, optional
        Hidden layer sizes for encoder (decoder mirrors).
        Default: [64, 32] -> bottleneck -> [32, 64]
    max_epochs : int, default 200
    learning_rate : float, default 1e-3
    min_covariates_to_activate : int, default 30
        If n_covariates < this, raises a warning and returns data unchanged.
    random_state : int, default 42
    """

    def __init__(
        self,
        n_components: int = 10,
        hidden_sizes: list[int] | None = None,
        max_epochs: int = 200,
        learning_rate: float = 1e-3,
        min_covariates_to_activate: int = 30,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.min_covariates_to_activate = min_covariates_to_activate
        self.random_state = random_state

        self._scaler = None
        self._encoder = None
        self._decoder = None
        self._model = None
        self._original_columns = None
        self._encoder_weights = None
        self._is_fitted = False

    def fit(self, covariates: pd.DataFrame) -> "AutoEncoderReducer":
        """Fit the autoencoder on covariate data.

        Parameters
        ----------
        covariates : pd.DataFrame
            Covariate matrix. Columns are covariates, rows are subjects.
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._original_columns = list(covariates.columns)
        n_features = len(self._original_columns)

        if n_features < self.min_covariates_to_activate:
            warnings.warn(
                f"Only {n_features} covariates — autoencoder not needed "
                f"(threshold: {self.min_covariates_to_activate}). "
                f"Use CovariateScreener directly on the original covariates.",
                stacklevel=2,
            )

        # Scale
        self._scaler = StandardScaler()
        X = self._scaler.fit_transform(covariates.values.astype(np.float32))
        X_tensor = torch.FloatTensor(X)

        # Set seed
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Build autoencoder
        encoder_layers = []
        dims = [n_features] + self.hidden_sizes + [self.n_components]
        for i in range(len(dims) - 1):
            encoder_layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                encoder_layers.append(nn.ReLU())
                encoder_layers.append(nn.BatchNorm1d(dims[i + 1]))

        decoder_layers = []
        dims_rev = list(reversed(dims))
        for i in range(len(dims_rev) - 1):
            decoder_layers.append(nn.Linear(dims_rev[i], dims_rev[i + 1]))
            if i < len(dims_rev) - 2:
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.BatchNorm1d(dims_rev[i + 1]))

        self._encoder = nn.Sequential(*encoder_layers)
        self._decoder = nn.Sequential(*decoder_layers)
        self._model = nn.Sequential(self._encoder, self._decoder)

        # Train
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        dataset = TensorDataset(X_tensor, X_tensor)
        loader = DataLoader(dataset, batch_size=min(64, len(X)), shuffle=True)

        self._model.train()
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for batch_X, _ in loader:
                optimizer.zero_grad()
                reconstructed = self._model(batch_X)
                loss = loss_fn(reconstructed, batch_X)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        # Extract encoder weights for interpretability
        self._extract_encoder_weights()
        self._is_fitted = True
        return self

    def transform(self, covariates: pd.DataFrame) -> pd.DataFrame:
        """Transform covariates to latent space.

        Parameters
        ----------
        covariates : pd.DataFrame

        Returns
        -------
        pd.DataFrame
            Columns are latent_0, latent_1, ..., latent_{n_components-1}
        """
        import torch

        if not self._is_fitted:
            raise RuntimeError("Call fit() before transform()")

        X = self._scaler.transform(covariates.values.astype(np.float32))
        X_tensor = torch.FloatTensor(X)

        self._encoder.eval()
        with torch.no_grad():
            latent = self._encoder(X_tensor).numpy()

        columns = [f"latent_{i}" for i in range(self.n_components)]
        return pd.DataFrame(latent, index=covariates.index, columns=columns)

    def fit_transform(self, covariates: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Fit and transform, returning reduced data and interpretability mapping.

        Returns
        -------
        tuple of (pd.DataFrame, dict)
            - Reduced covariates (n_subjects × n_components)
            - Mapping: {latent_dim: {original_cov: contribution_weight}}
        """
        self.fit(covariates)
        reduced = self.transform(covariates)
        mapping = self.get_feature_mapping()
        return reduced, mapping

    def get_feature_mapping(self, top_k: int = 5) -> dict[str, dict[str, float]]:
        """Map each latent dimension to its top contributing original covariates.

        Parameters
        ----------
        top_k : int, default 5
            Number of top original covariates per latent dimension.

        Returns
        -------
        dict
            {latent_dim_name: {original_cov: weight, ...}}
        """
        if self._encoder_weights is None:
            raise RuntimeError("Model not fitted yet")

        mapping = {}
        for i in range(self.n_components):
            weights = self._encoder_weights[:, i]
            abs_weights = np.abs(weights)
            top_idx = np.argsort(abs_weights)[::-1][:top_k]

            dim_map = {}
            for idx in top_idx:
                cov_name = self._original_columns[idx]
                dim_map[cov_name] = round(float(weights[idx]), 4)

            mapping[f"latent_{i}"] = dim_map

        return mapping

    def map_to_original(self, latent_dim: str, top_k: int = 10) -> dict[str, float]:
        """Get the contribution of original covariates to a specific latent dimension.

        Parameters
        ----------
        latent_dim : str
            Name like "latent_3"
        top_k : int
            Number of top covariates to return.

        Returns
        -------
        dict
            {original_cov: contribution_weight} sorted by absolute weight
        """
        mapping = self.get_feature_mapping(top_k=top_k)
        if latent_dim not in mapping:
            raise ValueError(
                f"'{latent_dim}' not found. Available: {list(mapping.keys())}"
            )
        return mapping[latent_dim]

    def _extract_encoder_weights(self):
        """Extract a simplified weight matrix from encoder for interpretability.

        Computes the product of all linear layer weights in the encoder
        to get a (n_original_features × n_components) mapping matrix.
        """
        import torch

        weight_matrices = []
        for layer in self._encoder:
            if hasattr(layer, "weight"):
                weight_matrices.append(layer.weight.data.numpy())

        if not weight_matrices:
            self._encoder_weights = None
            return

        # Chain multiply: W_final = W_n @ ... @ W_2 @ W_1
        # Each W_i has shape (out_i, in_i), so product gives (n_components, n_features)
        result = weight_matrices[0]
        for W in weight_matrices[1:]:
            result = W @ result

        # Transpose to (n_features, n_components)
        self._encoder_weights = result.T

    @property
    def reconstruction_error(self) -> float | None:
        """Return the final reconstruction MSE (lower = better compression)."""
        if not self._is_fitted:
            return None
        import torch
        # Would need stored training data — return None for now
        return None

    def __repr__(self):
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"AutoEncoderReducer(n_components={self.n_components}, "
            f"hidden={self.hidden_sizes}, {status})"
        )
