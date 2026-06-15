"""
Portfolio Optimization Module - Black-Litterman Model Implementation

This module implements the Black-Litterman portfolio optimization model,
which combines market equilibrium returns with investor views to generate
optimal portfolio weights using the Sharpe ratio as the optimization objective.

Key Components:
- CovarianceEstimator: Estimates asset covariance matrix from historical prices
- MarketPrior: Calculates equilibrium returns (prior) from market weights
- BlackLittermanModel: Core BL model for posterior return calculation
- PortfolioOptimizer: Optimizes portfolio weights to maximize Sharpe ratio
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    posterior_returns: Dict[str, float]
    views_count: int


class CovarianceEstimator:
    """Estimate covariance matrix from historical price data."""

    def __init__(
        self,
        lookback_days: int = 60,
        min_data_points: int = 20,
        regularization_method: str = "diagonal",
    ):
        """
        Args:
            lookback_days: Number of days to look back for covariance calculation
            min_data_points: Minimum data points required for covariance calculation
            regularization_method: Method for covariance regularization
                - "diagonal": Add small diagonal term
                - "ledoit_wolf": Ledoit-Wolf shrinkage (requires sklearn)
        """
        self.lookback_days = lookback_days
        self.min_data_points = min_data_points
        self.regularization_method = regularization_method

    def calculate_covariance_matrix(
        self,
        historical_prices: Dict[str, List[Dict]],
        fund_pool: List[str],
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate covariance matrix from historical prices.

        Args:
            historical_prices: Dict mapping fund_id to list of price data
            fund_pool: List of fund IDs to include

        Returns:
            Tuple of (covariance_matrix, valid_fund_list)
        """
        # Build returns data for each fund
        returns_dict = {}
        valid_funds = []

        for fund_id in fund_pool:
            prices = historical_prices.get(fund_id, [])
            if len(prices) < self.min_data_points:
                continue

            # Extract closes and calculate daily returns
            closes = []
            for p in prices[-self.lookback_days:]:
                close = p.get("close")
                if close is not None and not math.isnan(close):
                    closes.append(close)

            if len(closes) < 2:
                continue

            # Calculate daily returns
            returns = []
            for i in range(1, len(closes)):
                ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                if not math.isnan(ret) and not math.isinf(ret):
                    returns.append(ret)

            if len(returns) >= self.min_data_points:
                returns_dict[fund_id] = returns[-self.min_data_points:]
                valid_funds.append(fund_id)

        if len(valid_funds) < 2:
            logger.warning("Insufficient funds for covariance calculation")
            # Return identity matrix as fallback
            n = len(fund_pool)
            return np.eye(n) * 0.0001, fund_pool

        # Build returns array
        min_len = min(len(r) for r in returns_dict.values())
        returns_array = np.array([r[-min_len:] for r in returns_dict.values()])

        # Calculate covariance matrix
        cov_matrix = np.cov(returns_array)

        # Ensure positive definiteness
        cov_matrix = self.ensure_positive_definite(cov_matrix)

        return cov_matrix, valid_funds

    def ensure_positive_definite(self, cov_matrix: np.ndarray) -> np.ndarray:
        """Ensure covariance matrix is positive definite."""
        eigenvalues = np.linalg.eigvals(cov_matrix)

        if np.all(eigenvalues > 1e-8):
            return cov_matrix

        logger.warning("Covariance matrix not positive definite, applying regularization")

        if self.regularization_method == "diagonal":
            n = cov_matrix.shape[0]
            return cov_matrix + np.eye(n) * 1e-6
        elif self.regularization_method == "ledoit_wolf":
            try:
                from sklearn.covariance import LedoitWolf
                lw = LedoitWolf()
                return lw.fit(cov_matrix.T).covariance_
            except ImportError:
                logger.warning("sklearn not available, falling back to diagonal regularization")
                n = cov_matrix.shape[0]
                return cov_matrix + np.eye(n) * 1e-6
        else:
            n = cov_matrix.shape[0]
            return cov_matrix + np.eye(n) * 1e-6


class MarketPrior:
    """Calculate market equilibrium returns (prior) for Black-Litterman model."""

    def __init__(
        self,
        risk_aversion: float = 3.0,
        market_weight_method: str = "equal_weight",
    ):
        """
        Args:
            risk_aversion: Risk aversion coefficient (δ)
            market_weight_method: Method for determining market weights
                - "equal_weight": Equal weights for all assets
                - "inverse_vol": Inverse volatility weights
        """
        self.risk_aversion = risk_aversion
        self.market_weight_method = market_weight_method

    def calculate_market_weights(
        self,
        valid_funds: List[str],
        cov_matrix: np.ndarray,
        historical_prices: Dict[str, List[Dict]],
    ) -> np.ndarray:
        """
        Calculate market capitalization weights.

        Args:
            valid_funds: List of valid fund IDs
            cov_matrix: Covariance matrix
            historical_prices: Historical price data

        Returns:
            Array of market weights (sums to 1)
        """
        n = len(valid_funds)

        if self.market_weight_method == "equal_weight":
            return np.ones(n) / n
        elif self.market_weight_method == "inverse_vol":
            # Inverse volatility weights
            vols = np.sqrt(np.diag(cov_matrix))
            vols = np.where(vols < 1e-10, 1e-10, vols)  # Avoid division by zero
            inv_vols = 1.0 / vols
            return inv_vols / inv_vols.sum()
        else:
            return np.ones(n) / n

    def calculate_equilibrium_returns(
        self,
        cov_matrix: np.ndarray,
        market_weights: Optional[np.ndarray] = None,
        valid_funds: Optional[List[str]] = None,
        historical_prices: Optional[Dict[str, List[Dict]]] = None,
    ) -> np.ndarray:
        """
        Calculate equilibrium (prior) returns using reverse optimization.

        Formula: π = δ * Σ * w_market

        Args:
            cov_matrix: Covariance matrix
            market_weights: Market capitalization weights (optional)
            valid_funds: List of valid fund IDs (optional)
            historical_prices: Historical price data (optional, for inverse vol weights)

        Returns:
            Array of equilibrium returns
        """
        n = cov_matrix.shape[0]

        if market_weights is None:
            if valid_funds is not None and historical_prices is not None:
                market_weights = self.calculate_market_weights(
                    valid_funds, cov_matrix, historical_prices
                )
            else:
                market_weights = np.ones(n) / n

        # π = δ * Σ * w
        prior_returns = self.risk_aversion * (cov_matrix @ market_weights)

        return prior_returns


class BlackLittermanModel:
    """Core Black-Litterman model implementation."""

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 3.0,
        lookback_days: int = 60,
        omega_method: str = "confidence",
        market_weight_method: str = "equal_weight",
    ):
        """
        Args:
            tau: Scaling parameter controlling view influence (smaller = less influence)
            risk_aversion: Risk aversion coefficient
            lookback_days: Days for covariance calculation
            omega_method: Method for calculating uncertainty matrix Ω
                - "confidence": Based on view confidence levels
                - "proportional": Proportional to prior covariance
            market_weight_method: Prior equilibrium weight method
                - "equal_weight": Equal weights (default)
                - "inverse_vol": Inverse volatility (risk parity prior)
        """
        self.tau = tau
        self.risk_aversion = risk_aversion
        self.lookback_days = lookback_days
        self.omega_method = omega_method

        self.cov_estimator = CovarianceEstimator(lookback_days=lookback_days)
        self.market_prior = MarketPrior(
            risk_aversion=risk_aversion,
            market_weight_method=market_weight_method,
        )

    def calculate_omega(
        self,
        P: np.ndarray,
        cov_matrix: np.ndarray,
        confidences: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate uncertainty matrix Ω based on confidences.

        Formula: Ω_ii = (1/confidence_i^2) * P_i @ (τΣ) @ P_i^T

        Args:
            P: View matrix (k x n)
            cov_matrix: Covariance matrix (n x n)
            confidences: Array of confidence values (k,)

        Returns:
            Uncertainty matrix Ω (k x k)
        """
        k, n = P.shape
        tau_sigma = self.tau * cov_matrix

        omega = np.zeros((k, k))

        for i in range(k):
            # Confidence factor: lower confidence = higher uncertainty
            confidence_factor = 1.0 / max(confidences[i], 0.1)
            omega[i, i] = confidence_factor * (P[i:i+1] @ tau_sigma @ P[i:i+1].T).item()

        return omega

    def calculate_posterior_returns(
        self,
        prior_returns: np.ndarray,
        cov_matrix: np.ndarray,
        P: np.ndarray,
        Q: np.ndarray,
        confidences: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate posterior (Black-Litterman) returns.

        Formula:
            M = P @ (τΣ) @ P^T + Ω
            μ_BL = π + (τΣ) @ P^T @ M^(-1) @ (Q - P @ π)

        Args:
            prior_returns: Prior equilibrium returns (n x 1)
            cov_matrix: Covariance matrix (n x n)
            P: View matrix (k x n)
            Q: View returns (k x 1)
            confidences: Confidence levels for views (k x 1), optional
            omega: Uncertainty matrix (k x k), optional

        Returns:
            Tuple of (posterior_returns, posterior_covariance)
        """
        n = len(prior_returns)
        tau_sigma = self.tau * cov_matrix

        # Calculate Ω if not provided
        if omega is None:
            if confidences is not None:
                omega = self.calculate_omega(P, cov_matrix, confidences)
            else:
                # Default: proportional to P @ tau_sigma @ P^T
                omega = P @ tau_sigma @ P.T

        # Calculate M = P @ (τΣ) @ P^T + Ω
        M = P @ tau_sigma @ P.T + omega

        # Calculate posterior returns
        # μ_BL = π + (τΣ) @ P^T @ M^(-1) @ (Q - P @ π)
        try:
            M_inv = np.linalg.inv(M)
            view_adjustment = tau_sigma @ P.T @ M_inv @ (Q - P @ prior_returns)
            posterior_returns = prior_returns + view_adjustment
        except np.linalg.LinAlgError:
            logger.warning("Matrix inversion failed, using prior returns")
            posterior_returns = prior_returns

        # Calculate posterior covariance
        # Σ_BL = (1 + τ)Σ - (τΣ) @ P^T @ M^(-1) @ P @ (τΣ)
        try:
            posterior_cov = (1 + self.tau) * cov_matrix - tau_sigma @ P.T @ M_inv @ P @ tau_sigma
        except np.linalg.LinAlgError:
            logger.warning("Posterior covariance calculation failed, using prior")
            posterior_cov = cov_matrix

        return posterior_returns, posterior_cov

    def calculate_optimal_weights(
        self,
        posterior_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float = 0.03,
        method: str = "max_sharpe",
        min_weight: float = 0.0,
        max_weight: float = 0.4,
    ) -> np.ndarray:
        """
        Calculate optimal portfolio weights.

        Args:
            posterior_returns: Posterior expected returns (n x 1)
            cov_matrix: Covariance matrix (n x n)
            risk_free_rate: Risk-free rate (annualized)
            method: Optimization method
                - "max_sharpe": Maximize Sharpe ratio
                - "min_variance": Minimize portfolio variance
            min_weight: Minimum weight for any asset
            max_weight: Maximum weight for any asset

        Returns:
            Optimal weights (n x 1)
        """
        n = len(posterior_returns)

        if method == "max_sharpe":
            return self._maximize_sharpe_ratio(
                posterior_returns, cov_matrix, risk_free_rate, min_weight, max_weight
            )
        elif method == "min_variance":
            return self._minimize_variance(
                cov_matrix, min_weight, max_weight
            )
        else:
            # Default to equal weights
            return np.ones(n) / n

    def _maximize_sharpe_ratio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_free_rate: float,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """
        Maximize Sharpe ratio using numerical optimization.

        Max: (w^T μ - rf) / sqrt(w^T Σ w)
        Subject to: sum(w) = 1, min_weight <= w_i <= max_weight
        """
        from scipy.optimize import minimize

        n = len(expected_returns)

        # Daily risk-free rate (assuming 252 trading days)
        daily_rf = risk_free_rate / 252

        def objective(weights):
            """Negative Sharpe ratio (to minimize)"""
            portfolio_return = weights @ expected_returns
            portfolio_var = weights @ cov_matrix @ weights
            portfolio_std = np.sqrt(portfolio_var)
            if portfolio_std < 1e-10:
                return -1e10  # Large negative for near-zero variance
            sharpe = (portfolio_return - daily_rf) / portfolio_std
            return -sharpe

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(n)]

        # Initial guess: equal weights
        x0 = np.ones(n) / n

        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'ftol': 1e-9, 'maxiter': 1000}
        )

        if result.success:
            return result.x
        else:
            logger.warning(f"Optimization failed: {result.message}, using equal weights")
            return np.ones(n) / n

    def _minimize_variance(
        self,
        cov_matrix: np.ndarray,
        min_weight: float,
        max_weight: float,
    ) -> np.ndarray:
        """
        Minimize portfolio variance.

        Min: w^T Σ w
        Subject to: sum(w) = 1, min_weight <= w_i <= max_weight
        """
        from scipy.optimize import minimize

        n = cov_matrix.shape[0]

        def objective(weights):
            return weights @ cov_matrix @ weights

        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        ]

        bounds = [(min_weight, max_weight) for _ in range(n)]
        x0 = np.ones(n) / n

        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
        )

        if result.success:
            return result.x
        else:
            return np.ones(n) / n


class PortfolioOptimizer:
    """High-level portfolio optimizer combining BL model and optimization."""

    def __init__(
        self,
        tau: float = 0.05,
        risk_aversion: float = 3.0,
        lookback_days: int = 60,
        risk_free_rate: float = 0.03,
        optimization_method: str = "max_sharpe",
        min_weight: float = 0.0,
        max_weight: float = 0.4,
        market_weight_method: str = "equal_weight",
    ):
        """
        Args:
            tau: BL scaling parameter
            risk_aversion: Risk aversion coefficient
            lookback_days: Days for covariance calculation
            risk_free_rate: Annual risk-free rate
            optimization_method: "max_sharpe" or "min_variance"
            min_weight: Minimum weight constraint
            max_weight: Maximum weight constraint
            market_weight_method: Prior equilibrium weight method ("equal_weight" or "inverse_vol")
        """
        self.bl_model = BlackLittermanModel(
            tau=tau,
            risk_aversion=risk_aversion,
            lookback_days=lookback_days,
            market_weight_method=market_weight_method,
        )
        self.risk_free_rate = risk_free_rate
        self.optimization_method = optimization_method
        self.min_weight = min_weight
        self.max_weight = max_weight

    def optimize_portfolio(
        self,
        historical_prices: Dict[str, List[Dict]],
        views: List[Dict],
        fund_pool: List[str],
    ) -> OptimizationResult:
        """
        Optimize portfolio using Black-Litterman model.

        Args:
            historical_prices: Historical price data
            views: List of view dicts with keys:
                - fund_id: str
                - view_type: str ("absolute" or "relative")
                - expected_return: float
                - confidence: float (0-1)
                - benchmark_fund_id: str (for relative views)
            fund_pool: List of fund IDs

        Returns:
            OptimizationResult with weights and metrics
        """
        # Step 1: Calculate covariance matrix
        cov_matrix, valid_funds = self.bl_model.cov_estimator.calculate_covariance_matrix(
            historical_prices, fund_pool
        )
        n = len(valid_funds)

        # Step 2: Calculate prior returns
        prior_returns = self.bl_model.market_prior.calculate_equilibrium_returns(
            cov_matrix, valid_funds=valid_funds, historical_prices=historical_prices
        )

        # Step 3: Convert views to matrices
        P, Q, confidences = self._views_to_matrices(views, valid_funds)

        if P is None or len(Q) == 0:
            # No valid views, use prior returns with equal weights
            logger.info("No valid views, using prior optimization")
            weights = self.bl_model.calculate_optimal_weights(
                prior_returns,
                cov_matrix,
                self.risk_free_rate,
                method="min_variance",
                min_weight=self.min_weight,
                max_weight=self.max_weight,
            )
            posterior_returns = prior_returns
        else:
            # Step 4: Calculate posterior returns
            posterior_returns, _ = self.bl_model.calculate_posterior_returns(
                prior_returns,
                cov_matrix,
                P,
                Q,
                confidences,
            )

            # Step 5: Calculate optimal weights
            weights = self.bl_model.calculate_optimal_weights(
                posterior_returns,
                cov_matrix,
                self.risk_free_rate,
                method=self.optimization_method,
                min_weight=self.min_weight,
                max_weight=self.max_weight,
            )

        # Calculate metrics
        expected_return = float(weights @ posterior_returns)
        expected_var = weights @ cov_matrix @ weights
        expected_risk = float(np.sqrt(expected_var))
        daily_rf = self.risk_free_rate / 252
        sharpe_ratio = (expected_return - daily_rf) / expected_risk if expected_risk > 1e-10 else 0

        # Build result
        weights_dict = {valid_funds[i]: float(weights[i]) for i in range(n)}
        posterior_dict = {valid_funds[i]: float(posterior_returns[i]) for i in range(n)}

        return OptimizationResult(
            weights=weights_dict,
            expected_return=expected_return,
            expected_risk=expected_risk,
            sharpe_ratio=sharpe_ratio,
            posterior_returns=posterior_dict,
            views_count=len(Q),
        )

    def _views_to_matrices(
        self,
        views: List[Dict],
        valid_funds: List[str],
    ) -> Tuple[Optional[np.ndarray], np.ndarray, np.ndarray]:
        """
        Convert view list to BL matrices.

        Args:
            views: List of view dicts
            valid_funds: List of valid fund IDs

        Returns:
            Tuple of (P, Q, confidences) or (None, [], []) if no valid views
        """
        fund_to_idx = {fund: i for i, fund in enumerate(valid_funds)}
        valid_views = []

        for view in views:
            fund_id = view.get("fund_id")
            view_type = view.get("view_type", "absolute")
            expected_return = view.get("expected_return")
            confidence = view.get("confidence", 0.5)

            if fund_id not in fund_to_idx:
                continue
            if expected_return is None:
                continue
            if not (0 <= confidence <= 1):
                continue

            valid_views.append({
                "fund_id": fund_id,
                "view_type": view_type,
                "expected_return": expected_return,
                "confidence": max(confidence, 0.1),
                "benchmark_fund_id": view.get("benchmark_fund_id"),
            })

        if not valid_views:
            return None, np.array([]), np.array([])

        n = len(valid_funds)
        k = len(valid_views)

        P = np.zeros((k, n))
        Q = np.zeros(k)
        confidences = np.zeros(k)

        for i, view in enumerate(valid_views):
            fund_idx = fund_to_idx[view["fund_id"]]

            if view["view_type"] == "relative":
                benchmark = view.get("benchmark_fund_id")
                if benchmark and benchmark in fund_to_idx:
                    bench_idx = fund_to_idx[benchmark]
                    P[i, fund_idx] = 1
                    P[i, bench_idx] = -1
                else:
                    P[i, fund_idx] = 1
            else:  # absolute view
                P[i, fund_idx] = 1

            Q[i] = view["expected_return"]
            confidences[i] = view["confidence"]

        return P, Q, confidences
