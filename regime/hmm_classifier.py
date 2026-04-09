"""
regime/hmm_classifier.py
========================
3-state Gaussian HMM for volatility regime detection.

States:
    0 = LOW_VOL:    VIX ~12-18, low daily vol, steep contango
    1 = TRANSITION:  VIX ~18-25, moderate vol, flattening curve
    2 = HIGH_VOL:    VIX ~25+, high vol, backwardation

Key design decisions (from literature):
    - 3 states (Goutte et al. 2017) — better than 2 for VIX specifically
    - Full covariance (captures cross-feature dynamics)
    - Walk-forward: retrain quarterly on expanding window
    - Label assignment: post-hoc by variance (resolves label-switching)
    - Debounce: 3-day minimum holding to avoid whipsaw (Nystrup et al. 2020)
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from enum import IntEnum
import logging
import warnings

logger = logging.getLogger(__name__)

# Try importing hmmlearn
try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    logger.warning("hmmlearn not installed. pip install hmmlearn")

from config.settings import REGIME


class VolRegime(IntEnum):
    """Volatility regime labels."""
    LOW_VOL = 0
    TRANSITION = 1
    HIGH_VOL = 2


class RegimeClassifier:
    """
    Walk-forward 3-state Gaussian HMM for volatility regime detection.
    
    Usage:
        clf = RegimeClassifier()
        df = clf.fit_predict(df)  # Walk-forward: train → predict → retrain → predict
        
        # Or for real-time inference:
        clf.fit(training_df)
        regime, probs = clf.predict_realtime(new_observation)
    """
    
    def __init__(self, config: object = REGIME):
        self.config = config
        self.model = None
        self.label_map = None  # Maps HMM state IDs → VolRegime labels
        self._fitted = False
    
    def fit_predict(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Walk-forward regime classification.
        
        1. Use first `training_window_days` for initial fit
        2. Predict forward `refit_frequency_days`
        3. Expand training window, refit, predict next chunk
        4. Repeat until end of data
        
        Args:
            df: DataFrame with HMM feature columns
            feature_cols: Columns to use as HMM observations.
                         Default: ["HMM_VIX_LogReturn_Std", "HMM_TS_Slope_Std"]
        
        Returns:
            df with added columns: Regime, Regime_Prob_LowVol, Regime_Prob_Transition, Regime_Prob_HighVol
        """
        if not HMMLEARN_AVAILABLE:
            raise RuntimeError("hmmlearn required. pip install hmmlearn")
        
        if feature_cols is None:
            feature_cols = ["HMM_VIX_LogReturn_Std", "HMM_TS_Slope_Std"]
        
        # Validate features exist
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")
        
        df = df.copy()
        n = len(df)
        
        # Initialize output columns
        df["Regime"] = np.nan
        df["Regime_Prob_LowVol"] = np.nan
        df["Regime_Prob_Transition"] = np.nan
        df["Regime_Prob_HighVol"] = np.nan
        df["Regime_Raw"] = np.nan  # Before debouncing
        
        from config.settings import BACKTEST
        train_window = BACKTEST.training_window_days
        refit_freq = BACKTEST.refit_frequency_days
        min_train = BACKTEST.min_training_days
        
        if n < min_train:
            logger.warning(f"Not enough data ({n} rows) for minimum training ({min_train}). Returning NaN regimes.")
            return df
        
        # Walk-forward loop
        fit_start = 0
        predict_start = max(min_train, train_window)
        refit_count = 0
        
        while predict_start < n:
            predict_end = min(predict_start + refit_freq, n)
            
            # Extract training data (expanding window from start)
            train_data = df.iloc[fit_start:predict_start][feature_cols].dropna()
            
            if len(train_data) < min_train:
                predict_start = predict_end
                continue
            
            # Fit HMM
            X_train = train_data.values
            self._fit_hmm(X_train)
            refit_count += 1
            
            # Predict on the next chunk (and all prior data for consistency)
            predict_data = df.iloc[:predict_end][feature_cols].dropna()
            X_predict = predict_data.values
            
            if len(X_predict) == 0:
                predict_start = predict_end
                continue
            
            states, posteriors = self._predict(X_predict)
            
            # Map HMM states to VolRegime labels
            mapped_states = self._map_labels(states, X_train)
            
            # Write predictions for the current chunk only
            chunk_indices = df.iloc[predict_start:predict_end].index
            chunk_mask = predict_data.index.isin(chunk_indices)
            
            for i, idx in enumerate(predict_data.index):
                if idx in chunk_indices:
                    row_idx = df.index.get_loc(idx)
                    df.iloc[row_idx, df.columns.get_loc("Regime_Raw")] = mapped_states[i]
                    df.iloc[row_idx, df.columns.get_loc("Regime_Prob_LowVol")] = posteriors[i, self._get_hmm_state(VolRegime.LOW_VOL)]
                    df.iloc[row_idx, df.columns.get_loc("Regime_Prob_Transition")] = posteriors[i, self._get_hmm_state(VolRegime.TRANSITION)]
                    df.iloc[row_idx, df.columns.get_loc("Regime_Prob_HighVol")] = posteriors[i, self._get_hmm_state(VolRegime.HIGH_VOL)]
            
            predict_start = predict_end
        
        # Apply causal confirmation filter (replaces hindsight debounce)
        df["Regime"] = self._confirm_regimes(
            df["Regime_Raw"].values,
            df[["Regime_Prob_LowVol", "Regime_Prob_Transition", "Regime_Prob_HighVol"]].values,
            prob_threshold=self.config.regime_transition_prob_threshold,
            n_days=self.config.min_regime_holding_days,
        )
        
        logger.info(f"Walk-forward complete: {refit_count} refits, {n - min_train} predictions")
        
        # Log regime distribution
        regime_counts = df["Regime"].dropna().value_counts()
        for regime_val, count in regime_counts.items():
            regime_name = VolRegime(int(regime_val)).name if not np.isnan(regime_val) else "NaN"
            logger.info(f"  {regime_name}: {count} days ({count/len(df)*100:.1f}%)")
        
        return df
    
    def _fit_hmm(self, X: np.ndarray):
        """
        Fit 3-state Gaussian HMM with multiple random restarts.
        Select the model with highest log-likelihood.
        """
        best_score = -np.inf
        best_model = None
        
        n_starts = self.config.n_random_starts
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            for i in range(n_starts):
                model = GaussianHMM(
                    n_components=self.config.n_states,
                    covariance_type=self.config.covariance_type,
                    n_iter=self.config.n_iter,
                    tol=self.config.tol,
                    random_state=i * 42,
                )
                
                try:
                    model.fit(X)
                    score = model.score(X)
                    
                    if score > best_score:
                        best_score = score
                        best_model = model
                except Exception as e:
                    continue
        
        if best_model is None:
            raise RuntimeError("HMM fitting failed on all random starts")
        
        self.model = best_model
        self._fitted = True
    
    def _predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward-only filtered state probabilities.

        Returns P(state_t | obs_1:t) — uses NO future observations.
        hmmlearn's predict() (Viterbi) and predict_proba() (forward-backward)
        both use the full sequence; we need the raw forward lattice instead.

        Returns:
            states: (n_samples,) array of MAP state indices per day
            posteriors: (n_samples, n_states) filtered probability matrix
        """
        from hmmlearn import _hmmc

        log_frameprob = self.model._compute_log_likelihood(X)
        _, fwdlattice = _hmmc.forward_log(
            self.model.startprob_, self.model.transmat_, log_frameprob,
        )
        # fwdlattice is log P(obs_1:t, state_t) — normalize each row to get
        # P(state_t | obs_1:t)
        log_posteriors = fwdlattice - np.logaddexp.reduce(
            fwdlattice, axis=1, keepdims=True,
        )
        posteriors = np.exp(log_posteriors)
        states = posteriors.argmax(axis=1)
        return states, posteriors
    
    def _map_labels(self, states: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Map HMM state indices to VolRegime labels.
        
        Strategy: Assign labels by the VARIANCE of VIX log returns in each state.
        - Lowest variance → LOW_VOL
        - Highest variance → HIGH_VOL
        - Middle → TRANSITION
        
        This resolves the label-switching problem inherent in HMMs.
        """
        state_variances = {}
        for s in range(self.config.n_states):
            mask = states[:len(X_train)] == s
            if mask.sum() > 0:
                # Use variance of first feature (VIX log return)
                state_variances[s] = X_train[mask, 0].var()
            else:
                state_variances[s] = 0.0
        
        # Sort states by variance
        sorted_states = sorted(state_variances.keys(), key=lambda s: state_variances[s])
        
        # Build mapping: HMM state → VolRegime
        self.label_map = {
            sorted_states[0]: VolRegime.LOW_VOL,
            sorted_states[1]: VolRegime.TRANSITION,
            sorted_states[2]: VolRegime.HIGH_VOL,
        }
        
        # Apply mapping
        mapped = np.array([self.label_map.get(s, VolRegime.TRANSITION) for s in states])
        return mapped
    
    def _get_hmm_state(self, regime: VolRegime) -> int:
        """Get the HMM state index for a given VolRegime label."""
        if self.label_map is None:
            return int(regime)
        
        reverse_map = {v: k for k, v in self.label_map.items()}
        return reverse_map.get(regime, int(regime))
    
    @staticmethod
    def _confirm_regimes(
        raw_regimes: np.ndarray,
        probs: np.ndarray,
        prob_threshold: float = 0.70,
        n_days: int = 3,
    ) -> np.ndarray:
        """
        Causal regime confirmation — no future data used.

        A regime transition is confirmed only when the filtered probability
        for the new MAP state exceeds *prob_threshold* for *n_days*
        consecutive days. Until confirmed, the previous regime persists.

        This replaces the old debounce which scanned forward to measure
        run length (look-ahead).
        """
        n = len(raw_regimes)
        result = np.full(n, np.nan)

        if n == 0:
            return result

        # Bootstrap: first non-NaN raw regime becomes the active regime
        active_regime = np.nan
        streak = 0

        for t in range(n):
            if np.isnan(raw_regimes[t]):
                result[t] = np.nan
                continue

            candidate = int(raw_regimes[t])
            prob_col = candidate  # probs columns are [LowVol, Transition, HighVol]
            prob_t = probs[t, prob_col] if not np.isnan(probs[t, prob_col]) else 0.0

            if np.isnan(active_regime):
                # No active regime yet — accept first valid state immediately
                if prob_t >= prob_threshold:
                    streak += 1
                    if streak >= n_days:
                        active_regime = candidate
                else:
                    streak = 0
                result[t] = active_regime  # stays NaN until confirmed
                continue

            if candidate == active_regime:
                # Same regime — reset streak, continue
                streak = 0
                result[t] = active_regime
            else:
                # Different regime proposed — count consecutive days
                if prob_t >= prob_threshold:
                    streak += 1
                else:
                    streak = 0

                if streak >= n_days:
                    active_regime = candidate
                    streak = 0

                result[t] = active_regime

        return result
    
    def predict_realtime(self, observation: np.ndarray) -> Tuple[VolRegime, np.ndarray]:
        """
        Single-observation real-time prediction using forward filtering.

        Args:
            observation: (n_features,) array for one day

        Returns:
            regime: VolRegime enum value
            probs: (n_states,) filtered probabilities
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit_predict() first.")

        X = observation.reshape(1, -1)
        states, posteriors = self._predict(X)
        state = states[0]

        regime = self.label_map.get(state, VolRegime.TRANSITION)
        return regime, posteriors[0]
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Return the estimated transition matrix with readable labels."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        
        transmat = self.model.transmat_
        labels = [VolRegime(self.label_map[i]).name for i in range(self.config.n_states)]
        
        return pd.DataFrame(transmat, index=labels, columns=labels)
    
    def get_regime_params(self) -> Dict:
        """Return estimated parameters for each regime."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        
        params = {}
        for hmm_state in range(self.config.n_states):
            regime = self.label_map[hmm_state]
            params[VolRegime(regime).name] = {
                "mean": self.model.means_[hmm_state].tolist(),
                "covariance": self.model.covars_[hmm_state].tolist(),
                "stationary_prob": self.model.get_stationary_distribution()[hmm_state],
                "self_transition_prob": self.model.transmat_[hmm_state, hmm_state],
                "expected_duration_days": 1 / (1 - self.model.transmat_[hmm_state, hmm_state]),
            }
        
        return params
