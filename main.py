"""
main.py
=======
Orchestrator: runs the full pipeline from data fetch to backtest.

Usage:
    # Full pipeline (requires Bloomberg connection):
    python main.py --mode full
    
    # Backtest only (uses cached data):
    python main.py --mode backtest
    
    # Feature analysis only (no backtest):
    python main.py --mode features
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Project imports
from config.settings import BACKTEST, REGIME, SIGNAL, STRIKE, FEATURE
from data.bloomberg_fetcher import BloombergDataPipeline, OfflineDataLoader
from data.cot_fetcher import COTFetcher
from features.indicators import compute_all_features
from regime.hmm_classifier import RegimeClassifier
from signals.composite_score import CompositeSignal
from backtest.engine import BacktestEngine

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-25s | %(levelname)-5s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


def run_full_pipeline():
    """Full pipeline: Bloomberg fetch → features → regime → signals → backtest."""
    
    logger.info("=" * 70)
    logger.info("VIX SYSTEMATIC BULL CALL SPREAD STRATEGY")
    logger.info("Full Pipeline: Fetch → Features → Regime → Signals → Backtest")
    logger.info("=" * 70)
    
    # ── 1. DATA ──────────────────────────────────────────
    logger.info("\n[1/5] FETCHING DATA")
    
    # Bloomberg daily data
    bbg = BloombergDataPipeline(use_cache=True)
    df = bbg.fetch_all(
        start_date=BACKTEST.start_date.strftime("%Y%m%d"),
        end_date=BACKTEST.end_date.strftime("%Y%m%d"),
    )
    bbg.close()
    
    # CFTC COT data
    cot = COTFetcher()
    cot_df = cot.fetch_all(
        start_year=BACKTEST.start_date.year,
        end_year=BACKTEST.end_date.year,
    )
    
    logger.info(f"  Daily data: {len(df)} rows ({df.index.min().date()} to {df.index.max().date()})")
    logger.info(f"  COT data:   {len(cot_df)} weeks")
    
    # ── 2. FEATURES ──────────────────────────────────────
    logger.info("\n[2/5] COMPUTING FEATURES")
    df = compute_all_features(df, cot_df=cot_df)
    
    # Save feature-enriched dataset
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    df.to_parquet(output_dir / "features_dataset.parquet")
    logger.info(f"  Features saved: {len(df.columns)} columns")
    
    # ── 3. REGIME ────────────────────────────────────────
    logger.info("\n[3/5] CLASSIFYING REGIMES (3-state HMM)")
    clf = RegimeClassifier()
    df = clf.fit_predict(df)
    
    # Log regime params
    try:
        params = clf.get_regime_params()
        for regime_name, p in params.items():
            logger.info(f"  {regime_name}: duration={p['expected_duration_days']:.0f}d, "
                       f"self_prob={p['self_transition_prob']:.3f}")
    except Exception:
        pass
    
    # ── 4. SIGNALS ───────────────────────────────────────
    logger.info("\n[4/5] GENERATING SIGNALS")
    signal = CompositeSignal()
    df = signal.compute(df)
    
    # Save signal dataset
    df.to_parquet(output_dir / "signals_dataset.parquet")
    
    # ── 5. BACKTEST ──────────────────────────────────────
    logger.info("\n[5/5] RUNNING BACKTEST")
    engine = BacktestEngine()
    result = engine.run(df)
    
    # Print summary
    engine.print_summary(result)
    
    # Save trades
    trades_df = pd.DataFrame(result.trades)
    if not trades_df.empty:
        trades_df.to_csv(output_dir / "backtest_trades.csv", index=False)
        logger.info(f"\nTrades saved to {output_dir / 'backtest_trades.csv'}")
    
    # Save daily P&L
    result.daily_pnl.to_csv(output_dir / "daily_pnl.csv")
    result.cumulative_pnl.to_csv(output_dir / "cumulative_pnl.csv")
    
    return df, result


def run_backtest_only():
    """Backtest using cached data (no Bloomberg required)."""
    
    logger.info("=" * 70)
    logger.info("BACKTEST ONLY (using cached data)")
    logger.info("=" * 70)
    
    # Load cached data
    output_dir = Path("outputs")
    signals_path = output_dir / "signals_dataset.parquet"
    
    if signals_path.exists():
        logger.info("Loading pre-computed signals...")
        df = pd.read_parquet(signals_path)
    else:
        # Try to load raw cached data and recompute
        features_path = output_dir / "features_dataset.parquet"
        cache_path = Path("outputs/cache/vix_strategy_data.parquet")
        
        if features_path.exists():
            logger.info("Loading features, recomputing regime + signals...")
            df = pd.read_parquet(features_path)
        elif cache_path.exists():
            logger.info("Loading raw cache, recomputing everything...")
            df = pd.read_parquet(cache_path)
            df = compute_all_features(df)
        else:
            raise FileNotFoundError(
                "No cached data found. Run 'python main.py --mode full' first."
            )
        
        # Regime
        clf = RegimeClassifier()
        df = clf.fit_predict(df)
        
        # Signals
        signal = CompositeSignal()
        df = signal.compute(df)
    
    # Run backtest
    engine = BacktestEngine()
    result = engine.run(df)
    engine.print_summary(result)

    # Save trades
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    trades_df = pd.DataFrame(result.trades)
    if not trades_df.empty:
        trades_df.to_csv(output_dir / "backtest_trades.csv", index=False)
    result.daily_pnl.to_csv(output_dir / "daily_pnl.csv")
    result.cumulative_pnl.to_csv(output_dir / "cumulative_pnl.csv")
    
    return df, result


def run_features_only():
    """Compute and analyze features without backtesting."""
    
    logger.info("=" * 70)
    logger.info("FEATURE ANALYSIS ONLY")
    logger.info("=" * 70)
    
    # Load data
    cache_path = Path("outputs/cache/vix_strategy_data.parquet")
    if not cache_path.exists():
        raise FileNotFoundError("No cached data. Run 'python main.py --mode full' first.")
    
    df = pd.read_parquet(cache_path)
    
    # Try loading COT
    cot_path = Path("outputs/cache/cot_vix_data.parquet")
    cot_df = pd.read_parquet(cot_path) if cot_path.exists() else None
    
    # Compute features
    df = compute_all_features(df, cot_df=cot_df)
    
    # Print feature statistics
    logger.info("\nFEATURE STATISTICS:")
    feature_cols = [c for c in df.columns if any(
        c.startswith(p) for p in ["TS_", "VIX_", "VRP_", "VVIX_", "COT_", "SPX_", "NQ_"]
    )]
    
    stats = df[feature_cols].describe().T
    logger.info(f"\n{stats[['mean', 'std', 'min', 'max']].to_string()}")
    
    # Save
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    df.to_parquet(output_dir / "features_dataset.parquet")
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VIX Systematic Strategy")
    parser.add_argument(
        "--mode",
        choices=["full", "backtest", "features"],
        default="full",
        help="Pipeline mode: full (fetch+backtest), backtest (cached), features (analysis only)",
    )
    args = parser.parse_args()
    
    try:
        if args.mode == "full":
            df, result = run_full_pipeline()
        elif args.mode == "backtest":
            df, result = run_backtest_only()
        elif args.mode == "features":
            df = run_features_only()
        
        logger.info("\nDone.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
