#!/usr/bin/env python3
"""
ic_analysis.py

IC检验第二步：读取打分JSON/JSONL，计算信息系数（IC）分析。

Forward return约定:
  signal on day T  →  return = (close_{T+n} - close_T) / close_T
  即T日收盘→T+n日收盘（次日开始持有）

IC指标:
  IC (Information Coefficient) = Spearman相关系数(signal_t, return_{t→t+n})
  横截面IC: 每天对16个ETF截面计算，再时间轴平均
  ICIR = Mean(IC) / Std(IC)

用法示例:
  cd NLPCC_tasks
  python agent_platform/ic_analysis.py
  python agent_platform/ic_analysis.py --scores-dir ic_scores --horizons 1 3 5 20 --output-dir ic_results
  python agent_platform/ic_analysis.py --mode per_news   # 只分析单条新闻IC
"""
import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from config import DATA_DIRS
from server_platform.app.core.data_loader import init_data_loader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

INDUSTRY_FUND_POOL = [
    "512880.SH", "512800.SH", "512070.SH", "159995.SZ",
    "159819.SZ", "515880.SH", "159852.SZ", "512010.SH",
    "512170.SH", "159992.SZ", "515170.SH", "512690.SH",
    "512400.SH", "515220.SH", "159870.SZ", "512200.SH",
]


# ── Price data ─────────────────────────────────────────────────────────────────

def load_close_prices(fund_pool: List[str]) -> pd.DataFrame:
    """
    Load close prices for all ETFs, return DataFrame indexed by date_int.
    Columns = fund codes.
    """
    price_dir = str(DATA_DIRS["PRICE_DATA"])
    frames = {}
    for etf in fund_pool:
        csv_path = os.path.join(price_dir, f"{etf}.csv")
        if not os.path.exists(csv_path):
            continue
        df = pd.read_csv(csv_path, usecols=["date", "close"])
        df = df.dropna(subset=["close"])
        df["date"] = df["date"].astype(int)
        df = df.set_index("date")["close"]
        frames[etf] = df

    if not frames:
        raise FileNotFoundError(f"No price CSVs found in {price_dir}")

    return pd.DataFrame(frames)


def compute_forward_returns(
    close_df: pd.DataFrame,
    trading_dates: List[int],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Compute forward returns for each (date, etf, horizon).

    Returns DataFrame with MultiIndex (date_int, etf_code) and columns [ret_1d, ret_3d, ...]
    """
    date_to_idx = {d: i for i, d in enumerate(trading_dates)}
    records = []

    for date_int in trading_dates:
        if date_int not in close_df.index:
            continue
        t_idx = date_to_idx.get(date_int)
        if t_idx is None:
            continue
        close_t = close_df.loc[date_int]

        row = {"date": date_int}
        for n in horizons:
            tn_idx = t_idx + n
            if tn_idx >= len(trading_dates):
                break
            date_tn = trading_dates[tn_idx]
            if date_tn not in close_df.index:
                continue
            close_tn = close_df.loc[date_tn]
            ret = (close_tn - close_t) / close_t.replace(0, np.nan)
            for etf in close_df.columns:
                records.append({
                    "date": date_int,
                    "etf": etf,
                    f"ret_{n}d": ret.get(etf, np.nan) if hasattr(ret, "get") else ret[etf] if etf in ret.index else np.nan,
                })

    # Build pivoted form: (date, etf) → multiple return columns
    df_long = pd.DataFrame(records)
    if df_long.empty:
        return pd.DataFrame()

    # Aggregate multiple horizon columns
    df_pivot = df_long.groupby(["date", "etf"]).first().reset_index()
    return df_pivot


def _fast_forward_returns(
    close_df: pd.DataFrame,
    trading_dates: List[int],
    horizons: List[int],
) -> pd.DataFrame:
    """
    Efficient vectorised forward-return computation.
    Returns long DataFrame with columns: date, etf, ret_{n}d for each n in horizons.
    """
    # Build date-indexed numpy arrays for each ETF
    date_series = pd.Series(range(len(trading_dates)), index=trading_dates)
    valid_dates = [d for d in trading_dates if d in close_df.index]

    rows = []
    for etf in close_df.columns:
        prices = close_df[etf].reindex(trading_dates)  # NaN for non-existent dates
        for i, d in enumerate(trading_dates):
            p0 = prices.iloc[i] if i < len(prices) else np.nan
            if pd.isna(p0) or p0 == 0:
                continue
            row = {"date": d, "etf": etf}
            for n in horizons:
                j = i + n
                if j < len(trading_dates):
                    pn = prices.iloc[j]
                    row[f"ret_{n}d"] = (pn - p0) / p0 if not pd.isna(pn) else np.nan
                else:
                    row[f"ret_{n}d"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ── IC computation ─────────────────────────────────────────────────────────────

def cross_sectional_ic(
    signals: pd.Series,  # index = etf, values = signal
    returns: pd.Series,  # index = etf, values = return
) -> float:
    """Spearman IC for one date-horizon pair."""
    common = signals.index.intersection(returns.index)
    if len(common) < 3:
        return np.nan
    s = signals[common].values.astype(float)
    r = returns[common].values.astype(float)
    mask = ~(np.isnan(s) | np.isnan(r))
    if mask.sum() < 3:
        return np.nan
    corr, _ = spearmanr(s[mask], r[mask])
    return corr


def compute_daily_ic(
    score_df: pd.DataFrame,  # columns: date(int), etf, score
    return_df: pd.DataFrame,  # columns: date(int), etf, ret_{n}d
    horizons: List[int],
    score_col: str = "daily_score",
) -> pd.DataFrame:
    """
    For each (date, horizon), compute cross-sectional IC across 16 ETFs.
    Returns DataFrame: date, horizon, IC
    """
    merged = pd.merge(score_df, return_df, on=["date", "etf"], how="inner")
    rows = []
    for date_int, grp in merged.groupby("date"):
        for n in horizons:
            ret_col = f"ret_{n}d"
            if ret_col not in grp.columns:
                continue
            sig = grp.set_index("etf")[score_col]
            ret = grp.set_index("etf")[ret_col]
            ic = cross_sectional_ic(sig, ret)
            rows.append({"date": date_int, "horizon": n, "IC": ic})
    return pd.DataFrame(rows)


def ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate IC metrics by horizon."""
    records = []
    for n, grp in ic_df.groupby("horizon"):
        valid = grp["IC"].dropna()
        if len(valid) == 0:
            continue
        mean_ic = valid.mean()
        std_ic = valid.std()
        icir = mean_ic / std_ic if std_ic > 0 else np.nan
        records.append({
            "horizon": f"ret_{n}d",
            "n_days": n,
            "n_obs": len(valid),
            "mean_IC": round(mean_ic, 4),
            "std_IC": round(std_ic, 4),
            "ICIR": round(icir, 4),
            "pct_positive": round((valid > 0).mean(), 4),
            "pct_negative": round((valid < 0).mean(), 4),
        })
    return pd.DataFrame(records)


# ── Load score files ────────────────────────────────────────────────────────────

def load_daily_score_df(path: str) -> pd.DataFrame:
    """Load daily_scores.json → long DataFrame with columns: date(int), etf, daily_score."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    rows = []
    for date_str, entry in data.get("dates", {}).items():
        date_int = int(date_str.replace("-", ""))
        for etf, scores in entry.get("etf_scores", {}).items():
            rows.append({
                "date": date_int,
                "date_str": date_str,
                "etf": etf,
                "sentiment": scores.get("sentiment", "neutral"),
                "confidence": scores.get("confidence", 0.5),
                "daily_score": scores.get("daily_score", 0.0),
                "reason": scores.get("reason", ""),
            })
    return pd.DataFrame(rows)


def load_per_news_score_df(path: str) -> pd.DataFrame:
    """Load per_news_scores.jsonl → long DataFrame with (date, etf, score) per article."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            date_str = rec.get("date", "")
            if not date_str:
                continue
            date_int = int(date_str.replace("-", ""))
            for etf, scores in rec.get("etf_scores", {}).items():
                rows.append({
                    "date": date_int,
                    "date_str": date_str,
                    "etf": etf,
                    "source": rec.get("source", ""),
                    "ranking": rec.get("ranking", 0),
                    "title": rec.get("title", ""),
                    "relevance": scores.get("relevance", 0.0),
                    "direction": scores.get("direction", "neutral"),
                    "confidence": scores.get("confidence", 0.0),
                    "score": scores.get("score", 0.0),
                })
    return pd.DataFrame(rows)


# ── Plotting ────────────────────────────────────────────────────────────────────

def plot_ic_timeseries(ic_df: pd.DataFrame, output_dir: str, label: str = "daily") -> None:
    if not HAS_MATPLOTLIB:
        print("[ic_analysis] matplotlib not installed, skipping plots")
        return

    horizons = sorted(ic_df["horizon"].unique())
    fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 3 * len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, n in zip(axes, horizons):
        grp = ic_df[ic_df["horizon"] == n].copy()
        grp["date_dt"] = pd.to_datetime(grp["date"].astype(str), format="%Y%m%d")
        grp = grp.sort_values("date_dt")
        ax.bar(grp["date_dt"], grp["IC"], width=1, color=["green" if v > 0 else "red" for v in grp["IC"]], alpha=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        mean_ic = grp["IC"].mean()
        ax.axhline(mean_ic, color="blue", linestyle="--", linewidth=1, label=f"Mean IC={mean_ic:.3f}")
        ax.set_ylabel(f"IC (ret_{n}d)")
        ax.legend(fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())

    plt.suptitle(f"Cross-sectional IC Time Series ({label})", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"ic_timeseries_{label}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved IC time series plot: {out_path}")


def plot_ic_by_etf(score_df: pd.DataFrame, return_df: pd.DataFrame, horizons: List[int],
                   output_dir: str, score_col: str = "daily_score", label: str = "daily") -> pd.DataFrame:
    """For each ETF, compute time-series IC (correlation over all dates) at each horizon."""
    records = []
    merged = pd.merge(score_df, return_df, on=["date", "etf"], how="inner")
    for etf, grp in merged.groupby("etf"):
        row = {"etf": etf}
        for n in horizons:
            ret_col = f"ret_{n}d"
            if ret_col not in grp.columns:
                continue
            sig = grp[score_col].values.astype(float)
            ret = grp[ret_col].values.astype(float)
            mask = ~(np.isnan(sig) | np.isnan(ret))
            if mask.sum() < 3:
                row[f"IC_{n}d"] = np.nan
            else:
                corr, _ = spearmanr(sig[mask], ret[mask])
                row[f"IC_{n}d"] = round(corr, 4)
        records.append(row)

    df = pd.DataFrame(records)

    if HAS_MATPLOTLIB and not df.empty:
        ic_cols = [c for c in df.columns if c.startswith("IC_")]
        x = range(len(df))
        fig, axes = plt.subplots(1, len(ic_cols), figsize=(5 * len(ic_cols), 5), sharey=False)
        if len(ic_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, ic_cols):
            colors = ["green" if v > 0 else "red" for v in df[col].fillna(0)]
            ax.barh(df["etf"], df[col].fillna(0), color=colors, alpha=0.7)
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_title(col)
        plt.suptitle(f"Per-ETF Time-Series IC ({label})", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"ic_by_etf_{label}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved IC by ETF plot: {out_path}")

    return df


# ── Main ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="IC analysis for news sentiment scores")
    parser.add_argument("--scores-dir", default="ic_scores", help="Directory with scored output files")
    parser.add_argument(
        "--mode", choices=["daily", "per_news", "both"], default="both",
        help="Which score files to analyse"
    )
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[1, 3, 5, 20],
        help="Forward-return horizons in trading days"
    )
    parser.add_argument("--output-dir", default="ic_results", help="Output directory for results")
    parser.add_argument(
        "--per-news-agg", choices=["mean", "max", "first"], default="mean",
        help="How to aggregate multiple news-article scores per (date, etf) for IC analysis"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    daily_path = os.path.join(args.scores_dir, "daily_scores.json")
    per_news_path = os.path.join(args.scores_dir, "per_news_scores.jsonl")

    # Init data loader for price data
    data_loader = init_data_loader(
        str(DATA_DIRS["PRICE_DATA"]),
        str(DATA_DIRS["NEWS_DATA"]),
    )
    trading_dates = data_loader.trading_dates

    # Load close prices
    print("Loading ETF close prices...")
    close_df = load_close_prices(INDUSTRY_FUND_POOL)
    close_df = close_df.reindex(trading_dates)  # align to trading calendar

    # Compute forward returns
    print(f"Computing forward returns for horizons {args.horizons}...")
    return_df = _fast_forward_returns(close_df, trading_dates, args.horizons)
    print(f"  {len(return_df)} (date, etf) return pairs computed")

    # ── Daily aggregate IC ─────────────────────────────────────────────────────
    if args.mode in ("daily", "both") and os.path.exists(daily_path):
        print("\n=== Mode A: Daily Aggregate IC ===")
        score_df = load_daily_score_df(daily_path)
        print(f"  Loaded {len(score_df)} (date, etf) score pairs from {daily_path}")

        ic_df = compute_daily_ic(score_df, return_df, args.horizons, score_col="daily_score")
        summary_df = ic_summary(ic_df)
        print("\nIC Summary (Daily Aggregate):")
        print(summary_df.to_string(index=False))

        # Save
        summary_path = os.path.join(args.output_dir, "ic_summary_daily.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved: {summary_path}")

        ic_ts_path = os.path.join(args.output_dir, "ic_timeseries_daily.csv")
        ic_df.to_csv(ic_ts_path, index=False)
        print(f"Saved: {ic_ts_path}")

        plot_ic_timeseries(ic_df, args.output_dir, label="daily")

        etf_ic_df = plot_ic_by_etf(score_df, return_df, args.horizons, args.output_dir,
                                    score_col="daily_score", label="daily")
        etf_path = os.path.join(args.output_dir, "ic_by_etf_daily.csv")
        etf_ic_df.to_csv(etf_path, index=False)
        print(f"Saved: {etf_path}")

    elif args.mode in ("daily", "both"):
        print(f"[SKIP] daily_scores.json not found at {daily_path}")

    # ── Per-news IC ───────────────────────────────────────────────────────────
    if args.mode in ("per_news", "both") and os.path.exists(per_news_path):
        print("\n=== Mode B: Per-News IC ===")
        pn_df = load_per_news_score_df(per_news_path)
        print(f"  Loaded {len(pn_df)} (date, etf, article) records from {per_news_path}")

        # Aggregate per (date, etf) across multiple articles
        if args.per_news_agg == "mean":
            agg_df = pn_df.groupby(["date", "etf"])["score"].mean().reset_index()
        elif args.per_news_agg == "max":
            agg_df = pn_df.groupby(["date", "etf"])["score"].max().reset_index()
        else:  # first
            agg_df = pn_df.sort_values("ranking").groupby(["date", "etf"])["score"].first().reset_index()

        agg_df = agg_df.rename(columns={"score": "daily_score"})
        print(f"  After {args.per_news_agg} aggregation: {len(agg_df)} (date, etf) pairs")

        ic_df_pn = compute_daily_ic(agg_df, return_df, args.horizons, score_col="daily_score")
        summary_pn = ic_summary(ic_df_pn)
        print(f"\nIC Summary (Per-News, agg={args.per_news_agg}):")
        print(summary_pn.to_string(index=False))

        summary_path = os.path.join(args.output_dir, f"ic_summary_per_news_{args.per_news_agg}.csv")
        summary_pn.to_csv(summary_path, index=False)
        print(f"\nSaved: {summary_path}")

        ic_ts_path = os.path.join(args.output_dir, f"ic_timeseries_per_news_{args.per_news_agg}.csv")
        ic_df_pn.to_csv(ic_ts_path, index=False)
        print(f"Saved: {ic_ts_path}")

        plot_ic_timeseries(ic_df_pn, args.output_dir, label=f"per_news_{args.per_news_agg}")

        etf_ic_df_pn = plot_ic_by_etf(agg_df, return_df, args.horizons, args.output_dir,
                                       score_col="daily_score", label=f"per_news_{args.per_news_agg}")
        etf_path = os.path.join(args.output_dir, f"ic_by_etf_per_news_{args.per_news_agg}.csv")
        etf_ic_df_pn.to_csv(etf_path, index=False)
        print(f"Saved: {etf_path}")

        # Also analyse per-news scores WITHOUT aggregation (all article-level pairs)
        print("\n  --- Per-article IC (no aggregation, cross-(article×etf) Spearman) ---")
        pn_ret_df = pd.merge(pn_df, return_df, on=["date", "etf"], how="inner")
        article_ic_records = []
        for n in args.horizons:
            ret_col = f"ret_{n}d"
            if ret_col not in pn_ret_df.columns:
                continue
            sig = pn_ret_df["score"].values.astype(float)
            ret = pn_ret_df[ret_col].values.astype(float)
            mask = ~(np.isnan(sig) | np.isnan(ret))
            if mask.sum() < 10:
                article_ic_records.append({"horizon": f"ret_{n}d", "IC": np.nan, "n_obs": 0})
                continue
            corr, pval = spearmanr(sig[mask], ret[mask])
            article_ic_records.append({
                "horizon": f"ret_{n}d",
                "n_obs": int(mask.sum()),
                "IC_pooled": round(corr, 4),
                "p_value": round(pval, 6),
            })
        article_ic_df = pd.DataFrame(article_ic_records)
        print(article_ic_df.to_string(index=False))
        article_ic_path = os.path.join(args.output_dir, "ic_pooled_per_news.csv")
        article_ic_df.to_csv(article_ic_path, index=False)
        print(f"Saved: {article_ic_path}")

    elif args.mode in ("per_news", "both"):
        print(f"[SKIP] per_news_scores.jsonl not found at {per_news_path}")

    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
