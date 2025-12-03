import argparse
import ast
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
    }
)
_BASE_FONT = float(plt.rcParams.get("font.size", 10))
_LABEL_FONT = _BASE_FONT * 1.3


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def find_channel_metrics(log_folder: Path) -> Path:
    matches = sorted(
        log_folder.rglob("channel_metrics.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise FileNotFoundError(f"No channel_metrics.csv found under {log_folder}")
    return matches[0]


def _coerce_numeric(series: pd.Series) -> pd.Series:
    def parse_val(val):
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                parsed = ast.literal_eval(val)
            except Exception:
                return pd.NA
            return parse_val(parsed)
        if isinstance(val, (list, tuple)):
            if not val:
                return pd.NA
            # Assume binary; prefer the final entry if multiple probabilities are present.
            return parse_val(val[-1])
        return pd.NA

    return pd.to_numeric(series.map(parse_val), errors="coerce")


def ensure_length_seconds(df: pd.DataFrame) -> pd.DataFrame:
    if "length_in_seconds" not in df.columns and "length" in df.columns:
        df = df.copy()
        df["length_in_seconds"] = (df["length"] * 12 + 12) / 1000
    return df


def compute_bce(prob: pd.Series, target: pd.Series) -> pd.Series:
    eps = 1e-7
    prob = prob.astype(float).clip(eps, 1 - eps)
    target = target.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        bce = -(target * np.log(prob) + (1 - target) * np.log(1 - prob))
    return pd.Series(bce, index=prob.index)


def save_scatter(
    df: pd.DataFrame, x_col: str, y_col: str, title: str, y_label: str, out_path: Path
) -> Optional[Path]:
    data = df[[x_col, y_col]].dropna()
    if data.empty:
        print(f"[warn] No data for plot {out_path.name}")
        return None

    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, alpha=0.6, edgecolor="none")

    if len(x) > 1 and x.min() != x.max():
        lr = linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 100)
        y_line = lr.slope * x_line + lr.intercept
        ax.plot(
            x_line,
            y_line,
            color="firebrick",
            linewidth=2,
            linestyle="--",
            label="Linear fit",
        )
        if np.isfinite(lr.slope) and np.isfinite(lr.stderr):
            ci = 1.96 * lr.stderr
            info = (
                r"$\mathbf{Statistics}$"
                "\n"
                rf"$\beta={lr.slope:.3f}\ \pm\ {ci:.3f}$"
                "\n"
                rf"$r={lr.rvalue:.3f},\ p={lr.pvalue:.3f}$"
            )
            ax.text(
                0.02,
                0.98,
                info,
                transform=ax.transAxes,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round", facecolor="white", alpha=0.8, linewidth=0.5
                ),
            )
        ax.legend(fontsize=_LABEL_FONT)

    ax.set_xlabel(r"$\text{Channel Length (s)}$", fontsize=_LABEL_FONT)
    ax.set_ylabel(y_label, fontsize=_LABEL_FONT)
    if title:
        ax.set_title(title, fontsize=_LABEL_FONT)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[ok] Saved {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot channel metrics against channel length."
    )
    parser.add_argument(
        "log_folder",
        help="Name or path of the log folder under project logs/ containing channel_metrics.csv",
    )
    parser.add_argument(
        "-t",
        "--title",
        default=None,
        nargs="+",
        help="Optional title to add atop each plot (e.g., site name)",
    )
    args = parser.parse_args()

    project_root = _project_root()
    candidate = Path(args.log_folder)
    log_folder = (
        candidate if candidate.is_absolute() else (project_root / "logs" / candidate)
    )

    if not log_folder.exists():
        raise FileNotFoundError(f"Log folder does not exist: {log_folder}")

    channel_metrics_path = find_channel_metrics(log_folder)
    print(f"[info] Using channel metrics at {channel_metrics_path}")

    df = pd.read_csv(channel_metrics_path)
    df = ensure_length_seconds(df)

    if "length_in_seconds" not in df.columns:
        raise KeyError(
            "length_in_seconds column missing and length column not found to derive it."
        )

    df["ch_prob_num"] = _coerce_numeric(df.get("ch_prob", pd.Series(dtype=float)))
    df["ch_prob_cal_num"] = _coerce_numeric(
        df.get("ch_prob_calibrated", pd.Series(dtype=float))
    )
    df["target_num"] = _coerce_numeric(df.get("target", pd.Series(dtype=float)))

    df["pred_error"] = compute_bce(df["ch_prob_num"], df["target_num"])
    df["pred_error_calibrated"] = compute_bce(df["ch_prob_cal_num"], df["target_num"])

    out_dir = channel_metrics_path.parent
    title_str = " ".join(args.title) if args.title else None

    save_scatter(
        df,
        "length_in_seconds",
        "ch_prob_num",
        title_str,
        r"$\text{Channel Probability (uncalibrated)}$",
        out_dir / "channel_length_vs_prob.png",
    )
    save_scatter(
        df,
        "length_in_seconds",
        "ch_prob_cal_num",
        title_str,
        r"$\text{Channel Probability (calibrated)}$",
        out_dir / "channel_length_vs_prob_calibrated.png",
    )
    save_scatter(
        df,
        "length_in_seconds",
        "pred_error",
        title_str,
        r"$\text{Prediction Error (Uncalibrated)}$",
        out_dir / "channel_length_vs_pred_error.png",
    )
    save_scatter(
        df,
        "length_in_seconds",
        "pred_error_calibrated",
        title_str,
        r"$\text{Prediction Error (Calibrated)}$",
        out_dir / "channel_length_vs_pred_error_calibrated.png",
    )


if __name__ == "__main__":
    main()
