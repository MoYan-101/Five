import argparse
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _build_output_path(output_dir: str, output_file: str, with_timestamp: bool) -> str:
    os.makedirs(output_dir, exist_ok=True)
    base_name = output_file.strip() if output_file else "Fig4_topic_coherence_single.png"
    root, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".png"
    if with_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{root}_{ts}{ext}"
    else:
        base_name = f"{root}{ext}"
    return os.path.join(output_dir, base_name)


def _resolve_std_col(df: pd.DataFrame, metric_col: str, std_col: str | None) -> str | None:
    if std_col and std_col in df.columns:
        return std_col
    if metric_col.endswith("_Mean"):
        candidate = metric_col[:-5] + "_Std"
        if candidate in df.columns:
            return candidate
    return None


def _pick_selected_k(
    df: pd.DataFrame,
    k_col: str,
    metric_col: str,
    std_col: str | None,
    selected_col: str,
    selection_mode: str,
    manual_selected_k: int | None,
) -> int:
    if manual_selected_k is not None:
        return int(manual_selected_k)

    if selection_mode == "csv_flag" and selected_col in df.columns:
        selected_rows = df[df[selected_col].astype(str).str.lower().isin(["true", "1"])]
        if not selected_rows.empty:
            return int(selected_rows.iloc[0][k_col])

    if selection_mode == "upper_std":
        if std_col is None:
            return int(df.loc[df[metric_col].idxmax(), k_col])
        scores = df[metric_col].astype(float) + df[std_col].fillna(0.0).astype(float)
        return int(df.loc[scores.idxmax(), k_col])

    if selection_mode == "lower_std":
        if std_col is None:
            return int(df.loc[df[metric_col].idxmax(), k_col])
        scores = df[metric_col].astype(float) - df[std_col].fillna(0.0).astype(float)
        return int(df.loc[scores.idxmax(), k_col])

    return int(df.loc[df[metric_col].idxmax(), k_col])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot coherence curve (C_NPMI by K) from coherence-scan CSV."
    )
    parser.add_argument(
        "--input-csv",
        default=os.path.join("analysis_outputs", "analysis_report_coherence_scan.csv"),
        help="Path to coherence scan CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="Figure",
        help="Directory for output figure.",
    )
    parser.add_argument(
        "--output-file",
        default="Fig4_topic_coherence_single.png",
        help="Output filename.",
    )
    parser.add_argument(
        "--no-timestamp",
        action="store_true",
        help="Do not append timestamp to output filename.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional Run_Name filter when CSV contains multiple runs.",
    )
    parser.add_argument(
        "--k-col",
        default="K",
        help="Column name for topic number.",
    )
    parser.add_argument(
        "--metric-col",
        default="C_NPMI_Mean",
        help="Column name for main curve.",
    )
    parser.add_argument(
        "--std-col",
        default=None,
        help="Std column used for selection score (auto-try <metric>_Std).",
    )
    parser.add_argument(
        "--selected-col",
        default="Selected_K",
        help="Column marking selected K (bool-like).",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["upper_std", "mean", "csv_flag", "lower_std"],
        default="upper_std",
        help="How to select K for marker/vertical line. Default uses metric+std.",
    )
    parser.add_argument(
        "--selected-k",
        type=int,
        default=None,
        help="Manually set selected K (overrides selected_col).",
    )
    parser.add_argument(
        "--title",
        default="Topic Coherence by K",
        help="Figure title.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figure window after saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.exists(args.input_csv):
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    df = pd.read_csv(args.input_csv)
    if df.empty:
        raise ValueError(f"Input CSV is empty: {args.input_csv}")

    if args.run_name and "Run_Name" in df.columns:
        df = df[df["Run_Name"] == args.run_name].copy()
        if df.empty:
            raise ValueError(f"Run_Name '{args.run_name}' not found in {args.input_csv}")

    required_cols = [args.k_col, args.metric_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    df = df.sort_values(args.k_col).copy()
    std_col = _resolve_std_col(df, args.metric_col, args.std_col)
    if args.selection_mode in {"upper_std", "lower_std"} and std_col is None:
        print("[Warn] Std column not found. Falling back to mean-based selection.")

    selected_k = _pick_selected_k(
        df=df,
        k_col=args.k_col,
        metric_col=args.metric_col,
        std_col=std_col,
        selected_col=args.selected_col,
        selection_mode=args.selection_mode,
        manual_selected_k=args.selected_k,
    )

    if selected_k not in set(df[args.k_col].tolist()):
        raise ValueError(f"Selected K={selected_k} not found in column {args.k_col}.")

    metric_series = df[args.metric_col].astype(float)
    if std_col is not None:
        std_series = df[std_col].fillna(0.0).astype(float)
    else:
        std_series = pd.Series(0.0, index=df.index, dtype=float)

    if args.selection_mode == "upper_std" and std_col is not None:
        plot_series = metric_series + std_series
    elif args.selection_mode == "lower_std" and std_col is not None:
        plot_series = metric_series - std_series
    else:
        plot_series = metric_series

    selected_mask = df[args.k_col] == selected_k
    selected_y = float(plot_series.loc[selected_mask].iloc[0])

    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)

    x_vals = df[args.k_col].to_numpy()
    y_vals = plot_series.to_numpy()
    ax.plot(
        x_vals,
        y_vals,
        color="#1f77b4",
        marker="o",
        linewidth=2.2,
        markersize=7,
        label="C_NPMI",
    )

    ax.axvline(selected_k, color="#d62728", linestyle="--", linewidth=2.0, alpha=0.85)
    marker_label = f"Selected K={selected_k}"

    ax.scatter(
        [selected_k],
        [selected_y],
        marker="*",
        s=430,
        color="#2ca02c",
        zorder=6,
        label=marker_label,
    )

    y_min = float(y_vals.min())
    y_max = float(y_vals.max())
    y_span = y_max - y_min
    if y_span <= 0:
        y_span = max(0.01, abs(selected_y) * 0.05)

    # Slightly expand y-axis range to reduce crowding near top/bottom.
    y_pad = 0.10 * y_span
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    text_x = selected_k + 0.2
    if text_x > float(x_vals.max()) - 0.1:
        text_x = selected_k - 0.9

    # If the selected point is near the top, place label below the marker.
    if selected_y >= (y_max - 0.18 * y_span):
        text_y = selected_y - 0.12 * y_span
        va = "top"
    else:
        text_y = selected_y + 0.08 * y_span
        va = "bottom"

    ax.text(
        text_x,
        text_y,
        f"Selected K={selected_k}",
        color="#2ca02c",
        fontsize=18,
        weight="bold",
        va=va,
    )

    ax.set_title(args.title, fontsize=30, pad=16)
    ax.set_xlabel("K (Number of Topics)", fontsize=24)
    ax.set_ylabel("C_NPMI", fontsize=26)
    ax.tick_params(axis="both", labelsize=19)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=15, loc="best")
    fig.tight_layout()

    output_path = _build_output_path(
        output_dir=args.output_dir,
        output_file=args.output_file,
        with_timestamp=not args.no_timestamp,
    )
    fig.savefig(output_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print(f"[Done] Single-curve figure saved: {output_path}")
    selected_mean = float(df.loc[df[args.k_col] == selected_k, args.metric_col].iloc[0])
    selected_std = float(df.loc[df[args.k_col] == selected_k, std_col].iloc[0]) if std_col is not None else 0.0

    if args.selection_mode == "upper_std":
        score_val = selected_mean + selected_std
        score_name = f"{args.metric_col}+{std_col or '0'}"
    elif args.selection_mode == "lower_std":
        score_val = selected_mean - selected_std
        score_name = f"{args.metric_col}-{std_col or '0'}"
    else:
        score_val = selected_mean
        score_name = args.metric_col

    print(
        f"[Info] Selected K={selected_k}, mode={args.selection_mode}, "
        f"mean={selected_mean:.6f}, std={selected_std:.6f}, "
        f"plotted_value={selected_y:.6f}, score({score_name})={score_val:.6f}"
    )


if __name__ == "__main__":
    main()
