import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================================
# KONFIGŪRACIJA
# =========================================================
RUN_DIR = Path(".").resolve()
OUT_DIR = RUN_DIR / "plots_evalonly_advanced"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_SEG_PATH = RUN_DIR / "baseline_segment_metrics_evalonly.csv"
FT_SEG_PATH   = RUN_DIR / "finetuned_segment_metrics_evalonly.csv"
CMP_PATH      = RUN_DIR / "baseline_vs_finetuned_comparison_evalonly.csv"
SAMPLES_PATH  = RUN_DIR / "sample_comparison_predictions_evalonly.csv"
AGE_PATH      = RUN_DIR / "metrics_by_age_group_evalonly.csv"
TYPE_PATH     = RUN_DIR / "metrics_by_recording_type_evalonly.csv"
GENDER_PATH   = RUN_DIR / "metrics_by_gender_evalonly.csv"
LOSSY_PATH    = RUN_DIR / "metrics_by_lossy_evalonly.csv"

# =========================================================
# PAGALBINĖS FUNKCIJOS
# =========================================================
def log(msg: str) -> None:
    print(msg, flush=True)

def ensure_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Nerastas failas: {path}")

def save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    log(f"Išsaugotas grafikas: {path.name}")

def plot_group_metric_bars(agg_df: pd.DataFrame, group_col: str, metric: str, out_name: str, title: str):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(agg_df))
    width = 0.35

    ax.bar(x - width/2, agg_df[f"{metric}_baseline"], width, label="baseline")
    ax.bar(x + width/2, agg_df[f"{metric}_finetuned"], width, label="finetuned")

    ax.set_xticks(x)
    ax.set_xticklabels(agg_df[group_col], rotation=30 if len(agg_df) > 4 else 0, ha="right")
    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(metric.upper())
    ax.legend()

    save_plot(fig, OUT_DIR / out_name)

# =========================================================
# PAGRINDINĖ EIGA
# =========================================================
def main():
    log("=== Paleidžiamas plot_evalonly.py ===")
    log(f"RUN_DIR: {RUN_DIR}")
    log(f"OUT_DIR: {OUT_DIR}")

    for p in [BASE_SEG_PATH, FT_SEG_PATH, CMP_PATH, SAMPLES_PATH, AGE_PATH, TYPE_PATH, GENDER_PATH, LOSSY_PATH]:
        ensure_exists(p)

    base_seg = pd.read_csv(BASE_SEG_PATH)
    ft_seg = pd.read_csv(FT_SEG_PATH)
    cmp_df = pd.read_csv(CMP_PATH)
    samples_df = pd.read_csv(SAMPLES_PATH)
    age_agg = pd.read_csv(AGE_PATH)
    type_agg = pd.read_csv(TYPE_PATH)
    gender_agg = pd.read_csv(GENDER_PATH)
    lossy_agg = pd.read_csv(LOSSY_PATH)

    log(f"baseline segment rows: {len(base_seg)}")
    log(f"finetuned segment rows: {len(ft_seg)}")
    log(f"samples rows: {len(samples_df)}")

    base_row = cmp_df[cmp_df["model_variant"] == "baseline_non_finetuned"].iloc[0]
    ft_row   = cmp_df[cmp_df["model_variant"] == "finetuned_lora"].iloc[0]

    metrics = ["wer", "cer", "sem", "ped", "kr"]
    metric_labels = [m.upper() for m in metrics]

    summary_df = pd.DataFrame({
        "metric": metric_labels,
        "baseline": [base_row["wer"], base_row["cer"], base_row["sem"], base_row["ped"], base_row["kr"]],
        "finetuned": [ft_row["wer"], ft_row["cer"], ft_row["sem"], ft_row["ped"], ft_row["kr"]],
        "absolute_improvement": [
            base_row["wer"] - ft_row["wer"],
            base_row["cer"] - ft_row["cer"],
            ft_row["sem"] - base_row["sem"],
            base_row["ped"] - ft_row["ped"],
            ft_row["kr"] - base_row["kr"],
        ],
    })
    summary_df.to_csv(OUT_DIR / "table_main_metrics.csv", index=False)
    with open(OUT_DIR / "table_main_metrics.tex", "w", encoding="utf-8") as f:
        f.write(summary_df.to_latex(index=False, float_format=lambda x: f"{x:.4f}"))

    # 1. Pagrindinis metrikų grafikas
    fig, ax = plt.subplots(figsize=(10, 5))
    summary_df.set_index("metric")[["baseline", "finetuned"]].plot(kind="bar", ax=ax)
    ax.set_title("Bazinio ir LoRA modelių metrikų palyginimas")
    ax.set_xlabel("Metrika")
    ax.set_ylabel("Reikšmė")
    ax.tick_params(axis="x", rotation=0)
    save_plot(fig, OUT_DIR / "01_main_metrics_grouped.png")

    # 2. Pagerėjimo grafikas
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(summary_df["metric"], summary_df["absolute_improvement"])
    ax.set_title("LoRA modelio pagerėjimas bazinio modelio atžvilgiu")
    ax.set_xlabel("Metrika")
    ax.set_ylabel("Pagerėjimas")
    save_plot(fig, OUT_DIR / "02_metric_improvements.png")

    # 3. Prognozių ilgio boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(
        [base_seg["pred_len"], ft_seg["pred_len"]],
        tick_labels=["baseline", "finetuned"]
    )
    ax.set_title("Prognozių ilgio pasiskirstymas")
    ax.set_ylabel("Simbolių skaičius")
    save_plot(fig, OUT_DIR / "03_prediction_length_boxplot.png")

    # 4. Tuščių prognozių skaičius
    empty_counts = pd.DataFrame({
        "model": ["baseline", "finetuned"],
        "empty_predictions": [
            int(base_seg["is_empty_prediction"].sum()),
            int(ft_seg["is_empty_prediction"].sum()),
        ]
    })
    empty_counts.to_csv(OUT_DIR / "table_empty_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(empty_counts["model"], empty_counts["empty_predictions"])
    ax.set_title("Tuščių prognozių skaičius")
    ax.set_ylabel("Kiekis")
    save_plot(fig, OUT_DIR / "04_empty_predictions.png")

    # 5. WER/PED histogramų palyginimas
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(base_seg["wer"].dropna(), bins=40, alpha=0.7, label="baseline")
    axes[0].hist(ft_seg["wer"].dropna(), bins=40, alpha=0.7, label="finetuned")
    axes[0].set_title("WER pasiskirstymas")
    axes[0].legend()

    axes[1].hist(base_seg["ped"].dropna(), bins=40, alpha=0.7, label="baseline")
    axes[1].hist(ft_seg["ped"].dropna(), bins=40, alpha=0.7, label="finetuned")
    axes[1].set_title("PED pasiskirstymas")
    axes[1].legend()

    save_plot(fig, OUT_DIR / "05_wer_ped_histograms.png")

    # 6. SEM vs WER scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(base_seg["wer"], base_seg["sem"], alpha=0.35, label="baseline")
    ax.scatter(ft_seg["wer"], ft_seg["sem"], alpha=0.35, label="finetuned")
    ax.set_title("WER ir SEM ryšys segmentų lygmeniu")
    ax.set_xlabel("WER")
    ax.set_ylabel("SEM")
    ax.legend()
    save_plot(fig, OUT_DIR / "06_wer_sem_scatter.png")

    # 7. Dashboard pagal amžiaus grupę
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, metric in zip(axes.ravel(), ["wer", "cer", "sem", "kr"]):
        x = np.arange(len(age_agg))
        width = 0.35
        ax.bar(x - width/2, age_agg[f"{metric}_baseline"], width, label="baseline")
        ax.bar(x + width/2, age_agg[f"{metric}_finetuned"], width, label="finetuned")
        ax.set_xticks(x)
        ax.set_xticklabels(age_agg["age_group"])
        ax.set_title(f"{metric.upper()} pagal amžiaus grupę")
        ax.set_ylabel(metric.upper())
    axes[0, 0].legend()
    save_plot(fig, OUT_DIR / "07_age_dashboard.png")

    # 8. Dashboard pagal įrašo tipą
    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, metric in zip(axes.ravel(), ["wer", "cer", "sem", "kr"]):
        x = np.arange(len(type_agg))
        width = 0.35
        ax.bar(x - width/2, type_agg[f"{metric}_baseline"], width, label="baseline")
        ax.bar(x + width/2, type_agg[f"{metric}_finetuned"], width, label="finetuned")
        ax.set_xticks(x)
        ax.set_xticklabels(type_agg["recording_type"], rotation=30, ha="right")
        ax.set_title(f"{metric.upper()} pagal įrašo tipą")
        ax.set_ylabel(metric.upper())
    axes[0, 0].legend()
    save_plot(fig, OUT_DIR / "08_recording_type_dashboard.png")

    # 9–16. Atskiri grafikai pagal grupes
    plot_group_metric_bars(age_agg, "age_group", "wer", "09_age_wer.png", "WER pagal amžiaus grupę")
    plot_group_metric_bars(age_agg, "age_group", "cer", "10_age_cer.png", "CER pagal amžiaus grupę")
    plot_group_metric_bars(age_agg, "age_group", "sem", "11_age_sem.png", "SEM pagal amžiaus grupę")
    plot_group_metric_bars(age_agg, "age_group", "kr", "12_age_kr.png", "KR pagal amžiaus grupę")

    plot_group_metric_bars(type_agg, "recording_type", "wer", "13_type_wer.png", "WER pagal įrašo tipą")
    plot_group_metric_bars(type_agg, "recording_type", "cer", "14_type_cer.png", "CER pagal įrašo tipą")
    plot_group_metric_bars(type_agg, "recording_type", "sem", "15_type_sem.png", "SEM pagal įrašo tipą")
    plot_group_metric_bars(type_agg, "recording_type", "kr", "16_type_kr.png", "KR pagal įrašo tipą")

    # 17–20. Lytis / Lossy-Raw
    plot_group_metric_bars(gender_agg, "gender_label", "wer", "17_gender_wer.png", "WER pagal lytį")
    plot_group_metric_bars(gender_agg, "gender_label", "ped", "18_gender_ped.png", "PED pagal lytį")
    plot_group_metric_bars(lossy_agg, "lossy_label", "wer", "19_lossy_wer.png", "WER pagal Lossy / Raw")
    plot_group_metric_bars(lossy_agg, "lossy_label", "ped", "20_lossy_ped.png", "PED pagal Lossy / Raw")

    # 21. Geriausių pagerėjimų lentelė
    top_cols = [
        "segment_id",
        "reference",
        "prediction_raw_baseline",
        "prediction_raw_finetuned",
        "wer_baseline", "wer_finetuned",
        "cer_baseline", "cer_finetuned",
        "sem_baseline", "sem_finetuned",
        "ped_baseline", "ped_finetuned",
        "kr_baseline", "kr_finetuned",
        "combined_gain",
    ]
    top_best = samples_df.sort_values("combined_gain", ascending=False).head(20)
    top_best[top_cols].to_csv(OUT_DIR / "table_top_best_segments.csv", index=False)

    top_worst = samples_df.sort_values("combined_gain", ascending=True).head(20)
    top_worst[top_cols].to_csv(OUT_DIR / "table_top_worst_segments.csv", index=False)

    # 22. Tekstinė santrauka
    report_lines = []
    report_lines.append("PAGRINDINĖ SANTRAUKA")
    report_lines.append("")
    for _, row in summary_df.iterrows():
        report_lines.append(
            f"{row['metric']}: baseline={row['baseline']:.4f}, "
            f"finetuned={row['finetuned']:.4f}, "
            f"improvement={row['absolute_improvement']:.4f}"
        )
    report_lines.append("")
    report_lines.append(f"Tuščios baseline prognozės: {int(base_seg['is_empty_prediction'].sum())}")
    report_lines.append(f"Tuščios finetuned prognozės: {int(ft_seg['is_empty_prediction'].sum())}")

    with open(OUT_DIR / "summary_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    log("=== Sugeneruoti failai ===")
    for p in sorted(OUT_DIR.glob("*")):
        log(f"- {p.name}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nKLAIDA: {e}", file=sys.stderr, flush=True)
        raise