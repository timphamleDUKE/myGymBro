from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
BASELINE_DIR = BASE_DIR / "baseline"
XGBOOST_DIR = BASE_DIR / "xgboost"
XGBOOST_PLUS_DIR = BASE_DIR / "xgboost-plus"
BASELINE_PREDICTIONS_CSV = BASELINE_DIR / "baseline_predictions.csv"
XGBOOST_PREDICTIONS_CSV = XGBOOST_DIR / "xgboost_predictions.csv"
XGBOOST_PLUS_PREDICTIONS_CSV = XGBOOST_PLUS_DIR / "xgboost_plus_predictions.csv"


@dataclass(frozen=True)
class ModelEvaluationConfig:
    name: str
    slug: str
    predictions_csv: Path
    output_dir: Path


MODEL_EVALUATIONS = [
    ModelEvaluationConfig(
        name="Baseline",
        slug="baseline",
        predictions_csv=BASELINE_PREDICTIONS_CSV,
        output_dir=BASELINE_DIR,
    ),
    ModelEvaluationConfig(
        name="XGBoost",
        slug="xgboost",
        predictions_csv=XGBOOST_PREDICTIONS_CSV,
        output_dir=XGBOOST_DIR,
    ),
    ModelEvaluationConfig(
        name="XGBoost Plus",
        slug="xgboost_plus",
        predictions_csv=XGBOOST_PLUS_PREDICTIONS_CSV,
        output_dir=XGBOOST_PLUS_DIR,
    ),
]


def load_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {
        "exercise",
        "actual_next_weight",
        "predicted_next_weight",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in predictions CSV: {sorted(missing)}")
    return df


def add_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["error"] = scored["predicted_next_weight"] - scored["actual_next_weight"]
    scored["absolute_error"] = scored["error"].abs()
    scored["squared_error"] = scored["error"] ** 2
    scored["within_2_5_lb"] = scored["absolute_error"] <= 2.5
    scored["within_5_lb"] = scored["absolute_error"] <= 5.0
    scored["within_10_lb"] = scored["absolute_error"] <= 10.0
    return scored


def compute_metrics(df: pd.DataFrame) -> dict[str, float]:
    mae = float(df["absolute_error"].mean())
    rmse = float(df["squared_error"].mean() ** 0.5)
    mse = float(df["squared_error"].mean())
    mean_signed_error = float(df["error"].mean())

    return {
        "rows_evaluated": float(len(df)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "mse": round(mse, 3),
        "mean_signed_error": round(mean_signed_error, 3),
        "within_2_5_lb_accuracy": round(float(df["within_2_5_lb"].mean()), 4),
        "within_5_lb_accuracy": round(float(df["within_5_lb"].mean()), 4),
        "within_10_lb_accuracy": round(float(df["within_10_lb"].mean()), 4),
    }


def build_per_exercise_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("exercise", as_index=False)
        .agg(
            samples=("exercise", "size"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda s: (s.mean() ** 0.5)),
            mean_signed_error=("error", "mean"),
            within_5_lb_accuracy=("within_5_lb", "mean"),
        )
        .sort_values(["samples", "mae"], ascending=[False, True])
        .reset_index(drop=True)
    )

    numeric_columns = ["mae", "rmse", "mean_signed_error", "within_5_lb_accuracy"]
    for column in numeric_columns:
        summary[column] = summary[column].round(3)

    return summary


def save_metrics(metrics: dict[str, float], output_path: Path) -> None:
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(output_path, index=False)


def save_per_exercise_summary(summary: pd.DataFrame, output_path: Path) -> None:
    summary.to_csv(output_path, index=False)


def plot_actual_vs_predicted(df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(df["actual_next_weight"], df["predicted_next_weight"], alpha=0.6)

    min_weight = min(df["actual_next_weight"].min(), df["predicted_next_weight"].min())
    max_weight = max(df["actual_next_weight"].max(), df["predicted_next_weight"].max())
    plt.plot([min_weight, max_weight], [min_weight, max_weight], linestyle="--")

    plt.xlabel("Actual Next Weight")
    plt.ylabel("Predicted Next Weight")
    plt.title(f"{model_name}: Actual vs Predicted Next Weight")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_error_histogram(df: pd.DataFrame, output_path: Path, model_name: str) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(df["error"], bins=30)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction Error (Predicted - Actual)")
    plt.ylabel("Count")
    plt.title(f"{model_name} Prediction Error Distribution")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_top_exercise_mae(
    summary: pd.DataFrame,
    output_path: Path,
    model_name: str,
    top_n: int = 10,
) -> None:
    top_summary = summary.head(top_n).copy()
    if top_summary.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.barh(top_summary["exercise"], top_summary["mae"])
    plt.gca().invert_yaxis()
    plt.xlabel("MAE")
    plt.ylabel("Exercise")
    plt.title(f"{model_name} MAE by Exercise (Top {len(top_summary)} by sample count)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def print_report(model_name: str, metrics: dict[str, float], summary: pd.DataFrame) -> None:
    print(f"{model_name} Model Evaluation")
    print(f"Rows evaluated: {int(metrics['rows_evaluated'])}")
    print(f"MAE: {metrics['mae']}")
    print(f"RMSE: {metrics['rmse']}")
    print(f"MSE: {metrics['mse']}")
    print(f"Mean signed error: {metrics['mean_signed_error']}")
    print(f"Within 2.5 lb accuracy: {metrics['within_2_5_lb_accuracy']}")
    print(f"Within 5 lb accuracy: {metrics['within_5_lb_accuracy']}")
    print(f"Within 10 lb accuracy: {metrics['within_10_lb_accuracy']}")
    print()
    print("Top exercises by sample count:")
    print(summary.head(10).to_string(index=False))


def evaluate_model(config: ModelEvaluationConfig) -> None:
    if not config.predictions_csv.exists():
        print(f"[SKIP] {config.name}: predictions CSV not found at {config.predictions_csv}")
        print()
        return

    predictions_df = load_predictions(config.predictions_csv)
    scored_df = add_error_columns(predictions_df)
    metrics = compute_metrics(scored_df)
    summary = build_per_exercise_summary(scored_df)

    plots_dir = config.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    save_metrics(metrics, config.output_dir / f"{config.slug}_metrics.csv")
    save_per_exercise_summary(
        summary,
        config.output_dir / f"{config.slug}_per_exercise_metrics.csv",
    )

    plot_actual_vs_predicted(
        scored_df,
        plots_dir / f"{config.slug}_actual_vs_predicted.png",
        config.name,
    )
    plot_error_histogram(
        scored_df,
        plots_dir / f"{config.slug}_error_histogram.png",
        config.name,
    )
    plot_top_exercise_mae(
        summary,
        plots_dir / f"{config.slug}_top_exercise_mae.png",
        config.name,
    )

    print_report(config.name, metrics, summary)
    print()


def main() -> None:
    for config in MODEL_EVALUATIONS:
        evaluate_model(config)


if __name__ == "__main__":
    main()
