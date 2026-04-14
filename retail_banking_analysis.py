from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent
DATASET_CANDIDATES = [
    "Enhanced_Retail_Banking_Dataset.csv",
    "Retail_Banking_Dataset_JKBank.csv",
]
FINAL_OUTPUT_FILE = BASE_DIR / "final_project_output.csv"
MODEL_METRICS_FILE = BASE_DIR / "summary_model_metrics.csv"
SERVQUAL_CORRELATION_FILE = BASE_DIR / "summary_servqual_correlations.csv"
FEATURE_IMPORTANCE_FILE = BASE_DIR / "summary_feature_importance.csv"
THRESHOLD_FILE = BASE_DIR / "summary_thresholds.csv"
RELATIONSHIP_MANAGER_FILE = BASE_DIR / "summary_relationship_manager_effect.csv"
HIGH_RISK_CUSTOMERS_FILE = BASE_DIR / "likely_to_leave_customers.csv"

SEED = 42
RISK_BAND_SCORE_MAP = {
    "Band 1 - Very Low Risk": 5,
    "Band 2 - Low Risk": 4,
    "Band 3 - Moderate Risk": 3,
    "Band 4 - High Risk": 2,
    "Band 5 - Very High Risk": 1,
}
SERVQUAL_DIMENSION_COLUMNS = [
    "servqual_reliability",
    "servqual_responsiveness",
    "servqual_assurance",
    "servqual_empathy",
    "servqual_tangibles",
]
CONTROL_FEATURE_COLUMNS = [
    "income",
    "customer_tenure",
    "clv",
    "market_rate_diff",
    "product_count",
]
FEATURE_COLUMNS = SERVQUAL_DIMENSION_COLUMNS + CONTROL_FEATURE_COLUMNS
CHART_FILES = {
    "heatmap": BASE_DIR / "chart_servqual_retention_heatmap.png",
    "sqi_boxplot": BASE_DIR / "chart_sqi_by_target.png",
    "feature_importance": BASE_DIR / "chart_servqual_feature_importance.png",
    "app_threshold": BASE_DIR / "chart_app_rating_leave_risk.png",
    "resolution_threshold": BASE_DIR / "chart_resolution_days_leave_risk.png",
    "relationship_manager": BASE_DIR / "chart_relationship_manager_retention.png",
}


def locate_dataset():
    for file_name in DATASET_CANDIDATES:
        path = BASE_DIR / file_name
        if path.exists():
            return path

    candidate_list = ", ".join(DATASET_CANDIDATES)
    raise FileNotFoundError(
        f"No source dataset was found. Expected one of: {candidate_list}"
    )


def latest_fallback_path(path):
    return path.with_name(f"{path.stem}_latest{path.suffix}")


def write_csv_with_fallback(frame, preferred_path, index=False):
    try:
        frame.to_csv(preferred_path, index=index)
        return preferred_path
    except PermissionError:
        fallback_path = latest_fallback_path(preferred_path)
        frame.to_csv(fallback_path, index=index)
        print(
            f"Warning: {preferred_path.name} was locked, so the export was written to "
            f"{fallback_path.name} instead."
        )
        return fallback_path


def validate_columns(df):
    required_columns = {
        "customer_id",
        "transaction_accuracy",
        "fcr_rate",
        "resolution_days",
        "credit_score",
        "risk_band",
        "has_relationship_manager",
        "app_rating_score",
        "retention_score",
        "churn_flag",
        "income",
        "customer_tenure",
        "clv",
        "market_rate_diff",
        "product_count",
        "loan_type",
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "The dataset is missing the required columns: "
            + ", ".join(missing_columns)
        )


def normalize_series(series, higher_is_better=True):
    values = pd.to_numeric(series, errors="coerce")

    if values.notna().sum() == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)

    min_value = values.min()
    max_value = values.max()

    if np.isclose(min_value, max_value):
        normalized = pd.Series(0.5, index=series.index, dtype=float)
    else:
        normalized = (values - min_value) / (max_value - min_value)

    return normalized if higher_is_better else 1 - normalized


def build_servqual_framework(df):
    framework_df = df.copy()

    framework_df["risk_band_score"] = framework_df["risk_band"].map(RISK_BAND_SCORE_MAP)
    framework_df["risk_band_score"] = framework_df["risk_band_score"].fillna(
        framework_df["risk_band_score"].median()
    )

    # Reliability matters because error-free transactions and first-contact fixes
    # reduce operational friction that can quietly push good customers away.
    framework_df["transaction_accuracy_norm"] = normalize_series(
        framework_df["transaction_accuracy"], higher_is_better=True
    )
    framework_df["fcr_rate_norm"] = normalize_series(
        framework_df["fcr_rate"], higher_is_better=True
    )
    framework_df["servqual_reliability"] = framework_df[
        ["transaction_accuracy_norm", "fcr_rate_norm"]
    ].mean(axis=1)

    # Responsiveness is inverted because longer resolution times weaken trust.
    framework_df["resolution_days_norm"] = normalize_series(
        framework_df["resolution_days"], higher_is_better=False
    )
    framework_df["servqual_responsiveness"] = framework_df["resolution_days_norm"]

    # Assurance blends credit quality and bank-assessed risk standing to reflect
    # how safe and professionally managed the relationship appears to the customer.
    framework_df["credit_score_norm"] = normalize_series(
        framework_df["credit_score"], higher_is_better=True
    )
    framework_df["risk_band_score_norm"] = normalize_series(
        framework_df["risk_band_score"], higher_is_better=True
    )
    framework_df["servqual_assurance"] = framework_df[
        ["credit_score_norm", "risk_band_score_norm"]
    ].mean(axis=1)

    # Empathy is modeled as whether a human relationship layer exists.
    framework_df["servqual_empathy"] = (
        pd.to_numeric(
            framework_df["has_relationship_manager"], errors="coerce"
        ).fillna(0)
    ).clip(lower=0, upper=1)

    # Tangibles in this study are digital rather than physical branch assets.
    framework_df["app_rating_score_norm"] = normalize_series(
        framework_df["app_rating_score"], higher_is_better=True
    )
    framework_df["servqual_tangibles"] = framework_df["app_rating_score_norm"]

    framework_df["service_quality_index"] = framework_df[
        SERVQUAL_DIMENSION_COLUMNS
    ].mean(axis=1)
    framework_df["service_quality_index_pct"] = (
        framework_df["service_quality_index"] * 100
    ).round(2)

    return framework_df


def build_leave_target(df):
    working_df = df.copy()
    observed_churn = pd.to_numeric(
        working_df["churn_flag"], errors="coerce"
    ).fillna(0).astype(int)
    positive_cases = int((observed_churn == 1).sum())

    if observed_churn.nunique() > 1 and positive_cases > 0:
        working_df["model_target_flag"] = observed_churn
        return {
            "data": working_df,
            "target_column": "model_target_flag",
            "target_strategy": "observed_churn",
            "target_note": (
                "The model uses the observed churn_flag because both leavers and "
                "stayers are present in the dataset."
            ),
            "cutoff_value": np.nan,
            "positive_rate": round(working_df["model_target_flag"].mean(), 4),
        }

    retention_cutoff = float(working_df["retention_score"].quantile(0.25))

    # When the bank only has pre-churn customer data, the most actionable proxy is
    # the lowest-retention cohort. We intentionally do not use retention_score as a
    # model input, so the Random Forest still learns from service and financial signals.
    working_df["model_target_flag"] = (
        working_df["retention_score"] <= retention_cutoff
    ).astype(int)

    return {
        "data": working_df,
        "target_column": "model_target_flag",
        "target_strategy": "retention_proxy",
        "target_note": (
            "Observed churn_flag contains no leavers in this dataset, so the model "
            f"uses the bottom quartile of retention_score (<= {retention_cutoff:.2f}) "
            "as a proxy for customers most likely to leave."
        ),
        "cutoff_value": round(retention_cutoff, 4),
        "positive_rate": round(working_df["model_target_flag"].mean(), 4),
    }


def build_model(df, target_column):
    model_df = df.copy()
    X = model_df[FEATURE_COLUMNS].copy()
    y = model_df[target_column].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y,
    )

    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=5,
                    random_state=SEED,
                    class_weight="balanced_subsample",
                    n_jobs=1,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    model_df["predicted_leave_probability"] = model.predict_proba(X)[:, 1]
    model_df["predicted_leave_flag"] = (
        model_df["predicted_leave_probability"] >= 0.50
    ).astype(int)

    percentile_rank = model_df["predicted_leave_probability"].rank(pct=True)
    model_df["leave_risk_segment"] = np.select(
        [
            percentile_rank >= 0.95,
            percentile_rank >= 0.80,
            percentile_rank >= 0.50,
        ],
        ["Critical", "High", "Moderate"],
        default="Low",
    )

    feature_importance = pd.Series(
        model.named_steps["classifier"].feature_importances_,
        index=FEATURE_COLUMNS,
        name="importance",
    ).sort_values(ascending=False)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "test_rows": int(len(y_test)),
        "predicted_positive_rate_test": round(float(y_prob.mean()), 4),
    }

    return model_df, model, feature_importance, metrics


def strongest_servqual_dimension(feature_importance):
    servqual_importance = feature_importance[feature_importance.index.isin(
        SERVQUAL_DIMENSION_COLUMNS
    )]
    strongest_dimension = servqual_importance.idxmax()
    strongest_value = float(servqual_importance.max())
    return strongest_dimension, strongest_value


def build_threshold_summary(df, target_column):
    threshold_rows = []

    for feature_name, direction in [
        ("app_rating_score", "lower"),
        ("resolution_days", "higher"),
    ]:
        tree = DecisionTreeClassifier(max_depth=1, random_state=SEED)
        tree.fit(df[[feature_name]], df[target_column])

        threshold = float(tree.tree_.threshold[0])

        if direction == "lower":
            risky_mask = df[feature_name] <= threshold
            business_rule = f"{feature_name} <= {threshold:.1f}"
        else:
            risky_mask = df[feature_name] > threshold
            business_rule = f"{feature_name} > {threshold:.1f}"

        risky_rate = float(df.loc[risky_mask, target_column].mean())
        safer_rate = float(df.loc[~risky_mask, target_column].mean())
        threshold_rows.append(
            {
                "feature": feature_name,
                "risk_direction": direction,
                "threshold": round(threshold, 4),
                "business_rule": business_rule,
                "risky_group_customer_count": int(risky_mask.sum()),
                "other_group_customer_count": int((~risky_mask).sum()),
                "risky_group_leave_rate": round(risky_rate, 4),
                "other_group_leave_rate": round(safer_rate, 4),
                "risk_lift_multiple": round(
                    risky_rate / safer_rate, 4
                ) if safer_rate > 0 else np.nan,
            }
        )

    return pd.DataFrame(threshold_rows)


def build_relationship_manager_summary(df, target_column):
    rm_summary = (
        df.assign(
            relationship_manager_status=np.where(
                df["has_relationship_manager"] == 1,
                "Has Relationship Manager",
                "No Relationship Manager",
            )
        )
        .groupby("relationship_manager_status", as_index=False)
        .agg(
            customer_count=("customer_id", "count"),
            avg_retention_score=("retention_score", "mean"),
            avg_service_quality_index=("service_quality_index", "mean"),
            avg_predicted_leave_probability=("predicted_leave_probability", "mean"),
            likely_to_leave_rate=(target_column, "mean"),
        )
    )
    rm_summary["retention_rate_proxy"] = 1 - rm_summary["likely_to_leave_rate"]
    return rm_summary.round(4)


def recommend_action(row):
    if row["predicted_leave_probability"] >= 0.80 and row["has_relationship_manager"] == 0:
        return "Assign relationship manager and call within 24 hours"
    if row["resolution_days"] >= 7:
        return "Escalate service recovery and close complaint loop"
    if row["app_rating_score"] <= 3:
        return "Launch digital experience rescue journey"
    if row["servqual_reliability"] <= 0.45:
        return "Review repeat service failures and first-call resolution"
    return "Monitor closely and send personalized retention offer"


def build_high_risk_customer_extract(df):
    export_columns = [
        "customer_id",
        "loan_type",
        "income",
        "customer_tenure",
        "clv",
        "retention_score",
        "service_quality_index",
        "predicted_leave_probability",
        "leave_risk_segment",
        "app_rating_score",
        "resolution_days",
        "has_relationship_manager",
        "servqual_reliability",
        "servqual_responsiveness",
        "servqual_assurance",
        "servqual_empathy",
        "servqual_tangibles",
        "recommended_action",
    ]
    return df.sort_values(
        ["predicted_leave_probability", "clv"],
        ascending=[False, False],
    )[export_columns].head(250)


def plot_heatmap(df):
    corr_matrix = df[SERVQUAL_DIMENSION_COLUMNS + ["retention_score"]].corr()

    plt.figure(figsize=(9, 6))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Correlation"},
    )
    plt.title("SERVQUAL Dimensions vs Retention Score")
    plt.tight_layout()
    plt.savefig(CHART_FILES["heatmap"], dpi=160, bbox_inches="tight")
    plt.close()

    return corr_matrix.loc[SERVQUAL_DIMENSION_COLUMNS, ["retention_score"]].rename(
        columns={"retention_score": "correlation_with_retention_score"}
    ).sort_values("correlation_with_retention_score", ascending=False)


def plot_sqi_boxplot(df, target_strategy):
    label_map = {0: "Lower Leave Risk", 1: "Higher Leave Risk"}
    plot_df = df.copy()
    plot_df["target_label"] = plot_df["model_target_flag"].map(label_map)

    plt.figure(figsize=(8, 5))
    sns.boxplot(
        data=plot_df,
        x="target_label",
        y="service_quality_index",
        hue="target_label",
        palette=["#2a9d8f", "#e76f51"],
        legend=False,
    )
    plt.title("Service Quality Index by Leave-Risk Class")
    plt.xlabel(
        "Observed Churn Group" if target_strategy == "observed_churn" else "Proxy Leave-Risk Group"
    )
    plt.ylabel("Service Quality Index")
    plt.tight_layout()
    plt.savefig(CHART_FILES["sqi_boxplot"], dpi=160, bbox_inches="tight")
    plt.close()


def plot_feature_importance(feature_importance):
    plot_df = (
        feature_importance.reset_index()
        .rename(columns={"index": "feature"})
        .assign(
            feature_group=lambda x: np.where(
                x["feature"].isin(SERVQUAL_DIMENSION_COLUMNS),
                "SERVQUAL Dimension",
                "Control Variable",
            )
        )
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=plot_df,
        y="feature",
        x="importance",
        hue="feature_group",
        dodge=False,
        palette={
            "SERVQUAL Dimension": "#1f77b4",
            "Control Variable": "#9aa5b1",
        },
    )
    plt.title("Random Forest Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.legend(title="")
    plt.tight_layout()
    plt.savefig(CHART_FILES["feature_importance"], dpi=160, bbox_inches="tight")
    plt.close()


def plot_threshold_profiles(df, target_column):
    app_profile = (
        df.groupby("app_rating_score", as_index=False)[target_column]
        .mean()
        .rename(columns={target_column: "leave_rate"})
    )
    resolution_profile = (
        df.groupby("resolution_days", as_index=False)[target_column]
        .mean()
        .rename(columns={target_column: "leave_rate"})
    )
    overall_rate = df[target_column].mean()

    plt.figure(figsize=(8, 5))
    sns.barplot(data=app_profile, x="app_rating_score", y="leave_rate", color="#457b9d")
    plt.axhline(overall_rate, color="#e76f51", linestyle="--", linewidth=1.5)
    plt.title("Leave Risk by App Rating Score")
    plt.xlabel("App Rating Score")
    plt.ylabel("Leave-Risk Rate")
    plt.tight_layout()
    plt.savefig(CHART_FILES["app_threshold"], dpi=160, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=resolution_profile,
        x="resolution_days",
        y="leave_rate",
        marker="o",
        color="#264653",
    )
    plt.axhline(overall_rate, color="#e76f51", linestyle="--", linewidth=1.5)
    plt.title("Leave Risk by Complaint Resolution Days")
    plt.xlabel("Resolution Days")
    plt.ylabel("Leave-Risk Rate")
    plt.tight_layout()
    plt.savefig(CHART_FILES["resolution_threshold"], dpi=160, bbox_inches="tight")
    plt.close()


def plot_relationship_manager_summary(rm_summary):
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=rm_summary,
        x="relationship_manager_status",
        y="retention_rate_proxy",
        hue="relationship_manager_status",
        palette=["#c9d6df", "#2a9d8f"],
        legend=False,
    )
    plt.title("Retention Proxy Rate by Relationship Manager Coverage")
    plt.xlabel("")
    plt.ylabel("Retention Proxy Rate")
    plt.tight_layout()
    plt.savefig(CHART_FILES["relationship_manager"], dpi=160, bbox_inches="tight")
    plt.close()


def export_outputs(
    df,
    dataset_path,
    target_metadata,
    metrics,
    servqual_correlation,
    feature_importance,
    threshold_summary,
    relationship_manager_summary,
):
    strongest_dimension, strongest_importance = strongest_servqual_dimension(
        feature_importance
    )

    summary_metrics = pd.DataFrame(
        [
            {
                "dataset_file": dataset_path.name,
                "rows_loaded": int(len(df)),
                "model_target_column": target_metadata["target_column"],
                "target_strategy": target_metadata["target_strategy"],
                "target_note": target_metadata["target_note"],
                "target_cutoff_value": target_metadata["cutoff_value"],
                "target_positive_rate": target_metadata["positive_rate"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "mean_service_quality_index": round(
                    float(df["service_quality_index"].mean()), 4
                ),
                "mean_predicted_leave_probability": round(
                    float(df["predicted_leave_probability"].mean()), 4
                ),
                "strongest_servqual_dimension": strongest_dimension,
                "strongest_servqual_importance": round(strongest_importance, 4),
            }
        ]
    )

    exported_files = {}

    exported_files["final_output"] = write_csv_with_fallback(
        df, FINAL_OUTPUT_FILE, index=False
    )
    exported_files["metrics"] = write_csv_with_fallback(
        summary_metrics, MODEL_METRICS_FILE, index=False
    )
    exported_files["correlations"] = write_csv_with_fallback(
        servqual_correlation.reset_index().rename(
        columns={"index": "servqual_dimension"}
    ),
        SERVQUAL_CORRELATION_FILE,
        index=False,
    )
    exported_files["feature_importance"] = write_csv_with_fallback(
        feature_importance.reset_index().rename(
        columns={"index": "feature"}
    ),
        FEATURE_IMPORTANCE_FILE,
        index=False,
    )
    exported_files["thresholds"] = write_csv_with_fallback(
        threshold_summary, THRESHOLD_FILE, index=False
    )
    exported_files["relationship_manager"] = write_csv_with_fallback(
        relationship_manager_summary, RELATIONSHIP_MANAGER_FILE, index=False
    )

    high_risk_customers = build_high_risk_customer_extract(df)
    exported_files["high_risk_customers"] = write_csv_with_fallback(
        high_risk_customers, HIGH_RISK_CUSTOMERS_FILE, index=False
    )

    return exported_files


def main():
    sns.set_theme(style="whitegrid", context="talk")
    pd.set_option("display.max_columns", None)

    dataset_path = locate_dataset()
    df = pd.read_csv(dataset_path)
    validate_columns(df)

    print(f"Dataset loaded from: {dataset_path.name}")
    print(f"Shape: {df.shape}")

    df = build_servqual_framework(df)
    target_metadata = build_leave_target(df)
    df = target_metadata["data"]

    print("\nTarget strategy:")
    print(target_metadata["target_note"])

    df, model, feature_importance, metrics = build_model(
        df=df,
        target_column=target_metadata["target_column"],
    )
    df["recommended_action"] = df.apply(recommend_action, axis=1)

    servqual_correlation = plot_heatmap(df)
    plot_sqi_boxplot(df, target_metadata["target_strategy"])
    plot_feature_importance(feature_importance)

    threshold_summary = build_threshold_summary(
        df=df,
        target_column=target_metadata["target_column"],
    )
    plot_threshold_profiles(df, target_metadata["target_column"])

    relationship_manager_summary = build_relationship_manager_summary(
        df=df,
        target_column=target_metadata["target_column"],
    )
    plot_relationship_manager_summary(relationship_manager_summary)

    exported_files = export_outputs(
        df=df,
        dataset_path=dataset_path,
        target_metadata=target_metadata,
        metrics=metrics,
        servqual_correlation=servqual_correlation,
        feature_importance=feature_importance,
        threshold_summary=threshold_summary,
        relationship_manager_summary=relationship_manager_summary,
    )

    strongest_dimension, strongest_importance = strongest_servqual_dimension(
        feature_importance
    )
    print("\nRandom Forest performance:")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")

    print("\nSERVQUAL correlation with retention_score:")
    print(servqual_correlation.to_string())

    print("\nStrongest SERVQUAL predictor:")
    print(
        f"{strongest_dimension} with feature importance {strongest_importance:.4f}"
    )

    print("\nCritical thresholds:")
    print(threshold_summary.to_string(index=False))

    print("\nRelationship manager comparison:")
    print(relationship_manager_summary.to_string(index=False))

    print("\nFiles exported:")
    for exported_path in exported_files.values():
        print(f"- {exported_path.name}")
    for chart_file in CHART_FILES.values():
        print(f"- {chart_file.name}")


if __name__ == "__main__":
    main()
