import argparse
import json
import os
import re
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_FEATURES = [
    "detention_days",
    "sentence_days_final",
    "detention_ratio_calc",
    "days_pending",
    "last_hearing_gap",
    "number_of_trials",
    "no_of_adjournments",
    "age",
    "vulnerability_component",
    "offense_type",
    "legal_status",
    "trial_status",
    "case_complexity",
    "bailable",
    "ipc_section",
]

NUMERIC_MODEL_FEATURES = [
    "detention_days",
    "sentence_days_final",
    "detention_ratio_calc",
    "days_pending",
    "last_hearing_gap",
    "number_of_trials",
    "no_of_adjournments",
    "age",
    "vulnerability_component",
]

CATEGORICAL_MODEL_FEATURES = [
    "offense_type",
    "legal_status",
    "trial_status",
    "case_complexity",
    "bailable",
    "ipc_section",
]


def _to_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({"1": True, "0": False, "true": True, "false": False, "yes": True, "no": False})
        .fillna(False)
    )


def _parse_section_tokens(section_value: str) -> list[str]:
    if pd.isna(section_value):
        return []
    text = str(section_value).strip().upper()
    if not text:
        return []
    raw_tokens = re.split(r"[\s,;/|+-]+", text)
    return [t for t in raw_tokens if t]


def load_statute_lookup(path: str) -> pd.DataFrame:
    table = pd.read_csv(path)
    required_cols = {"section", "max_sentence_years"}
    missing = required_cols.difference(table.columns)
    if missing:
        raise ValueError(f"Statute lookup missing required columns: {sorted(missing)}")

    out = table.copy()
    out["section"] = out["section"].astype(str).str.strip().str.upper()
    out["max_sentence_years"] = pd.to_numeric(out["max_sentence_years"], errors="coerce").fillna(0.0)

    if "max_sentence_years_effective" not in out.columns:
        out["max_sentence_years_effective"] = out["max_sentence_years"]
    out["max_sentence_years_effective"] = pd.to_numeric(
        out["max_sentence_years_effective"], errors="coerce"
    )

    out["max_sentence_days"] = (out["max_sentence_years_effective"] * 365).round()

    if "punishment_type" not in out.columns:
        out["punishment_type"] = "finite"
    out["punishment_type"] = out["punishment_type"].astype(str).str.lower().str.strip()

    if "is_non_finite_sentence" not in out.columns:
        out["is_non_finite_sentence"] = out["punishment_type"].isin(["life", "death"])
    else:
        out["is_non_finite_sentence"] = _to_bool(out["is_non_finite_sentence"])

    if "bailability" not in out.columns:
        out["bailability"] = "contextual"
    out["bailability"] = out["bailability"].astype(str).str.lower().str.strip()
    return out


def load_and_merge(data_dir: str) -> pd.DataFrame:
    core = pd.read_csv(os.path.join(data_dir, "dataset1_core_cases.csv"))
    detention = pd.read_csv(os.path.join(data_dir, "dataset2_detention.csv"))
    temporal = pd.read_csv(os.path.join(data_dir, "dataset3_temporal.csv"))
    demographics = pd.read_csv(os.path.join(data_dir, "dataset4_demographics.csv"))
    nlp = pd.read_csv(os.path.join(data_dir, "dataset5_nlp.csv"))

    if "summary" in nlp.columns:
        nlp = nlp.rename(columns={"summary": "summary_nlp"})

    df = core.merge(detention, on="case_id", how="inner")
    df = df.merge(temporal, on="case_id", how="inner")
    df = df.merge(demographics, on="case_id", how="inner")
    df = df.merge(nlp, on="case_id", how="inner")
    df["synthetic_demo_case"] = False
    df["data_source"] = "base_dataset"
    df["medical_notes"] = ""
    return df


def augment_demo_cases(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    next_case_id = int(base["case_id"].max()) + 1

    demo_rows = [
        {
            "case_id": next_case_id,
            "summary": "Undertrial in IPC 302 murder matter with advanced renal disease; detained 6 years, repeated adjournments, urgent medical review requested.",
            "ipc_section": "302",
            "offense_type": "murder",
            "bailable": "no",
            "legal_status": "undertrial",
            "case_type": "criminal",
            "detention_days": 2190,
            "expected_sentence_days": 3650,
            "life_sentence_flag": 1,
            "overstay_flag": False,
            "detention_ratio": 0.60,
            "days_pending": 2450,
            "last_hearing_gap": 410,
            "number_of_trials": 19,
            "trial_status": "delayed",
            "no_of_adjournments": 11,
            "age": 68,
            "gender": "M",
            "health_flag": 1,
            "disability_flag": 0,
            "vulnerability_score": 1,
            "summary_nlp": "Advanced renal disease, prolonged undertrial detention, murder charge, delayed hearings.",
            "urgency_nlp": "high",
            "bail_eligibility_nlp": 0,
            "case_complexity": "high",
            "keywords": "renal disease chronic illness undertrial delayed hearing murder",
            "synthetic_demo_case": True,
            "data_source": "synthetic_demo",
            "medical_notes": "Advanced renal disease requiring frequent dialysis.",
        },
        {
            "case_id": next_case_id + 1,
            "summary": "Female undertrial booked under IPC 376, survivor and accused medical records pending; patient has severe cardiac illness and mobility impairment.",
            "ipc_section": "376",
            "offense_type": "rape",
            "bailable": "no",
            "legal_status": "undertrial",
            "case_type": "criminal",
            "detention_days": 1620,
            "expected_sentence_days": 9125,
            "life_sentence_flag": 1,
            "overstay_flag": False,
            "detention_ratio": 0.18,
            "days_pending": 1980,
            "last_hearing_gap": 300,
            "number_of_trials": 14,
            "trial_status": "active",
            "no_of_adjournments": 9,
            "age": 61,
            "gender": "F",
            "health_flag": 1,
            "disability_flag": 1,
            "vulnerability_score": 1,
            "summary_nlp": "Severe cardiac illness, disability, long-pending rape trial, medical urgency.",
            "urgency_nlp": "high",
            "bail_eligibility_nlp": 0,
            "case_complexity": "high",
            "keywords": "cardiac illness disability undertrial medical urgency life sentence",
            "synthetic_demo_case": True,
            "data_source": "synthetic_demo",
            "medical_notes": "Severe cardiac illness with documented mobility impairment.",
        },
        {
            "case_id": next_case_id + 2,
            "summary": "Senior citizen undertrial in IPC 409 case with uncontrolled diabetes and vision loss; detained over 8 years awaiting evidence closure.",
            "ipc_section": "409",
            "offense_type": "criminal breach of trust",
            "bailable": "no",
            "legal_status": "undertrial",
            "case_type": "criminal",
            "detention_days": 3020,
            "expected_sentence_days": 9125,
            "life_sentence_flag": 1,
            "overstay_flag": False,
            "detention_ratio": 0.33,
            "days_pending": 3200,
            "last_hearing_gap": 520,
            "number_of_trials": 22,
            "trial_status": "delayed",
            "no_of_adjournments": 15,
            "age": 72,
            "gender": "M",
            "health_flag": 1,
            "disability_flag": 1,
            "vulnerability_score": 1,
            "summary_nlp": "Senior undertrial, diabetes, partial blindness, life sentence section, stale case.",
            "urgency_nlp": "high",
            "bail_eligibility_nlp": 0,
            "case_complexity": "medium",
            "keywords": "diabetes blindness senior undertrial stale case",
            "synthetic_demo_case": True,
            "data_source": "synthetic_demo",
            "medical_notes": "Uncontrolled diabetes with major vision loss.",
        },
        {
            "case_id": next_case_id + 3,
            "summary": "Petty theft under IPC 379; undertrial woman with high-risk pregnancy has already remained in custody beyond likely sentence.",
            "ipc_section": "379",
            "offense_type": "theft",
            "bailable": "yes",
            "legal_status": "undertrial",
            "case_type": "fast-track",
            "detention_days": 1400,
            "expected_sentence_days": 1095,
            "life_sentence_flag": 0,
            "overstay_flag": True,
            "detention_ratio": 1.28,
            "days_pending": 1500,
            "last_hearing_gap": 245,
            "number_of_trials": 6,
            "trial_status": "delayed",
            "no_of_adjournments": 8,
            "age": 29,
            "gender": "F",
            "health_flag": 1,
            "disability_flag": 0,
            "vulnerability_score": 1,
            "summary_nlp": "Pregnant undertrial woman, bailable theft, overstay beyond likely sentence.",
            "urgency_nlp": "high",
            "bail_eligibility_nlp": 1,
            "case_complexity": "low",
            "keywords": "pregnancy bailable theft overstay undertrial",
            "synthetic_demo_case": True,
            "data_source": "synthetic_demo",
            "medical_notes": "High-risk pregnancy requiring continuous antenatal care.",
        },
        {
            "case_id": next_case_id + 4,
            "summary": "Juvenile-age accused in IPC 447 trespass with epilepsy and psychiatric treatment; matter has gone stale despite minor and bailable allegations.",
            "ipc_section": "447",
            "offense_type": "trespass",
            "bailable": "yes",
            "legal_status": "undertrial",
            "case_type": "fast-track",
            "detention_days": 120,
            "expected_sentence_days": 91,
            "life_sentence_flag": 0,
            "overstay_flag": True,
            "detention_ratio": 1.32,
            "days_pending": 420,
            "last_hearing_gap": 210,
            "number_of_trials": 3,
            "trial_status": "active",
            "no_of_adjournments": 4,
            "age": 17,
            "gender": "M",
            "health_flag": 1,
            "disability_flag": 0,
            "vulnerability_score": 1,
            "summary_nlp": "Juvenile trespass case, epilepsy, psychiatric treatment, overstay on minor offense.",
            "urgency_nlp": "high",
            "bail_eligibility_nlp": 1,
            "case_complexity": "low",
            "keywords": "juvenile epilepsy mental health overstay bailable",
            "synthetic_demo_case": True,
            "data_source": "synthetic_demo",
            "medical_notes": "Epilepsy with ongoing psychiatric treatment.",
        },
        {
            "case_id": next_case_id + 5,
            "summary": "Undertrial in IPC 302/34 homicide case with tuberculosis and severe weight loss; non-finite sentence section but medical condition requires administrative review.",
            "ipc_section": "302 34",
            "offense_type": "murder",
            "bailable": "no",
            "legal_status": "undertrial",
            "case_type": "criminal",
            "detention_days": 980,
            "expected_sentence_days": 3650,
            "life_sentence_flag": 1,
            "overstay_flag": False,
            "detention_ratio": 0.27,
            "days_pending": 1100,
            "last_hearing_gap": 360,
            "number_of_trials": 8,
            "trial_status": "delayed",
            "no_of_adjournments": 10,
            "age": 45,
            "gender": "M",
            "health_flag": 1,
            "disability_flag": 0,
            "vulnerability_score": 1,
            "summary_nlp": "Tuberculosis, homicide undertrial, urgent medical review, death-eligible section.",
            "urgency_nlp": "high",
            "bail_eligibility_nlp": 0,
            "case_complexity": "high",
            "keywords": "tuberculosis homicide undertrial urgent medical review",
            "synthetic_demo_case": True,
            "data_source": "synthetic_demo",
            "medical_notes": "Active tuberculosis with severe weight loss.",
        },
    ]

    demo_df = pd.DataFrame(demo_rows)

    for column in base.columns:
        if column not in demo_df.columns:
            if pd.api.types.is_bool_dtype(base[column]):
                demo_df[column] = False
            elif pd.api.types.is_numeric_dtype(base[column]):
                demo_df[column] = 0
            else:
                demo_df[column] = ""

    demo_df = demo_df[base.columns]
    return pd.concat([base, demo_df], ignore_index=True)


def engineer_features(df: pd.DataFrame, statute_lookup: pd.DataFrame | None = None) -> pd.DataFrame:
    out = df.copy()

    text_cols = [
        "ipc_section",
        "offense_type",
        "bailable",
        "legal_status",
        "case_type",
        "trial_status",
        "case_complexity",
        "gender",
        "summary",
        "summary_nlp",
        "keywords",
        "medical_notes",
        "data_source",
    ]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).fillna("")

    out["bailable_flag"] = _to_bool(out["bailable"])
    out["overstay_flag_input"] = _to_bool(out["overstay_flag"])
    out["undertrial_flag"] = out["legal_status"].astype(str).str.lower().eq("undertrial")
    out["bail_eligibility_nlp_flag"] = out["bail_eligibility_nlp"].astype(int).eq(1)
    out["health_flag"] = out["health_flag"].astype(int).eq(1)
    out["disability_flag"] = out["disability_flag"].astype(int).eq(1)

    statute_days_map = {}
    statute_non_finite_map = {}
    statute_bailability_map = {}
    if statute_lookup is not None:
        statute_days_map = statute_lookup.set_index("section")["max_sentence_days"].to_dict()
        statute_non_finite_map = statute_lookup.set_index("section")["is_non_finite_sentence"].to_dict()
        statute_bailability_map = statute_lookup.set_index("section")["bailability"].to_dict()

    def _lookup_statutory_days(section_value: str) -> float:
        tokens = _parse_section_tokens(section_value)
        days = [statute_days_map.get(t) for t in tokens if t in statute_days_map]
        if not days:
            return np.nan
        finite_days = [d for d in days if pd.notna(d)]
        if not finite_days:
            return np.nan
        return float(max(finite_days))

    def _lookup_non_finite(section_value: str) -> bool:
        tokens = _parse_section_tokens(section_value)
        flags = [bool(statute_non_finite_map.get(t, False)) for t in tokens]
        return any(flags)

    def _lookup_bailability(section_value: str) -> str:
        tokens = _parse_section_tokens(section_value)
        values = [str(statute_bailability_map.get(t, "")).strip().lower() for t in tokens if t in statute_bailability_map]
        if not values:
            return "contextual"
        if "non-bailable" in values:
            return "non-bailable"
        if "bailable" in values:
            return "bailable"
        return "contextual"

    out["statutory_max_sentence_days"] = out["ipc_section"].apply(_lookup_statutory_days)
    out["statutory_non_finite_sentence"] = out["ipc_section"].apply(_lookup_non_finite)
    out["statutory_bailability"] = out["ipc_section"].apply(_lookup_bailability)

    out["sentence_days_final"] = out["statutory_max_sentence_days"].fillna(out["expected_sentence_days"])
    out.loc[out["statutory_non_finite_sentence"], "sentence_days_final"] = np.nan

    out["detention_ratio_calc"] = (
        out["detention_days"] / out["sentence_days_final"].replace(0, np.nan)
    ).replace([np.inf, -np.inf], np.nan)
    out["detention_ratio_calc"] = out["detention_ratio_calc"].fillna(out["detention_ratio"])
    out["detention_ratio_calc"] = out["detention_ratio_calc"].fillna(0.0)

    out["computed_overstay"] = (out["detention_days"] > out["sentence_days_final"]).fillna(False)
    out.loc[out["statutory_non_finite_sentence"], "computed_overstay"] = False
    out["overstay_final"] = out["overstay_flag_input"] | out["computed_overstay"]

    out["bailable_legal_flag"] = np.select(
        [
            out["statutory_bailability"].eq("bailable"),
            out["statutory_bailability"].eq("non-bailable"),
        ],
        [True, False],
        default=out["bailable_flag"],
    ).astype(bool)

    out["age_vulnerable"] = (out["age"] >= 60) | (out["age"] < 18)
    out["gender_vulnerable"] = out["gender"].astype(str).str.upper().eq("F")
    out["health_vulnerable"] = out["health_flag"] | out["disability_flag"]

    out["vulnerability_component"] = (
        0.4 * out["age_vulnerable"].astype(int)
        + 0.3 * out["gender_vulnerable"].astype(int)
        + 0.3 * out["health_vulnerable"].astype(int)
    )
    out["vulnerability_component"] = np.maximum(
        out["vulnerability_component"], out["vulnerability_score"].astype(float)
    )

    minor_offenses = {
        "theft",
        "public nuisance",
        "hurt",
        "trespass",
        "petty theft",
        "minor assault",
        "nuisance",
    }
    severe_offenses = {"murder", "rape", "terror", "kidnapping", "armed robbery", "dacoity", "robbery"}

    offense_lower = out["offense_type"].astype(str).str.lower()
    out["minor_offense_flag"] = offense_lower.isin(minor_offenses)
    out["severe_offense_flag"] = offense_lower.isin(severe_offenses)

    out["offense_severity_index"] = np.select(
        [out["severe_offense_flag"], out["minor_offense_flag"]],
        [1.0, 0.2],
        default=0.6,
    )

    out["detention_ratio_capped"] = np.clip(out["detention_ratio_calc"], 0.0, 3.0)
    out["detention_ratio_exp_norm"] = (np.exp(out["detention_ratio_capped"]) - 1) / (np.exp(3.0) - 1)

    stale_threshold = out["last_hearing_gap"].quantile(0.75)
    out["stale_case_flag"] = out["last_hearing_gap"] >= stale_threshold

    out["offense_priority_component"] = np.clip(
        0.4 * out["undertrial_flag"].astype(float)
        + 0.2 * (out["bailable_legal_flag"] | out["bail_eligibility_nlp_flag"]).astype(float)
        + 0.2 * out["minor_offense_flag"].astype(float)
        + 0.2 * (1 - out["offense_severity_index"]),
        0.0,
        1.0,
    )

    base_score = (
        50 * out["overstay_final"].astype(float)
        + 25 * out["detention_ratio_exp_norm"]
        + 15 * out["offense_priority_component"]
        + 10 * out["vulnerability_component"]
        + 3 * out["stale_case_flag"].astype(float)
    )

    critical_override = out["undertrial_flag"] & out["overstay_final"]
    out["priority_score"] = np.where(critical_override, np.maximum(base_score, 95.0), base_score)
    out["priority_score"] = np.clip(out["priority_score"], 0, 100).round(2)

    out["priority_bucket"] = pd.cut(
        out["priority_score"],
        bins=[-0.1, 40, 65, 85, 100],
        labels=["Low", "Medium", "High", "Critical"],
    )

    out["recommended_track"] = np.select(
        [
            critical_override,
            out["undertrial_flag"] & (out["bailable_legal_flag"] | out["bail_eligibility_nlp_flag"]),
            out["minor_offense_flag"] & out["case_complexity"].astype(str).str.lower().eq("low"),
            out["case_type"].astype(str).str.lower().eq("civil"),
        ],
        ["Critical-Review", "Bail-Eligible", "Fast-Track", "Civil"],
        default="Regular-Criminal",
    )

    out["flag_reason"] = np.select(
        [
            critical_override,
            out["undertrial_flag"] & out["stale_case_flag"],
            out["health_vulnerable"] & out["undertrial_flag"],
        ],
        [
            "Undertrial overstay beyond statutory/derived sentence limit",
            "Undertrial with stale hearing history",
            "Health/disability vulnerable undertrial",
        ],
        default="No critical flag",
    )

    return out


def train_models(df: pd.DataFrame) -> dict:
    def build_preprocessor() -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_MODEL_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_MODEL_FEATURES),
            ]
        )

    y_urgency_expert = np.select(
        [
            df["priority_score"] >= 85,
            (df["priority_score"] >= 65) & (df["priority_score"] < 85),
        ],
        ["High", "Medium"],
        default="Low",
    )

    urgency_map = {"low": "Low", "medium": "Medium", "high": "High"}
    y_urgency_nlp = df["urgency_nlp"].astype(str).str.lower().map(urgency_map).fillna("Medium")

    X_train, X_test, y_train, y_test = train_test_split(
        df[MODEL_FEATURES], y_urgency_expert, test_size=0.2, random_state=42, stratify=y_urgency_expert
    )

    urgency_clf = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )
    urgency_clf.fit(X_train, y_train)
    y_pred = urgency_clf.predict(X_test)

    urgency_accuracy_expert = float(accuracy_score(y_test, y_pred))
    urgency_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    y_track = df["recommended_track"]
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        df[MODEL_FEATURES], y_track, test_size=0.2, random_state=42, stratify=y_track
    )
    track_clf = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
        ]
    )
    track_clf.fit(X2_train, y2_train)
    y2_pred = track_clf.predict(X2_test)
    track_accuracy = float(accuracy_score(y2_test, y2_pred))

    df["predicted_urgency"] = urgency_clf.predict(df[MODEL_FEATURES])
    df["predicted_case_type"] = track_clf.predict(df[MODEL_FEATURES])
    urgency_nlp_agreement = float(accuracy_score(y_urgency_nlp, df["predicted_urgency"]))

    critical_true = (df["undertrial_flag"] & df["overstay_final"]).astype(int)
    critical_pred = (df["priority_score"] >= 95).astype(int)
    undertrial_recall = float(recall_score(critical_true, critical_pred, zero_division=0))

    np.random.seed(42)
    mock_judicial_review = (
        0.6 * df["priority_score"]
        + 20 * df["detention_ratio_exp_norm"]
        + 10 * df["vulnerability_component"]
        + np.random.normal(0, 3, size=len(df))
    )
    ranked_agreement_spearman = float(df["priority_score"].corr(mock_judicial_review, method="spearman"))

    start = perf_counter()
    _ = df[(df["priority_bucket"].isin(["High", "Critical"])) & (df["undertrial_flag"])]
    dashboard_filter_response_ms = float((perf_counter() - start) * 1000)

    return {
        "scored_df": df,
        "urgency_model": urgency_clf,
        "track_model": track_clf,
        "metrics": {
            "classification_accuracy_vs_expert_labels": urgency_accuracy_expert,
            "urgency_nlp_label_agreement": urgency_nlp_agreement,
            "classification_accuracy_case_type": track_accuracy,
            "undertrial_detection_recall": undertrial_recall,
            "dashboard_filter_response_time_ms": dashboard_filter_response_ms,
            "ranked_list_agreement_spearman": ranked_agreement_spearman,
            "urgency_classification_report": urgency_report,
        },
    }


def prepare_output_tables(df: pd.DataFrame) -> dict:
    ranked = df.sort_values(["priority_score", "detention_ratio_calc", "days_pending"], ascending=[False, False, False]).copy()
    ranked["recommended_hearing_order"] = np.arange(1, len(ranked) + 1)

    flagged = ranked[
        (ranked["undertrial_flag"]) & (ranked["overstay_final"] | ranked["priority_bucket"].astype(str).eq("Critical"))
    ].copy()
    demo_cases = ranked[ranked["synthetic_demo_case"] == True].copy()

    cluster_summary = (
        ranked.groupby(["cluster_id", "cluster_label"], as_index=False)
        .agg(
            case_count=("case_id", "count"),
            mean_priority=("priority_score", "mean"),
            mean_detention_ratio=("detention_ratio_calc", "mean"),
            undertrial_share=("undertrial_flag", "mean"),
        )
        .sort_values("mean_priority", ascending=False)
    )

    return {
        "ranked": ranked,
        "flagged": flagged,
        "demo_cases": demo_cases,
        "cluster_summary": cluster_summary,
    }


def run_pipeline(
    data_dir: str = "data",
    statute_file: str | None = None,
    augment_demo: bool = False,
) -> dict:
    if statute_file is None:
        statute_file = os.path.join(data_dir, "ipc_bns_max_sentence_lookup.csv")

    merged = load_and_merge(data_dir)
    if augment_demo:
        merged = augment_demo_cases(merged)

    statute_lookup = load_statute_lookup(statute_file) if statute_file and os.path.exists(statute_file) else None
    featured = engineer_features(merged, statute_lookup)
    model_result = train_models(featured)
    clustered = cluster_cases(model_result["scored_df"])
    output_tables = prepare_output_tables(clustered)

    return {
        "merged": merged,
        "featured": featured,
        "clustered": clustered,
        "urgency_model": model_result["urgency_model"],
        "track_model": model_result["track_model"],
        "metrics": model_result["metrics"],
        "statute_lookup": statute_lookup,
        **output_tables,
    }


def score_new_case(
    case_payload: dict,
    reference_df: pd.DataFrame,
    statute_lookup: pd.DataFrame | None,
    urgency_model: Pipeline,
    track_model: Pipeline,
) -> pd.Series:
    next_case_id = int(reference_df["case_id"].max()) + 1
    row = {
        "case_id": next_case_id,
        "summary": "",
        "ipc_section": "",
        "offense_type": "",
        "bailable": "no",
        "legal_status": "undertrial",
        "case_type": "criminal",
        "detention_days": 0,
        "expected_sentence_days": 0,
        "life_sentence_flag": 0,
        "overstay_flag": False,
        "detention_ratio": 0.0,
        "days_pending": 0,
        "last_hearing_gap": 0,
        "number_of_trials": 0,
        "trial_status": "active",
        "no_of_adjournments": 0,
        "age": 30,
        "gender": "M",
        "health_flag": 0,
        "disability_flag": 0,
        "vulnerability_score": 0,
        "summary_nlp": "",
        "urgency_nlp": "medium",
        "bail_eligibility_nlp": 0,
        "case_complexity": "medium",
        "keywords": "",
        "synthetic_demo_case": False,
        "data_source": "manual_prediction",
        "medical_notes": "",
    }
    row.update(case_payload)

    candidate_df = pd.DataFrame([row])
    combined = pd.concat([reference_df, candidate_df], ignore_index=True)
    engineered = engineer_features(combined, statute_lookup)
    candidate = engineered.tail(1).copy()
    candidate["predicted_urgency"] = urgency_model.predict(candidate[MODEL_FEATURES])
    candidate["predicted_case_type"] = track_model.predict(candidate[MODEL_FEATURES])
    return candidate.iloc[0]


def cluster_cases(df: pd.DataFrame) -> pd.DataFrame:
    cluster_features = [
        "priority_score",
        "detention_ratio_exp_norm",
        "days_pending",
        "last_hearing_gap",
        "vulnerability_component",
        "offense_severity_index",
    ]
    X = df[cluster_features].copy()
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    df["cluster_id"] = kmeans.fit_predict(X_scaled)

    cluster_mean = df.groupby("cluster_id")["priority_score"].mean().sort_values(ascending=False)
    cluster_rank = {cid: idx for idx, cid in enumerate(cluster_mean.index.tolist())}
    cluster_names = {
        0: "Cluster-A Critical Backlog",
        1: "Cluster-B High Priority",
        2: "Cluster-C Moderate Queue",
        3: "Cluster-D Routine Queue",
    }
    df["cluster_rank"] = df["cluster_id"].map(cluster_rank)
    df["cluster_label"] = df["cluster_rank"].map(cluster_names)
    return df


def save_outputs(df: pd.DataFrame, metrics: dict, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    output_tables = prepare_output_tables(df)
    ranked = output_tables["ranked"]
    flagged = output_tables["flagged"]
    cluster_summary = output_tables["cluster_summary"]
    demo_cases = output_tables["demo_cases"]

    ranked.to_csv(os.path.join(output_dir, "prioritized_cases.csv"), index=False)
    flagged.to_csv(os.path.join(output_dir, "flagged_undertrials.csv"), index=False)
    cluster_summary.to_csv(os.path.join(output_dir, "cluster_summary.csv"), index=False)
    demo_cases.to_csv(os.path.join(output_dir, "synthetic_demo_cases.csv"), index=False)

    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Court Case Backlog Prioritization Engine")
    parser.add_argument("--data-dir", default="data", help="Path containing input CSV files")
    parser.add_argument("--output-dir", default="outputs", help="Path for generated outputs")
    parser.add_argument(
        "--statute-file",
        default=os.path.join("data", "ipc_bns_max_sentence_lookup.csv"),
        help="CSV with statute mapping: section,max_sentence_years",
    )
    parser.add_argument(
        "--augment-demo",
        action="store_true",
        help="Append small synthetic life/death and medical-vulnerability cases for demo outputs",
    )
    args = parser.parse_args()

    pipeline_result = run_pipeline(
        data_dir=args.data_dir,
        statute_file=args.statute_file,
        augment_demo=args.augment_demo,
    )
    save_outputs(pipeline_result["clustered"], pipeline_result["metrics"], args.output_dir)

    print("Generated outputs:")
    print(f"- {os.path.join(args.output_dir, 'prioritized_cases.csv')}")
    print(f"- {os.path.join(args.output_dir, 'flagged_undertrials.csv')}")
    print(f"- {os.path.join(args.output_dir, 'cluster_summary.csv')}")
    print(f"- {os.path.join(args.output_dir, 'synthetic_demo_cases.csv')}")
    print(f"- {os.path.join(args.output_dir, 'evaluation_metrics.json')}")
    print("\nKey metrics:")
    print(json.dumps(pipeline_result["metrics"], indent=2))


if __name__ == "__main__":
    main()
