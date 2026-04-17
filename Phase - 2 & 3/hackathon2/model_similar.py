"""
SIMILAR MODEL - Recreates the original prioritization model architecture
with identical algorithms and approach but enhanced documentation and modularity.

Uses:
- RandomForest Classification for urgency and case type
- KMeans Clustering with 4 clusters
- Same feature engineering pipeline
"""

import os
import json
from time import perf_counter
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

# ==================== FEATURE DEFINITIONS ====================
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

MODEL_FEATURES = NUMERIC_MODEL_FEATURES + CATEGORICAL_MODEL_FEATURES

CLUSTER_FEATURES = [
    "priority_score",
    "detention_ratio_exp_norm",
    "days_pending",
    "last_hearing_gap",
    "vulnerability_component",
    "offense_severity_index",
]

class SimilarModel:
    """Similar model - RF + KMeans approach"""
    
    def __init__(self):
        self.urgency_model = None
        self.track_model = None
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
        self.metrics = {}
        
    def build_preprocessor(self):
        """Build preprocessing pipeline"""
        return ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), NUMERIC_MODEL_FEATURES),
                ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_MODEL_FEATURES),
            ]
        )

    def train(self, df: pd.DataFrame) -> dict:
        """Train urgency and track classification models"""
        
        # Urgency Classification Target (based on priority score)
        y_urgency = np.select(
            [
                df["priority_score"] >= 85,
                (df["priority_score"] >= 65) & (df["priority_score"] < 85),
            ],
            ["High", "Medium"],
            default="Low",
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df[MODEL_FEATURES], y_urgency, test_size=0.2, random_state=42, stratify=y_urgency
        )
        
        # Train urgency classifier
        self.urgency_model = Pipeline(
            steps=[
                ("preprocessor", self.build_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ]
        )
        self.urgency_model.fit(X_train, y_train)
        y_pred_urgency = self.urgency_model.predict(X_test)
        
        urgency_accuracy = float(accuracy_score(y_test, y_pred_urgency))
        urgency_f1 = float(f1_score(y_test, y_pred_urgency, average='weighted'))
        urgency_report = classification_report(y_test, y_pred_urgency, output_dict=True, zero_division=0)
        
        # Track Classification Target
        y_track = df["recommended_track"]
        X2_train, X2_test, y2_train, y2_test = train_test_split(
            df[MODEL_FEATURES], y_track, test_size=0.2, random_state=42, stratify=y_track
        )
        
        # Train track classifier
        self.track_model = Pipeline(
            steps=[
                ("preprocessor", self.build_preprocessor()),
                ("model", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
            ]
        )
        self.track_model.fit(X2_train, y2_train)
        y_pred_track = self.track_model.predict(X2_test)
        
        track_accuracy = float(accuracy_score(y2_test, y_pred_track))
        track_f1 = float(f1_score(y2_test, y_pred_track, average='weighted'))
        
        # Calculate predictions on full dataset
        df["predicted_urgency"] = self.urgency_model.predict(df[MODEL_FEATURES])
        df["predicted_case_type"] = self.track_model.predict(df[MODEL_FEATURES])
        
        # Additional metrics
        critical_true = (df["undertrial_flag"] & df["overstay_final"]).astype(int)
        critical_pred = (df["priority_score"] >= 95).astype(int)
        undertrial_recall = float(recall_score(critical_true, critical_pred, zero_division=0))
        
        self.metrics = {
            "model_type": "Similar (RandomForest + KMeans)",
            "urgency_classification_accuracy": urgency_accuracy,
            "urgency_classification_f1": urgency_f1,
            "track_classification_accuracy": track_accuracy,
            "track_classification_f1": track_f1,
            "undertrial_detection_recall": undertrial_recall,
            "urgency_classification_report": urgency_report,
        }
        
        return df
    
    def cluster(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform KMeans clustering"""
        X = df[CLUSTER_FEATURES].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        df["cluster_id"] = self.kmeans.fit_predict(X_scaled)
        
        # Map clusters to priority-based names
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
    
    def save_models(self, output_dir: str):
        """Save trained models"""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.urgency_model, os.path.join(output_dir, "urgency_model.pkl"))
        joblib.dump(self.track_model, os.path.join(output_dir, "track_model.pkl"))
        joblib.dump(self.kmeans, os.path.join(output_dir, "kmeans_model.pkl"))
        joblib.dump(self.scaler, os.path.join(output_dir, "scaler.pkl"))
        json.dump(self.metrics, open(os.path.join(output_dir, "metrics_similar.json"), "w"), indent=2)


if __name__ == "__main__":
    print("Similar Model module ready for import")
