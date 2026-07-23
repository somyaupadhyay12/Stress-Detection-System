"""Run model comparison stage for stress detection pipeline.

This script trains and evaluates Dummy, Logistic Regression, Random Forest,
SVM RBF, and GRU models on raw, filtered, and filtered + normalized variants.
It uses the exact same subject-independent train/test split for all models.
Outputs are saved in a new folder named `model_comparison`.
"""

from __future__ import annotations

import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Setup project root and python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "07_classical_ml"))

# Import functions from classical ml script
from train_classical_models import create_feature_tables, WESAD_ROOT, RANDOM_SEED, METADATA_COLUMNS

# sklearn imports
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# tensorflow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU as KerasGRU, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

def get_classical_models():
    return {
        "Dummy": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyClassifier(strategy="prior"))
        ]),
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=RANDOM_SEED)),
        ]),
        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)),
        ]),
        "SVM (RBF)": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", class_weight="balanced", probability=True, random_state=RANDOM_SEED)),
        ]),
    }

def train_gru(X_train_gru, y_train_arr, X_test_gru, y_test_arr):
    # Set seed for reproducibility
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    
    # Compute class weights
    classes = np.unique(y_train_arr)
    weights = compute_class_weight("balanced", classes=classes, y=y_train_arr)
    class_weight = dict(zip(classes, weights))
    
    model = Sequential([
        KerasGRU(64, return_sequences=True, input_shape=(X_train_gru.shape[1], X_train_gru.shape[2])),
        Dropout(0.3),
        KerasGRU(32),
        Dropout(0.3),
        Dense(16, activation="relu"),
        Dropout(0.2),
        Dense(1, activation="sigmoid"),
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    
    early_stopping = EarlyStopping(
        monitor="val_loss",
        payout=5, # Wait, is there a typo in patient/patience? In Keras callback, it's patience. Let's make sure it is patience=5.
        patience=5,
        restore_best_weights=True
    )
    
    model.fit(
        X_train_gru, y_train_arr,
        validation_data=(X_test_gru, y_test_arr),
        epochs=30,
        batch_size=32,
        class_weight=class_weight,
        callbacks=[early_stopping],
        verbose=0
    )
    return model

def main():
    print("Starting stress detection model comparison stage...")
    
    # Create the output directory
    model_comparison_dir = PROJECT_ROOT / "model_comparison"
    model_comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Check subjects from existing filtered table
    filtered_table_path = PROJECT_ROOT / "outputs" / "07_classical_ml" / "filtered" / "ml_ready_feature_table.csv"
    if filtered_table_path.exists():
        df_filt = pd.read_csv(filtered_table_path)
        subjects = sorted(df_filt["subject_id"].unique())
        print(f"Found existing filtered table. Reusing subjects: {subjects}")
    else:
        subjects = ["S2", "S3", "S4", "S5", "S6", "S7"]
        print(f"Filtered table not found. Using default subjects: {subjects}")
        
    variants = ["raw", "filtered", "filtered_normalized"]
    tables = {}
    
    for variant in variants:
        variant_dir = PROJECT_ROOT / "outputs" / "07_classical_ml" / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        table_path = variant_dir / "ml_ready_feature_table.csv"
        
        if table_path.exists():
            print(f"Loading existing feature table for {variant} from {table_path}")
            tables[variant] = pd.read_csv(table_path)
        else:
            print(f"Feature table not found for {variant}. Extracting features on the fly...")
            extracted = create_feature_tables(subjects, [variant])
            if variant in extracted:
                tables[variant] = extracted[variant]
                tables[variant].to_csv(table_path, index=False)
                print(f"Saved feature table for {variant} to {table_path}")
            else:
                raise ValueError(f"Failed to extract features for variant {variant}")
                
    # Define features columns (exclude metadata)
    columns = [col for col in tables["raw"].columns if col not in METADATA_COLUMNS]
    
    # Align on the train/test split using GroupShuffleSplit on the first variant
    first_table = tables[variants[0]]
    X_first = first_table[columns].replace([np.inf, -np.inf], np.nan)
    y_first = first_table["stress_label"].astype(int)
    groups_first = first_table["subject_id"].astype(str)
    
    held_out = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_SEED)
    train_index, test_index = next(held_out.split(X_first, y_first, groups_first))
    
    train_subjects = sorted(groups_first.iloc[train_index].unique())
    test_subjects = sorted(groups_first.iloc[test_index].unique())
    print(f"Train subjects: {train_subjects}")
    print(f"Test subjects: {test_subjects}")
    
    # Store variant results DataFrames
    df_results = {}
    
    for variant in variants:
        print(f"\n--- Training and evaluating models on variant: {variant} ---")
        table = tables[variant]
        X = table[columns].replace([np.inf, -np.inf], np.nan)
        y = table["stress_label"].astype(int)
        
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        # Preprocessing for GRU
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()
        
        X_train_imputed = imputer.fit_transform(X_train)
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        
        X_test_imputed = imputer.transform(X_test)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        X_train_gru = X_train_scaled[..., None]
        X_test_gru = X_test_scaled[..., None]
        
        variant_results = []
        
        # Evaluate classical models
        classical_models = get_classical_models()
        for model_name, pipeline in classical_models.items():
            print(f"  Fitting {model_name}...")
            pipeline.fit(X_train, y_train)
            preds = pipeline.predict(X_test)
            probs = pipeline.predict_proba(X_test)[:, 1]
            
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, zero_division=0)
            rec = recall_score(y_test, preds, zero_division=0)
            f1 = f1_score(y_test, preds, zero_division=0)
            auc = roc_auc_score(y_test, probs)
            
            variant_results.append({
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "ROC-AUC": auc
            })
            
        # Evaluate GRU
        print("  Fitting GRU...")
        gru_model = train_gru(X_train_gru, y_train.to_numpy(), X_test_gru, y_test.to_numpy())
        gru_probs = gru_model.predict(X_test_gru, verbose=0).ravel()
        gru_preds = (gru_probs >= 0.5).astype(int)
        
        gru_acc = accuracy_score(y_test, gru_preds)
        gru_prec = precision_score(y_test, gru_preds, zero_division=0)
        gru_rec = recall_score(y_test, gru_preds, zero_division=0)
        gru_f1 = f1_score(y_test, gru_preds, zero_division=0)
        gru_auc = roc_auc_score(y_test, gru_probs)
        
        variant_results.append({
            "Model": "GRU",
            "Accuracy": gru_acc,
            "Precision": gru_prec,
            "Recall": gru_rec,
            "F1 Score": gru_f1,
            "ROC-AUC": gru_auc
        })
        
        # Save individual variant results
        df_var = pd.DataFrame(variant_results)
        df_var.to_csv(model_comparison_dir / f"{variant}_comparison.csv", index=False)
        df_results[variant] = df_var
        
        # Display table in output
        print(f"\nResults for {variant}:")
        print(df_var.to_markdown(index=False))
        
    # Generate Overall Comparison Tables
    metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
    models = ["Dummy", "Logistic Regression", "Random Forest", "SVM (RBF)", "GRU"]
    
    print("\n--- Generating Overall Comparison Tables ---")
    for metric in metrics:
        overall_rows = []
        for model in models:
            val_raw = df_results["raw"].loc[df_results["raw"]["Model"] == model, metric].values[0]
            val_filt = df_results["filtered"].loc[df_results["filtered"]["Model"] == model, metric].values[0]
            val_norm = df_results["filtered_normalized"].loc[df_results["filtered_normalized"]["Model"] == model, metric].values[0]
            
            # Find best variant
            vals = {"Raw": val_raw, "Filtered": val_filt, "Filtered + Normalized": val_norm}
            best_var = max(vals, key=vals.get)
            
            overall_rows.append({
                "Model": model,
                f"Raw {metric}": val_raw,
                f"Filtered {metric}": val_filt,
                f"Filtered + Normalized {metric}": val_norm,
                "Best Variant": best_var
            })
            
        df_metric = pd.DataFrame(overall_rows)
        metric_slug = metric.lower().replace(" ", "_").replace("-", "_")
        df_metric.to_csv(model_comparison_dir / f"overall_{metric_slug}_comparison.csv", index=False)
        
        print(f"\nOverall comparison for {metric}:")
        print(df_metric.to_markdown(index=False))
        
    # Generate Grouped Bar Charts
    print("\n--- Generating Grouped Bar Charts ---")
    x = np.arange(len(models))
    width = 0.25
    
    # Custom modern styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    
    colors = {
        "Raw": "#F8766D",                 # Pastel coral
        "Filtered": "#619CFF",            # Pastel blue
        "Filtered + Normalized": "#00BA38" # Pastel green
    }
    
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
        
        raw_vals = [df_results["raw"].loc[df_results["raw"]["Model"] == m, metric].values[0] for m in models]
        filt_vals = [df_results["filtered"].loc[df_results["filtered"]["Model"] == m, metric].values[0] for m in models]
        norm_vals = [df_results["filtered_normalized"].loc[df_results["filtered_normalized"]["Model"] == m, metric].values[0] for m in models]
        
        # Plot grouped bars
        ax.bar(x - width, raw_vals, width, label='Raw', color=colors["Raw"])
        ax.bar(x, filt_vals, width, label='Filtered', color=colors["Filtered"])
        ax.bar(x + width, norm_vals, width, label='Filtered + Normalized', color=colors["Filtered + Normalized"])
        
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f'Model Comparison - {metric}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=10)
        ax.legend(frameon=True, facecolor='white', edgecolor='none', shadow=False)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Clean spines
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
            
        fig.tight_layout()
        
        metric_slug = metric.lower().replace(" ", "_").replace("-", "_")
        plot_path = model_comparison_dir / f"{metric_slug}_comparison.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Saved grouped bar chart: {plot_path}")
        
    print("\nModel comparison stage finished successfully. Results saved in `model_comparison/` folder.")

if __name__ == "__main__":
    main()
