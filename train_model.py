import os
import sys
import csv
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


def load_dataset(csv_path):
    features = []
    labels = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # skip header
        
        for row in reader:
            feat = [float(x) for x in row[:-1]]
            label = row[-1]
            features.append(feat)
            labels.append(label)
    
    return np.array(features), np.array(labels), header[:-1]


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "dataset_keypoints.csv")
    
    if not os.path.exists(csv_path):
        print("Error: dataset_keypoints.csv not found!")
        print("Run extract_keypoints.py first.")
        sys.exit(1)
    
    print("Loading dataset...")
    X, y, feature_names = load_dataset(csv_path)
    print(f"  Samples: {len(X)}")
    print(f"  Features per sample: {X.shape[1]}")
    print(f"  Classes: {np.unique(y)}")
    print(f"  Class distribution:")
    for cls in np.unique(y):
        count = np.sum(y == cls)
        print(f"    {cls}: {count} samples ({100*count/len(y):.1f}%)")
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        ))
    ])
    
    n_classes = len(np.unique(y_encoded))
    cv_scores = None
    report = None
    
    if n_classes >= 2:
        print("\nRunning 5-fold cross-validation...")
        cv_folds = min(5, n_classes)
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=cv_folds, scoring='accuracy')
        print(f"  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        if len(X) > 20:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )
            
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            
            report = classification_report(
                y_test, y_pred, 
                target_names=label_encoder.classes_,
                zero_division=0
            )
            print("\nClassification Report (on 20% test set):")
            print(report)
            
            cm = confusion_matrix(y_test, y_pred)
            print("Confusion Matrix:")
            print(cm)
    else:
        print(f"\n  Single class detected ('{np.unique(y)[0]}').")
        print("  Skipping cross-validation — model will learn reference angles for correction.")
    
    print("\nTraining final model on full dataset...")
    pipeline.fit(X, y_encoded)
    
    importances = pipeline.named_steps['classifier'].feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    print("\nTop 15 Most Important Features:")
    for idx in top_indices:
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    model_path = os.path.join(base_dir, "pose_classifier.joblib")
    encoder_path = os.path.join(base_dir, "label_encoder.joblib")
    
    joblib.dump(pipeline, model_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\n✅ Model saved to {model_path}")
    print(f"✅ Label encoder saved to {encoder_path}")
    
    report_path = os.path.join(base_dir, "training_report.txt")
    with open(report_path, 'w') as f:
        f.write("Yoga Pose Classifier — Training Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Features per sample: {X.shape[1]}\n")
        f.write(f"Classes: {list(label_encoder.classes_)}\n\n")
        if cv_scores is not None:
            f.write(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n\n")
        else:
            f.write("CV: Skipped (single class)\n\n")
        if report:
            f.write("Classification Report (20% test):\n")
            f.write(report + "\n")
        f.write("\nTop 15 Features:\n")
        for idx in top_indices:
            f.write(f"  {feature_names[idx]}: {importances[idx]:.4f}\n")
    
    print(f"✅ Report saved to {report_path}")
    
    print("\nComputing reference angles for pose correction...")
    compute_reference_angles(X, y, feature_names, label_encoder, base_dir)


def compute_reference_angles(X, y, feature_names, label_encoder, base_dir):
    angle_features = [f for f in feature_names if 'angle' in f or 'inclination' in f]
    angle_indices = [feature_names.index(f) for f in angle_features]
    
    reference = {}
    
    for cls in np.unique(y):
        mask = y == cls
        cls_data = X[mask][:, angle_indices]
        
        ref = {}
        for i, angle_name in enumerate(angle_features):
            values = cls_data[:, i]
            valid_values = values[values > 0]
            if len(valid_values) > 0:
                ref[angle_name] = {
                    "mean": float(np.mean(valid_values)),
                    "std": float(np.std(valid_values)),
                    "min": float(np.percentile(valid_values, 10)),
                    "max": float(np.percentile(valid_values, 90))
                }
        reference[cls] = ref
    
    import json
    ref_path = os.path.join(base_dir, "pose_references.json")
    with open(ref_path, 'w') as f:
        json.dump(reference, f, indent=2)
    
    print(f"✅ Reference angles saved to {ref_path}")
    
    for cls, angles in reference.items():
        print(f"\n  {cls}:")
        for angle_name, stats in angles.items():
            print(f"    {angle_name}: {stats['mean']:.1f}° ± {stats['std']:.1f}° "
                  f"[{stats['min']:.1f}° - {stats['max']:.1f}°]")


if __name__ == "__main__":
    main()
