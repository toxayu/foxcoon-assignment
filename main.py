import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb

def load_and_preprocess_data(train_path, test_path):
    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Separate features and target
    X_train = train_data.drop(['Exited', 'CustomerId', 'Surname'], axis=1)
    y_train = train_data['Exited']
    X_test = test_data.drop(['CustomerId', 'Surname'], axis=1)
    
    # Handle categorical variables
    X_train = pd.get_dummies(X_train, drop_first=True)
    X_test = pd.get_dummies(X_test, drop_first=True)
    
    # Ensure X_test has the same columns as X_train
    for col in X_train.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, y_train, X_test_scaled, test_data['CustomerId'], X_train.columns

def train_models(X_train, y_train):
    models = {
        # 'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
        # 'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
    return models

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    return f1, cm

def ensemble_predict(models, X):
    predictions = np.zeros((X.shape[0], len(models)))
    for i, (name, model) in enumerate(models.items()):
        predictions[:, i] = model.predict_proba(X)[:, 1]
    return np.mean(predictions, axis=1)

def plot_feature_importance(models, feature_names):
    plt.figure(figsize=(25, 5))
    
    for i, (name, model) in enumerate(models.items()):
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.subplot(1, 5, i+1)
            plt.title(f"{name} Feature Importances")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def main():
    # Load and preprocess data
    X_train, y_train, X_test, customer_ids, feature_names = load_and_preprocess_data('train.csv', 'test.csv')
    
    # Split training data for validation
    X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    # Train models
    models = train_models(X_train_split, y_train_split)
    
    # Evaluate ensemble model
    y_val_pred = ensemble_predict(models, X_val)
    y_val_pred_binary = (y_val_pred > 0.5).astype(int)
    f1 = f1_score(y_val, y_val_pred_binary)
    cm = confusion_matrix(y_val, y_val_pred_binary)
    
    # Make predictions on test set
    y_test_proba = ensemble_predict(models, X_test)
    y_test_pred_binary = (y_test_proba > 0.5).astype(int)
    
    # Create output DataFrame
    output_df = pd.DataFrame({
        'CustomerId': customer_ids,
        'predicted_exit': y_test_pred_binary
    })
    
    # Save predictions to CSV
    output_df.to_csv('predictions.csv', index=False)
    
    # Print evaluation metrics
    print(f"Ensemble Model F1 Score: {f1}")
    print("Ensemble Model Confusion Matrix:")
    print(cm)
    print("\nEnsemble Model Classification Report:")
    print(classification_report(y_val, y_val_pred_binary))
    
    # Plot feature importance
    plot_feature_importance(models, feature_names)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm)

if __name__ == "__main__":
    
    main()
