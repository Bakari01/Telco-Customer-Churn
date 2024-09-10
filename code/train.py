import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbalancedPipeline
import pickle

def load_data(train_path, val_path, test_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    return train_df, val_df, test_df

def preprocess_data(train_df, val_df, test_df):
    features = train_df.columns.drop('churn')
    target = 'churn'
    
    numerical_features = train_df[features].select_dtypes(include=['int64', 'float64']).columns
    categorical_features = train_df[features].select_dtypes(include=['object']).columns
    
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor, features, target

def create_model():
    return GradientBoostingClassifier(
        learning_rate=0.01,
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=4,
        random_state=42
    )

def train_model(train_df, preprocessor, features, target):
    pipeline = ImbalancedPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', create_model())
    ])
    
    pipeline.fit(train_df[features], train_df[target])
    
    return pipeline

def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'Precision': precision_score(y, y_pred),
        'Recall': recall_score(y, y_pred),
        'F1 Score': f1_score(y, y_pred),
        'AUC': roc_auc_score(y, y_pred_proba)
    }

def save_model(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f) 
        
def main():
    # Load data
    train_df, val_df, test_df = load_data('C:/Users/HP/Desktop/telco-customer-churn/data/processed/train_processed.csv', 
                                          'C:/Users/HP/Desktop/telco-customer-churn/data/processed/validation_processed.csv', 
                                          'C:/Users/HP/Desktop/telco-customer-churn/data/processed/test_processed.csv')
    
    # Preprocess data
    preprocessor, features, target = preprocess_data(train_df, val_df, test_df)
    
    # Train model
    model = train_model(train_df, preprocessor, features, target)
    
    # Evaluate model
    train_results = evaluate_model(model, train_df[features], train_df[target])
    val_results = evaluate_model(model, val_df[features], val_df[target])
    test_results = evaluate_model(model, test_df[features], test_df[target])
    
    # Print results
    print("Train Results:", train_results)
    print("Validation Results:", val_results)
    print("Test Results:", test_results)
    
    # Save model
    save_model(model, 'model.pkl')
    print("Model saved successfully.")

if __name__ == "__main__":
    main()