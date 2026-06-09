import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
import pickle
import joblib

class SupplyChainMLPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.model_metrics = {}
        self.feature_importance = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load the CSV data"""
        self.df = pd.read_csv(self.data_path, encoding='ISO-8859-1')
        print(f"Data loaded: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Basic EDA"""
        print("\n=== DATA EXPLORATION ===")
        print(f"Shape: {self.df.shape}")
        print(f"\nMissing values:\n{self.df.isnull().sum()}")
        print(f"\nTarget variable distribution:\n{self.df['Late_delivery_risk'].value_counts()}")
        print(f"\nClass balance: {self.df['Late_delivery_risk'].value_counts(normalize=True)}")
        return self.df.describe()
    
    def preprocess_data(self):
        """Data preprocessing and cleaning"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Create a copy
        df = self.df.copy()
        
        # Remove duplicates
        df = df.drop_duplicates()
        print(f"Duplicates removed. Shape: {df.shape}")
        
        # Handle missing values
        # Drop rows with missing critical columns
        critical_cols = ['Late_delivery_risk', 'Days for shipping (real)', 
                        'Days for shipment (scheduled)', 'Shipping Mode', 'Market']
        df = df.dropna(subset=critical_cols)
        
        # Fill missing values for other columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        print(f"Missing values after cleaning: {df.isnull().sum().sum()}")
        
        self.df = df
        return df
    
    def engineer_features(self):
        """Feature engineering - create new predictive indicators"""
        print("\n=== FEATURE ENGINEERING ===")
        
        df = self.df.copy()
        
        # Shipping pressure index (actual vs scheduled)
        df['shipping_pressure_index'] = (
            df['Days for shipping (real)'] / (df['Days for shipment (scheduled)'] + 1)
        )
        
        # Schedule deviation (days late/early)
        df['schedule_deviation'] = (
            df['Days for shipping (real)'] - df['Days for shipment (scheduled)']
        )
        
        # Order complexity score (based on discount and quantity)
        df['order_complexity_score'] = (
            df['Order Item Quantity'] * (1 + df['Order Item Discount Rate'])
        )
        
        # Profit efficiency
        df['profit_efficiency'] = (
            df['Order Profit Per Order'] / (df['Sales'] + 1)
        )
        
        # Discount impact
        df['discount_impact'] = df['Order Item Discount'] / (df['Sales'] + 1)
        
        # Product price to category average
        df['price_category_ratio'] = (
            df['Product Price'] / (df['Order Item Product Price'] + 1)
        )
        
        # Customer purchase value quartile
        df['customer_value_tier'] = pd.qcut(df['Sales per customer'], 
                                             q=4, labels=[0, 1, 2, 3], duplicates='drop')
        
        # Days for shipping risk category
        df['shipping_speed_category'] = pd.cut(df['Days for shipping (real)'], 
                                                bins=[0, 5, 10, 20, 100], 
                                                labels=[0, 1, 2, 3])
        
        print("New features created:")
        print("- shipping_pressure_index")
        print("- schedule_deviation")
        print("- order_complexity_score")
        print("- profit_efficiency")
        print("- discount_impact")
        print("- price_category_ratio")
        print("- customer_value_tier")
        print("- shipping_speed_category")
        
        self.df = df
        return df
    
    def prepare_features(self):
        """Prepare features for modeling"""
        print("\n=== FEATURE PREPARATION ===")
        
        df = self.df.copy()
        
        # Drop rows with ANY missing values
        df = df.dropna()
        print(f"After dropping NaN rows: {df.shape}")
        
        # Select features for modeling
        feature_cols = [
            # Time-based features
            'Days for shipping (real)',
            'Days for shipment (scheduled)',
            'shipping_pressure_index',
            'schedule_deviation',
            'shipping_speed_category',
            
            # Order features
            'Order Item Quantity',
            'Order Item Discount',
            'Order Item Discount Rate',
            'Order Item Product Price',
            'Order Item Profit Ratio',
            'Sales',
            'Order Profit Per Order',
            'Benefit per order',
            'Sales per customer',
            'order_complexity_score',
            'profit_efficiency',
            'discount_impact',
            'price_category_ratio',
            
            # Categorical features
            'Type',
            'Category Id',
            'Customer Segment',
            'Department Id',
            'Market',
            'Order Region',
            'Shipping Mode',
            'Delivery Status',
            'Order Status',
            'customer_value_tier'
        ]
        
        # Remove features that don't exist
        feature_cols = [col for col in feature_cols if col in df.columns]
        self.feature_names = feature_cols  # Store feature names for later use
        
        # Separate numeric and categorical features
        numeric_features = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        categorical_features = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numeric features: {len(numeric_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        # Encode categorical variables
        for col in categorical_features:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        
        X = df[feature_cols].copy()
        y = df['Late_delivery_risk'].copy()
        
        # Ensure NO NaN values remain
        X = X.fillna(X.median(numeric_only=True))
        X = X.astype(float)
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median(numeric_only=True))
        
        # Scale numeric features
        X[numeric_features] = self.scaler.fit_transform(X[numeric_features])
        
        # Verify no NaN before split
        assert not X.isnull().any().any(), "X still contains NaN"
        print("Verified: No NaN in features")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class balance in train: {self.y_train.value_counts(normalize=True).to_dict()}")
        
        # Store as dataframes before converting to numpy
        self.X_train_df = self.X_train.copy()
        self.X_test_df = self.X_test.copy()
        
        # Convert to numpy arrays to ensure no NaN
        self.X_train = np.nan_to_num(self.X_train.values)
        self.X_test = np.nan_to_num(self.X_test.values)
        self.y_train = self.y_train.values
        self.y_test = self.y_test.values
        
        # Apply SMOTE to handle class imbalance
        print("Applying SMOTE for class balance...")
        smote = SMOTE(random_state=42, k_neighbors=3)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        print(f"After SMOTE: {self.X_train.shape}")
        print(f"Class balance after SMOTE: {pd.Series(self.y_train).value_counts(normalize=True).to_dict()}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """Train multiple models"""
        print("\n=== MODEL TRAINING ===")
        
        models_to_train = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }
        
        for name, model in models_to_train.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            self.models[name] = model
            print(f"{name} trained successfully")
        
        return self.models
    
    def evaluate_models(self):
        """Evaluate all models"""
        print("\n=== MODEL EVALUATION ===")
        
        for name, model in self.models.items():
            print(f"\n--- {name} ---")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Metrics
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            self.model_metrics[name] = {
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred, output_dict=True)
            }
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"\nConfusion Matrix:\n{self.model_metrics[name]['confusion_matrix']}")
            print(f"\n{classification_report(self.y_test, y_pred)}")
        
        return self.model_metrics
    
    def feature_importance_analysis(self):
        """Extract feature importance from models"""
        print("\n=== FEATURE IMPORTANCE ANALYSIS ===")
        
        feature_names = self.feature_names  # Use stored feature names
        
        # Random Forest feature importance
        if 'Random Forest' in self.models:
            rf_importance = self.models['Random Forest'].feature_importances_
            self.feature_importance['Random Forest'] = sorted(
                zip(feature_names, rf_importance),
                key=lambda x: x[1],
                reverse=True
            )
            print("\nRandom Forest - Top 10 Features:")
            for i, (feat, imp) in enumerate(self.feature_importance['Random Forest'][:10], 1):
                print(f"{i}. {feat}: {imp:.4f}")
        
        # Gradient Boosting feature importance
        if 'Gradient Boosting' in self.models:
            gb_importance = self.models['Gradient Boosting'].feature_importances_
            self.feature_importance['Gradient Boosting'] = sorted(
                zip(feature_names, gb_importance),
                key=lambda x: x[1],
                reverse=True
            )
            print("\nGradient Boosting - Top 10 Features:")
            for i, (feat, imp) in enumerate(self.feature_importance['Gradient Boosting'][:10], 1):
                print(f"{i}. {feat}: {imp:.4f}")
        
        return self.feature_importance
    
    def save_artifacts(self, output_dir):
        """Save models and artifacts"""
        print(f"\n=== SAVING ARTIFACTS TO {output_dir} ===")
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{output_dir}/{name.replace(' ', '_').lower()}_model.pkl")
        
        # Save scaler
        joblib.dump(self.scaler, f"{output_dir}/scaler.pkl")
        
        # Save label encoders
        joblib.dump(self.label_encoders, f"{output_dir}/label_encoders.pkl")
        
        # Save feature names
        with open(f"{output_dir}/feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        # Save metrics
        with open(f"{output_dir}/model_metrics.pkl", 'wb') as f:
            pickle.dump(self.model_metrics, f)
        
        # Save feature importance
        with open(f"{output_dir}/feature_importance.pkl", 'wb') as f:
            pickle.dump(self.feature_importance, f)
        
        print("All artifacts saved successfully")
    
    def run_full_pipeline(self, output_dir='/home/claude/models'):
        """Execute the complete pipeline"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("="*60)
        print("APL LOGISTICS - LATE DELIVERY RISK PREDICTION")
        print("Machine Learning Pipeline")
        print("="*60)
        
        self.load_data()
        self.explore_data()
        self.preprocess_data()
        self.engineer_features()
        self.prepare_features()
        self.train_models()
        self.evaluate_models()
        self.feature_importance_analysis()
        self.save_artifacts(output_dir)
        
        print("\n" + "="*60)
        print("Pipeline execution completed successfully!")
        print("="*60)
        
        return self

# Run the pipeline
if __name__ == "__main__":
    pipeline = SupplyChainMLPipeline(r"C:\Users\Neeraj_Thakur\OneDrive\Documents\APL Logistics ML Project\APL_Logistics_Data.csv")
    pipeline.run_full_pipeline()


