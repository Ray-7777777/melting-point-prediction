"""
Pipeline principal de prédiction du point de fusion
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import catboost as cb
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')
import os

class MeltingPointPredictor:
    """Classe principale pour la prédiction du point de fusion"""
    
    def __init__(self):
        self.models = {}
        self.features = None
        self.scaler = None
        
    def load_data(self, train_path, test_path):
        """Charge et prépare les données"""
        print("="*60)
        print("CHARGEMENT DES DONNÉES")
        print("="*60)
        
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        return self.preprocess_data(train_df, test_df)
    
    def preprocess_data(self, train_df, test_df):
        """Prétraitement identique au pipeline Kaggle"""
        # Séparation features/target
        X = train_df.drop(['id', 'SMILES', 'Tm'], axis=1)
        y = train_df['Tm']
        X_test = test_df.drop(['id', 'SMILES'], axis=1)
        test_ids = test_df['id']
        
        # Imputation des valeurs manquantes
        X = X.fillna(X.mean())
        X_test = X_test.fillna(X_test.mean())
        
        # Sélection des features par variance
        variances = X.var()
        keep_cols = variances[variances > 0.0001].index
        X = X[keep_cols]
        X_test = X_test[keep_cols]
        
        self.features = keep_cols
        print(f"Features gardées: {len(keep_cols)}")
        
        return X.values, y, X_test.values, test_ids
    
    def train_models(self, X, y):
        """Entraîne les modèles de gradient boosting"""
        print("\n" + "="*60)
        print("ENTRAÎNEMENT DES MODÈLES")
        print("="*60)
        
        # Split pour early stopping
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.1, random_state=42
        )
        
        # 1. CatBoost
        print("\n1. Entraînement CatBoost...")
        cat_params = {
            'iterations': 1500,
            'learning_rate': 0.045,
            'depth': 6,
            'l2_leaf_reg': 3,
            'border_count': 128,
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'MAE'
        }
        
        cat_model = cb.CatBoostRegressor(**cat_params)
        cat_model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=100
        )
        self.models['catboost'] = cat_model
        
        # 2. XGBoost
        print("\n2. Entraînement XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            eval_metric='mae'
        )
        xgb_model.fit(X_train, y_train)
        self.models['xgboost'] = xgb_model
        
        # 3. LightGBM
        print("\n3. Entraînement LightGBM...")
        lgb_model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            metric='mae'
        )
        lgb_model.fit(X_train, y_train)
        self.models['lightgbm'] = lgb_model
        
        print("\n✅ Entraînement terminé !")
        return self.models
    
    def predict_ensemble(self, X_test):
        """Prédictions par ensemble stacking"""
        predictions = {}
        
        for name, model in self.models.items():
            pred = model.predict(X_test)
            predictions[name] = pred
        
        # Stacking simple : moyenne des prédictions
        ensemble_pred = np.mean(list(predictions.values()), axis=0)
        
        return ensemble_pred, predictions
    
    def create_submission(self, predictions, test_ids, output_path='submission.csv'):
        """Crée un fichier de soumission Kaggle"""
        submission = pd.DataFrame({
            'id': test_ids,
            'Tm': predictions
        })
        submission.to_csv(output_path, index=False)
        print(f"\n✅ Soumission créée : {output_path}")
        return submission
    
    def evaluate_model(self, X_val, y_val):
        """Évalue les performances du modèle"""
        print("\n" + "="*60)
        print("ÉVALUATION DU MODÈLE")
        print("="*60)
        
        results = {}
        for name, model in self.models.items():
            pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, pred)
            results[name] = mae
            print(f"{name}: MAE = {mae:.2f} K")
        
        # Évaluation de l'ensemble
        ensemble_pred, _ = self.predict_ensemble(X_val)
        ensemble_mae = mean_absolute_error(y_val, ensemble_pred)
        results['ensemble'] = ensemble_mae
        print(f"\n📊 Ensemble final: MAE = {ensemble_mae:.2f} K")
        
        return results

def main():
    """Fonction principale exécutable"""
    # Chemins des données (à adapter)
    train_path = 'data/train.csv'  # À télécharger depuis Kaggle
    test_path = 'data/test.csv'    # À télécharger depuis Kaggle
    output_path = 'submission_final.csv'
    
    # Initialiser le prédicteur
    predictor = MeltingPointPredictor()
    
    # 1. Charger et prétraiter les données
    X, y, X_test, test_ids = predictor.load_data(train_path, test_path)
    
    # 2. Entraîner les modèles
    predictor.train_models(X, y)
    
    # 3. Faire des prédictions (validation)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    results = predictor.evaluate_model(X_val, y_val)
    
    # 4. Prédire sur le test set
    final_predictions, _ = predictor.predict_ensemble(X_test)
    
    # 5. Créer la soumission
    predictor.create_submission(final_predictions, test_ids, output_path)
    
    print("\n" + "="*60)
    print("PIPELINE TERMINÉ AVEC SUCCÈS !")
    print("="*60)
    print(f"MAE final estimé : {results.get('ensemble', 'N/A')} K")
    print(f"Fichier de soumission : {output_path}")

if __name__ == "__main__":
    main()