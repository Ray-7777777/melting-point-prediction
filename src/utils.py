"""
Fonctions utilitaires pour le projet
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

def subtle_micro_corrections(predictions: np.ndarray, 
                            y_train_values: np.ndarray, 
                            aggression: float = 0.001) -> np.ndarray:
    """
    Applique des corrections subtiles aux prédictions.
    
    Args:
        predictions: Prédictions du modèle
        y_train_values: Valeurs réelles d'entraînement
        aggression: Niveau d'agressivité de la correction (0.001 = 0.1%)
    
    Returns:
        Prédictions corrigées
    """
    predictions = predictions.copy()
    
    # Statistiques de référence
    train_mean = np.mean(y_train_values)
    pred_mean = np.mean(predictions)
    
    # Correction de moyenne subtile
    mean_diff = train_mean - pred_mean
    if abs(mean_diff) > 0.2:
        correction = 1 + (mean_diff / pred_mean) * aggression
        predictions = predictions * correction
    
    # Ajustement des quantiles
    for percentile in [10, 25, 75, 90]:
        pred_q = np.percentile(predictions, percentile)
        train_q = np.percentile(y_train_values, percentile)
        diff = train_q - pred_q
        
        if abs(diff) > 1.0:
            micro_correction = 1 + (diff / pred_q) * (aggression / 2)
            
            if percentile <= 50:
                mask = predictions <= pred_q
            else:
                mask = predictions >= pred_q
            
            if np.sum(mask) > 10:
                predictions[mask] = predictions[mask] * micro_correction
    
    # Clipping doux
    q00_25 = np.percentile(y_train_values, 0.25)
    q99_75 = np.percentile(y_train_values, 99.75)
    predictions = np.clip(predictions, q00_25, q99_75)
    
    # Arrondi
    predictions = np.round(predictions, 1)
    
    return predictions

def create_ensemble_variants(base_pred: np.ndarray, y_train: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Crée différentes variantes pour l'ensemblage.
    
    Args:
        base_pred: Prédictions de base
        y_train: Valeurs d'entraînement
    
    Returns:
        Dictionnaire de variantes de prédictions
    """
    variants = {}
    
    # Variante 1: Original
    variants['original'] = base_pred
    
    # Variante 2: Correction ultra-subtile
    variants['ultra_subtle'] = subtle_micro_corrections(base_pred, y_train, 0.0005)
    
    # Variante 3: Shift vers la médiane
    median = np.median(y_train)
    variants['median_shift'] = base_pred * 0.9995 + median * 0.0005
    
    # Variante 4: Compression des extrêmes
    z_scores = (base_pred - np.mean(base_pred)) / (np.std(base_pred) + 1e-8)
    variants['compressed'] = np.mean(base_pred) + np.std(base_pred) * np.tanh(z_scores * 0.99)
    
    return variants

def analyze_predictions(predictions: np.ndarray, actual: np.ndarray = None) -> Dict:
    """
    Analyse statistique des prédictions.
    
    Args:
        predictions: Prédictions à analyser
        actual: Valeurs réelles (optionnel)
    
    Returns:
        Dictionnaire de statistiques
    """
    stats = {
        'mean': float(np.mean(predictions)),
        'median': float(np.median(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'q10': float(np.percentile(predictions, 10)),
        'q90': float(np.percentile(predictions, 90))
    }
    
    if actual is not None:
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        stats['mae'] = float(mean_absolute_error(actual, predictions))
        stats['rmse'] = float(np.sqrt(mean_squared_error(actual, predictions)))
        stats['r2'] = float(r2_score(actual, predictions))
    
    return stats

def save_model_artifacts(model, path: str):
    """
    Sauvegarde les artefacts du modèle.
    
    Args:
        model: Modèle à sauvegarder
        path: Chemin de sauvegarde
    """
    import joblib
    joblib.dump(model, path)
    print(f"✅ Modèle sauvegardé : {path}")