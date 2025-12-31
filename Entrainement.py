"""
GLM Model Selection and Deployment Framework
============================================
A production-ready framework for GLM model selection, training, and serving.

"""
import os
import json
import pickle
import logging

import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum

import random
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelSelectionStrategy(Enum):
    """Stratégies de sélection de modèle disponibles."""
    RANDOM = "random"
    EXHAUSTIVE = "exhaustive"
    FORWARD = "forward"
    BACKWARD = "backward"


@dataclass
class ModelConfig:
    """Configuration pour la sélection et l'entraînement du modèle."""
    target_column: str = "presence_unpaid"
    predictors: List[str] = field(default_factory=list)
    max_iterations: int = 100
    random_seed: int = 42
    test_size: float = 0.2
    min_predictors: int = 1
    max_predictors: Optional[int] = None
    selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.RANDOM
    confidence_level: float = 0.95

    def validate(self) -> None:
        """Valide la configuration."""
        if self.test_size <= 0 or self.test_size >= 1:
            raise ValueError("test_size must be between 0 and 1")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.min_predictors <= 0:
            raise ValueError("min_predictors must be positive")
        if self.max_predictors and self.max_predictors < self.min_predictors:
            raise ValueError("max_predictors must be >= min_predictors")


@dataclass
class ModelMetrics:
    """Métriques d'évaluation du modèle."""
    aic: float
    bic: float
    auc: float
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    log_likelihood: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    roc_curve: Optional[Dict[str, List[float]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit les métriques en dictionnaire."""
        data = asdict(self)
        if self.confusion_matrix is not None:
            data['confusion_matrix'] = self.confusion_matrix.tolist()
        return data


@dataclass
class ModelResult:
    """Résultat d'un modèle entraîné."""
    formula: str
    predictors: List[str]
    model: Any
    metrics: ModelMetrics
    timestamp: datetime = field(default_factory=datetime.now)
    config: Optional[ModelConfig] = None


class DataValidator:
    """Validateur de données pour l'entraînement."""

    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame,
        target_column: str,
        predictors: List[str]
    ) -> None:
        """Valide le DataFrame d'entrée."""
        if df.empty:
            raise ValueError("Input dataframe is empty")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")

        missing_predictors = set(predictors) - set(df.columns)
        if missing_predictors:
            raise ValueError(f"Predictors not found: {missing_predictors}")

        null_counts = df[predictors + [target_column]].isnull().sum()
        if null_counts.any():
            logger.warning(f"Missing values detected: \n{null_counts[null_counts > 0]}")

        unique_targets = df[target_column].unique()
        if len(unique_targets) != 2:
            raise ValueError(f"Target must be binary, found {len(unique_targets)} unique values")

        constant_cols = [col for col in predictors if df[col].nunique() == 1]
        if constant_cols:
            logger.warning(f"Constant predictors detected: {constant_cols}")


class GLMModelSelector:
    """Sélecteur de modèle GLM avec différentes stratégies de recherche."""

    def __init__(self, config: ModelConfig):
        config.validate()
        self.config = config
        self.best_model: Optional[ModelResult] = None
        self.all_models: List[ModelResult] = []
        self.train_data: Optional[pd.DataFrame] = None
        self.test_data: Optional[pd.DataFrame] = None

        np.random.seed(config.random_seed)
        random.seed(config.random_seed)

    def prepare_data(
        self,
        data: pd.DataFrame,
        train_data: Optional[pd.DataFrame] = None,
        test_data: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prépare les données d'entraînement et de test."""
        if train_data is not None and test_data is not None:
            self.train_data = train_data.copy()
            self.test_data = test_data.copy()
        else:
            self.train_data, self.test_data = train_test_split(
                data,
                test_size=self.config.test_size,
                random_state=self.config.random_seed,
                stratify=data[self.config.target_column]
            )

        DataValidator.validate_dataframe(
            self.train_data,
            self.config.target_column,
            self.config.predictors
        )
        DataValidator.validate_dataframe(
            self.test_data,
            self.config.target_column,
            self.config.predictors
        )

        logger.info(f'Data prepared: {len(self.train_data)} train, {len(self.test_data)} test samples')
        return self.train_data, self.test_data

    def _fit_model(
            self,
            predictors: List[str],
            train_data: pd.DataFrame,
            test_data: pd.DataFrame
    ) -> ModelResult:
        """Entraîne un modèle GLM avec les prédicteurs spécifiés."""
        formula = f"{self.config.target_column} ~ {' + '.join(predictors)}"

        try:
            model = smf.glm(
                formula=formula,
                data=train_data,
                family=sm.families.Binomial()
            ).fit()

            y_test = test_data[self.config.target_column]
            predicted_probs = model.predict(test_data)

            auc = roc_auc_score(y_test, predicted_probs)

            threshold = 0.5
            predicted_classes = (predicted_probs >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, predicted_classes).ravel()
            accuracy = (tn + tp) / (tn + fp + fn + tp)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            fpr, tpr, thresholds = roc_curve(y_test, predicted_probs)

            metrics = ModelMetrics(
                aic=model.aic,
                bic=model.bic,
                auc=auc,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                log_likelihood=model.llf,
                confusion_matrix=confusion_matrix(y_test, predicted_classes),
                roc_curve={
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'thresholds': thresholds.tolist()
                }
            )

            return ModelResult(
                formula=formula,
                predictors=predictors,
                model=model,
                metrics=metrics,
                config=self.config
            )

        except Exception as e:
            logger.error(f"Failed to fit model with predictors {predictors}: {str(e)}")
            raise

    def _random_search(self) -> ModelResult:
        """Effectue une recherche aléatoire de modèles."""
        best_aic = float('inf')
        best_model = None

        for iteration in range(self.config.max_iterations):
            max_k = self.config.max_predictors or len(self.config.predictors)
            k = random.randint(
                self.config.min_predictors,
                min(max_k, len(self.config.predictors))
            )

            selected_predictors = random.sample(self.config.predictors, k)

            try:
                model_result = self._fit_model(
                    selected_predictors,
                    self.train_data,
                    self.test_data
                )

                self.all_models.append(model_result)

                if model_result.metrics.aic < best_aic:
                    best_aic = model_result.metrics.aic
                    best_model = model_result
                    logger.info(
                        f"Iteration {iteration + 1}: New best model found "
                        f"(AIC={best_aic:.2f}, AUC={model_result.metrics.auc:.4f})"
                    )

            except Exception as e:
                logger.warning(f"Iteration {iteration + 1} failed: {str(e)}")
                continue

        if best_model is None:
            raise ValueError(
                f"No valid model found after {self.config.max_iterations} iterations"
            )

        logger.info(
            f"Random search completed: Best AIC={best_model.metrics.aic:.2f}, "
            f"AUC={best_model.metrics.auc:.4f}, "
            f"Variables={best_model.predictors}"
        )

        return best_model

    def fit(self) -> ModelResult:
        """Lance la sélection et l'entraînement du modèle."""
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data must be prepared before fitting")

        logger.info(f"Starting model selection with strategy: {self.config.selection_strategy.value}")

        if self.config.selection_strategy == ModelSelectionStrategy.RANDOM:
            self.best_model = self._random_search()
        else:
            raise NotImplementedError(f"Strategy {self.config.selection_strategy} not implemented")

        if self.best_model is None:
            raise RuntimeError("No valid model found")

        logger.info(
            f"Best model selected with {len(self.best_model.predictors)} predictors, "
            f"AIC={self.best_model.metrics.aic:.2f}, AUC={self.best_model.metrics.auc:.4f}"
        )

        return self.best_model

    def predict(
            self,
            X: pd.DataFrame,
            return_proba: bool = True,
            threshold: float = 0.5,
            return_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Effectue des prédictions avec le meilleur modèle."""
        if self.best_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")

        missing_cols = set(self.best_model.predictors) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns: {missing_cols}. "
                f"Required: {self.best_model.predictors}"
            )

        probabilities = self.best_model.model.predict(X)

        if return_dataframe:
            result = pd.DataFrame({
                'proba_default': probabilities,
                'predicted_class': (probabilities >= threshold).astype(int),
                'decision': ['REFUSE' if p >= threshold else 'ACCEPT'
                             for p in probabilities]
            }, index=X.index)
            return result
        elif return_proba:
            return probabilities
        else:
            return (probabilities >= threshold).astype(int)

    def save_model(self, filepath: str) -> None:
        """Sauvegarde le meilleur modèle."""
        if self.best_model is None:
            raise ValueError("No model to save. Call fit() first.")

        joblib.dump(self.best_model, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'GLMModelSelector':
        """Charge un modèle sauvegardé."""
        model_result = joblib.load(filepath)

        if not isinstance(model_result, ModelResult):
            raise ValueError("Invalid model file format")

        selector = cls(model_result.config)
        selector.best_model = model_result
        logger.info(f"Model loaded from {filepath}")

        return selector
