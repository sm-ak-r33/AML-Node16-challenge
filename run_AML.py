# Money Laundering Detection logging and structure

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, average_precision_score,
                            precision_recall_curve, auc)
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any
import joblib
from tabulate import tabulate
from datetime import datetime
import sys
import copy


class ResultLogger:
    """Handles logging to both console and results.txt file."""
    
    def __init__(self, filepath='results.txt'):
        self.filepath = filepath
        self.file = open(filepath, 'w', encoding='utf-8')
        
    def log(self, message, to_console=True):
        """Log message to file and optionally to console."""
        if to_console:
            print(message)
        self.file.write(message + '\n')
        self.file.flush()
    
    def close(self):
        """Close the file."""
        self.file.close()


class AMLModel:
    """
    Production-ready Anti-Money Laundering Model with proper federated learning.
    
    Key Features:
    - Parquet file reading with fastparquet
    - Native categorical feature handling (no encoding needed)
    - Temporal data splits (no random splits)
    - True federated learning (sequential training, not data merging)
    - PR-AUC as primary evaluation metric
    - Automatic class weight handling via is_unbalance=True
    """
    
    def __init__(self, categorical_features=None, random_state=42, verbose=False, bank_name=None):
        """
        Initialize AML Model.
        
        Args:
            categorical_features: List of categorical feature names
            random_state: Random seed for reproducibility
            verbose: If True, show detailed logs
            bank_name: Name identifier for this bank/institution
        """
        self.categorical_features = categorical_features or []
        self.random_state = random_state
        self.verbose = verbose
        self.bank_name = bank_name or "Unknown Bank"
        self.model = None
        self.X_train = self.X_val = self.X_test = None
        self.y_train = self.y_val = self.y_test = None
        self.best_params = None
        self.df = None
        self.grid_search_results = []
        
        # Configure logging
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging and suppress unnecessary warnings."""
        if not self.verbose:
            warnings.filterwarnings('ignore', category=UserWarning)
            warnings.filterwarnings('ignore', message='.*Unknown parameter.*')
            
            try:
                import lightgbm
                if hasattr(lightgbm, 'set_log_level'):
                    lightgbm.set_log_level('ERROR')
            except:
                pass
        
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            self.logger.setLevel(logging.INFO)
        else:
            self.logger.setLevel(logging.WARNING)

    def load_data(self, filepath: str, logger: ResultLogger = None) -> pd.DataFrame:
        """Load and preprocess data from Parquet file."""
        try:
            df = pd.read_parquet(filepath, engine='fastparquet')
        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {filepath}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")
        
        if "Is Laundering" not in df.columns:
            raise ValueError("Target column 'Is Laundering' not found in data")
        
        # Remove duplicates
        initial_size = len(df)
        df.drop_duplicates(inplace=True)
        duplicates_removed = initial_size - len(df)
        
        # Process timestamp for temporal splitting
        if "Timestamp" in df.columns:
            df["Timestamp"] = pd.to_datetime(df["Timestamp"])
            # Extract temporal features
            df["Day"] = df["Timestamp"].dt.day_name().astype('category')
            df["Hour"] = df["Timestamp"].dt.hour
            df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
            df["Month"] = df["Timestamp"].dt.month
            df["Year"] = df["Timestamp"].dt.year
            
            # Sort by timestamp for temporal splits
            df = df.sort_values('Timestamp').reset_index(drop=True)
        else:
            raise ValueError("Timestamp column required for temporal splits")
        
        # Convert specified categorical features to category dtype
        for col in self.categorical_features:
            if col in df.columns and col != 'Day':  # Day already converted
                df[col] = df[col].astype('category')
        
        # Also convert any object columns that aren't in categorical_features to category
        # (in case user forgot to specify them)
        for col in df.columns:
            if df[col].dtype == 'object' and col not in ['Timestamp', 'Is Laundering']:
                df[col] = df[col].astype('category')
                if col not in self.categorical_features:
                    self.categorical_features.append(col)
        
        self.df = df
        
        msg = f"\n{self.bank_name} - Data Loaded:"
        msg += f"\n  Total rows: {len(df)}"
        msg += f"\n  Duplicates removed: {duplicates_removed}"
        
        pos_count = (df["Is Laundering"] == 1).sum()
        neg_count = (df["Is Laundering"] == 0).sum()
        msg += f"\n  Class distribution:"
        msg += f"\n    Positive (Laundering): {pos_count} ({pos_count/len(df)*100:.2f}%)"
        msg += f"\n    Negative (Normal): {neg_count} ({neg_count/len(df)*100:.2f}%)"
        
        # Report categorical features
        cat_features = [col for col in df.columns if df[col].dtype == 'category']
        if cat_features:
            msg += f"\n  Categorical features detected: {cat_features}"
        
        if logger:
            logger.log(msg)
        elif self.verbose:
            print(msg)
        
        return df

    def temporal_split(self, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, 
                      logger: ResultLogger = None) -> None:
        """
        Split data temporally (chronologically) to mimic real-world scenarios.
        No random shuffling - maintains temporal order.
        
        Args:
            train_ratio: Proportion of earliest data for training
            val_ratio: Proportion of middle data for validation
            test_ratio: Proportion of latest data for testing
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
            raise ValueError("Ratios must sum to 1.0")
        
        # Data is already sorted by timestamp in load_data()
        n = len(self.df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Temporal split
        train_df = self.df.iloc[:train_end]
        val_df = self.df.iloc[train_end:val_end]
        test_df = self.df.iloc[val_end:]
        
        # Remove Timestamp column after split (not needed for modeling)
        cols_to_drop = ["Is Laundering"]
        if 'Timestamp' in train_df.columns:
            cols_to_drop.append('Timestamp')
        
        self.X_train = train_df.drop(cols_to_drop, axis=1)
        self.y_train = train_df["Is Laundering"]
        
        self.X_val = val_df.drop(cols_to_drop, axis=1)
        self.y_val = val_df["Is Laundering"]
        
        self.X_test = test_df.drop(cols_to_drop, axis=1)
        self.y_test = test_df["Is Laundering"]
        
        msg = f"\n{self.bank_name} - Temporal Split (maintaining chronological order):"
        msg += f"\n  Train: {len(self.X_train)} samples ({self.y_train.sum()} positive, {len(self.y_train) - self.y_train.sum()} negative)"
        msg += f"\n  Val:   {len(self.X_val)} samples ({self.y_val.sum()} positive, {len(self.y_val) - self.y_val.sum()} negative)"
        msg += f"\n  Test:  {len(self.X_test)} samples ({self.y_test.sum()} positive, {len(self.y_test) - self.y_test.sum()} negative)"
        
        if logger:
            logger.log(msg)
        elif self.verbose:
            print(msg)

    def build_model(self, lgb_params=None) -> lgb.LGBMClassifier:
        """
        Build LightGBM classifier with native categorical feature support.
        No encoding needed - LightGBM handles categories natively.
        """
        if self.X_train is None:
            raise ValueError("Data not split. Call temporal_split() first.")
        
        if lgb_params is None:
            lgb_params = {}
        
        # Ensure categorical features exist in the dataset
        valid_categorical = [col for col in self.categorical_features if col in self.X_train.columns]
        
        # Default parameters - no class_weight, using is_unbalance instead
        default_params = {
            'random_state': self.random_state,
            'is_unbalance': True,  # Automatic class weight handling
            'force_col_wise': True,
            'verbose': -1 if not self.verbose else 0,
        }
        
        final_params = {**default_params, **lgb_params}
        
        self.model = lgb.LGBMClassifier(**final_params)
        self.valid_categorical = valid_categorical  # Store for use in training
        
        return self.model

    def train(self, init_model=None) -> None:
        """
        Train the model.
        
        Args:
            init_model: If provided, continue training from this model (federated learning)
        """
        if self.model is None and init_model is None:
            raise ValueError("No model to train. Call build_model() first or provide init_model.")
        if self.X_train is None:
            raise ValueError("Data not split. Call temporal_split() first.")
        
        # Get valid categorical features
        valid_categorical = [col for col in self.categorical_features if col in self.X_train.columns]
        
        if init_model is not None:
            # Federated learning: continue training from previous bank's model
            # Use init_model parameter which is the correct way in LightGBM
            self.model.fit(
                self.X_train, self.y_train,
                categorical_feature=valid_categorical if valid_categorical else 'auto',
                init_model=init_model
            )
        else:
            # Initial training
            self.model.fit(
                self.X_train, self.y_train,
                categorical_feature=valid_categorical if valid_categorical else 'auto'
            )

    def evaluate(self, X, y, dataset_name="") -> Dict[str, float]:
        """
        Evaluate model with PR-AUC as primary metric.
        
        Returns comprehensive metrics including PR-AUC, ROC-AUC, F1, precision, recall.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        y_pred = self.model.predict(X)
        y_pred_proba = self.model.predict_proba(X)[:, 1]
        
        # Calculate PR-AUC (Primary metric)
        pr_auc = average_precision_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0
        
        # Other metrics
        metrics = {
            "pr_auc": pr_auc,  # PRIMARY METRIC
            "roc_auc": roc_auc_score(y, y_pred_proba) if len(np.unique(y)) > 1 else 0.0,
            "accuracy": accuracy_score(y, y_pred),
            "precision": precision_score(y, y_pred, zero_division=0),
            "recall": recall_score(y, y_pred, zero_division=0),
            "f1": f1_score(y, y_pred, zero_division=0),
        }
        
        return metrics

    def local_grid_search(self, param_grid, logger: ResultLogger = None) -> Tuple[Dict, List[Dict]]:
        """
        Perform grid search on LOCAL data only to find best hyperparameters.
        Uses PR-AUC as the optimization metric.
        """
        if self.X_train is None or self.X_val is None:
            raise ValueError("Data not split. Call temporal_split() first.")
        
        from sklearn.model_selection import ParameterGrid
        
        best_score = -1
        best_params = None
        results = []
        
        total_combinations = len(list(ParameterGrid(param_grid)))
        
        msg = f"\n{'='*80}"
        msg += f"\nLOCAL Grid Search for {self.bank_name}"
        msg += f"\n{'='*80}"
        msg += f"\nTesting {total_combinations} parameter combinations on LOCAL data"
        msg += f"\nOptimization metric: PR-AUC (Precision-Recall Area Under Curve)\n"
        
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        for idx, params in enumerate(ParameterGrid(param_grid), 1):
            self.build_model(params)
            self.train()
            
            # Evaluate on local validation set
            val_metrics = self.evaluate(self.X_val, self.y_val)
            
            result = {
                "params": params,
                "val_metrics": val_metrics,
                "bank": self.bank_name
            }
            
            score = val_metrics["pr_auc"]  # Using PR-AUC as optimization metric
            
            if score > best_score:
                best_score = score
                best_params = params
            
            results.append(result)
            
            msg = f"[{idx}/{total_combinations}] Params: {params}"
            msg += f"\n  Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
            msg += f"F1: {val_metrics['f1']:.4f}, "
            msg += f"Prec: {val_metrics['precision']:.4f}, "
            msg += f"Rec: {val_metrics['recall']:.4f}\n"
            
            if logger:
                logger.log(msg)
            else:
                print(msg)
        
        self.best_params = best_params
        self.grid_search_results = results
        
        msg = f"\n✓ Best LOCAL params for {self.bank_name} (Val PR-AUC={best_score:.4f}):"
        msg += f"\n  {best_params}\n"
        
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        return best_params, results

    def train_final_local_model(self, logger: ResultLogger = None):
        """Train final model with best parameters on local data."""
        if self.best_params is None:
            raise ValueError("No best params found. Run local_grid_search() first.")
        
        msg = f"Training final LOCAL model for {self.bank_name}..."
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        self.build_model(self.best_params)
        self.train()
        
        # Evaluate on local test set
        test_metrics = self.evaluate(self.X_test, self.y_test)
        
        msg = f"✓ {self.bank_name} LOCAL model trained"
        msg += f"\n  Local Test PR-AUC: {test_metrics['pr_auc']:.4f}, "
        msg += f"F1: {test_metrics['f1']:.4f}, "
        msg += f"Prec: {test_metrics['precision']:.4f}, "
        msg += f"Rec: {test_metrics['recall']:.4f}\n"
        
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        return test_metrics

    def save_model(self, filepath: str, logger: ResultLogger = None) -> None:
        """Save trained model to file."""
        if self.model is None:
            raise ValueError("No model to save. Train a model first.")
        
        joblib.dump({
            'model': self.model,
            'categorical_features': self.categorical_features,
            'best_params': self.best_params,
            'bank_name': self.bank_name
        }, filepath)
        
        msg = f"✓ Model saved to {filepath}"
        if logger:
            logger.log(msg)
        else:
            print(msg)

    def load_model(self, filepath: str, logger: ResultLogger = None) -> None:
        """Load trained model from file."""
        data = joblib.load(filepath)
        self.model = data['model']
        self.categorical_features = data['categorical_features']
        self.best_params = data.get('best_params')
        self.bank_name = data.get('bank_name', 'Unknown Bank')
        
        msg = f"✓ Model loaded from {filepath}"
        if logger:
            logger.log(msg)
        else:
            print(msg)

    @staticmethod
    def federated_training_sequential(bank_models: List['AMLModel'], 
                                     best_params: Dict,
                                     logger: ResultLogger = None) -> 'AMLModel':
        """
        True Federated Learning: Sequential training on each bank's data using init_model.
        
        This uses LightGBM's init_model parameter to properly continue training:
        1. Bank A trains a model on its local data
        2. Model is passed to Bank B via init_model parameter
        3. Bank B continues training on its local data
        4. Process continues for all banks
        5. Final model has learned from all banks without sharing raw data
        """

        msg = f"\nTraining model SEQUENTIALLY on {len(bank_models)} banks' data"
        msg += f"\nUsing LightGBM's init_model for proper continuation training"
        msg += f"\nNo data merging - only model parameters are shared!"
        msg += f"\nUsing hyperparameters: {best_params}\n"
        
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        init_model_for_next = None
        
        # Sequential training through all banks
        for idx, bank in enumerate(bank_models, 1):
            msg = f"\n[Step {idx}/{len(bank_models)}] Training on {bank.bank_name}'s data..."
            if logger:
                logger.log(msg)
            else:
                print(msg)
            
            # Create a temporary model for this bank
            temp_model = AMLModel(
                categorical_features=bank.categorical_features,
                random_state=bank.random_state,
                verbose=False,
                bank_name=bank.bank_name
            )
            
            # Use the same train/val/test split
            temp_model.X_train = bank.X_train
            temp_model.y_train = bank.y_train
            temp_model.X_val = bank.X_val
            temp_model.y_val = bank.y_val
            temp_model.X_test = bank.X_test
            temp_model.y_test = bank.y_test
            
            # Build model with same hyperparameters
            temp_model.build_model(best_params)
            
            # Train with init_model if available
            temp_model.train(init_model=init_model_for_next)
            
            # Save this model to pass to next bank
            init_model_for_next = temp_model.model
            
            # Evaluate on this bank's validation set
            val_metrics = temp_model.evaluate(bank.X_val, bank.y_val)
            
            msg = f"  {bank.bank_name} Val PR-AUC: {val_metrics['pr_auc']:.4f}, "
            msg += f"F1: {val_metrics['f1']:.4f}"
            if logger:
                logger.log(msg)
            else:
                print(msg)
        
        # After training on all banks sequentially, create final model
        final_model = AMLModel(
            categorical_features=bank_models[0].categorical_features,
            random_state=bank_models[0].random_state,
            verbose=False,
            bank_name="Global-Federated"
        )
        final_model.model = init_model_for_next
        final_model.best_params = best_params
        
        msg = f"\n✓ Federated model training complete\n"
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        return final_model

    @staticmethod
    def compare_local_vs_global(bank_models: List['AMLModel'], global_model: 'AMLModel',
                               logger: ResultLogger = None):
        """
        Compare local models vs global federated model on each bank's LOCAL test set.
        """
        msg = f"\nComparing performance on each bank's LOCAL test set\n"
        
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        comparison_results = []
        
        for bank in bank_models:
            msg = f"\n{bank.bank_name} Test Set Evaluation:"
            msg += f"\n{'-'*80}"
            if logger:
                logger.log(msg)
            else:
                print(msg)
            
            # Local model on local test set
            local_metrics = bank.evaluate(bank.X_test, bank.y_test)
            msg = f"LOCAL Model (trained only on {bank.bank_name} data):"
            msg += f"\n  PR-AUC    : {local_metrics['pr_auc']:.4f} ← PRIMARY METRIC"
            msg += f"\n  ROC-AUC   : {local_metrics['roc_auc']:.4f}"
            msg += f"\n  F1-Score  : {local_metrics['f1']:.4f}"
            msg += f"\n  Precision : {local_metrics['precision']:.4f}"
            msg += f"\n  Recall    : {local_metrics['recall']:.4f}"
            msg += f"\n  Accuracy  : {local_metrics['accuracy']:.4f}"
            
            if logger:
                logger.log(msg)
            else:
                print(msg)
            
            # Global federated model on this bank's local test set
            global_metrics = global_model.evaluate(bank.X_test, bank.y_test)
            msg = f"\nGLOBAL FEDERATED Model (trained sequentially on all banks):"
            msg += f"\n  PR-AUC    : {global_metrics['pr_auc']:.4f} ← PRIMARY METRIC"
            msg += f"\n  ROC-AUC   : {global_metrics['roc_auc']:.4f}"
            msg += f"\n  F1-Score  : {global_metrics['f1']:.4f}"
            msg += f"\n  Precision : {global_metrics['precision']:.4f}"
            msg += f"\n  Recall    : {global_metrics['recall']:.4f}"
            msg += f"\n  Accuracy  : {global_metrics['accuracy']:.4f}"
            
            if logger:
                logger.log(msg)
            else:
                print(msg)
            
            # Calculate improvements
            pr_auc_change = global_metrics['pr_auc'] - local_metrics['pr_auc']
            f1_change = global_metrics['f1'] - local_metrics['f1']
            
            msg = f"\nIMPROVEMENT (Global vs Local):"
            if local_metrics['pr_auc'] > 0:
                msg += f"\n  PR-AUC : {pr_auc_change:+.4f} ({pr_auc_change/local_metrics['pr_auc']*100:+.2f}%)"
            else:
                msg += f"\n  PR-AUC : N/A"
            
            if local_metrics['f1'] > 0:
                msg += f"\n  F1     : {f1_change:+.4f} ({f1_change/local_metrics['f1']*100:+.2f}%)"
            else:
                msg += f"\n  F1     : N/A"
            
            if logger:
                logger.log(msg)
            else:
                print(msg)
            
            comparison_results.append({
                'Bank': bank.bank_name,
                'Local PR-AUC': f"{local_metrics['pr_auc']:.4f}",
                'Global PR-AUC': f"{global_metrics['pr_auc']:.4f}",
                'PR-AUC Δ': f"{pr_auc_change:+.4f}",
                'Local F1': f"{local_metrics['f1']:.4f}",
                'Global F1': f"{global_metrics['f1']:.4f}",
                'F1 Δ': f"{f1_change:+.4f}",
            })
        
        msg = f"\n{'='*80}"
        msg += f"\nSUMMARY: Local vs Global Performance"
        msg += f"\n{'='*80}"
        summary_table = tabulate(comparison_results, headers='keys', tablefmt='grid')
        msg += f"\n{summary_table}\n"
        
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        return comparison_results

    @staticmethod
    def setup_federated_banks(filepaths: List[str], bank_names: List[str], 
                             categorical_features: List[str],
                             train_ratio=0.6, val_ratio=0.2, test_ratio=0.2,
                             random_state=42, verbose=False,
                             logger: ResultLogger = None) -> List['AMLModel']:
        """Convenience method to set up multiple banks for federated learning."""
        if len(filepaths) != len(bank_names):
            raise ValueError("Number of filepaths must match number of bank names")
        
        banks = []
        msg = f"\nSetting up {len(bank_names)} banks for federated learning..."
        
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        for filepath, bank_name in zip(filepaths, bank_names):
            msg = f"Preparing {bank_name}..."
            if logger:
                logger.log(msg)
            else:
                print(msg)
            
            bank = AMLModel(
                categorical_features=categorical_features,
                random_state=random_state,
                verbose=verbose,
                bank_name=bank_name
            )
            
            try:
                bank.load_data(filepath, logger=logger)
                bank.temporal_split(
                    train_ratio=train_ratio, 
                    val_ratio=val_ratio, 
                    test_ratio=test_ratio,
                    logger=logger
                )
                banks.append(bank)
                
                msg = f"✓ {bank_name} ready\n"
                if logger:
                    logger.log(msg)
                else:
                    print(msg)
            except Exception as e:
                msg = f"✗ Error setting up {bank_name}: {str(e)}\n"
                if logger:
                    logger.log(msg)
                else:
                    print(msg)
                raise
        
        msg = f"{'='*80}"
        msg += f"\nAll {len(banks)} banks processed and ready for training!\n"
        if logger:
            logger.log(msg)
        else:
            print(msg)
        
        return banks


# Example usage - Complete Federated Learning Workflow
if __name__ == "__main__":
    # Initialize result logger
    logger = ResultLogger('results.txt')
    
    try:
        # Log start time
        start_time = datetime.now()
        logger.log(f"AML Model Training Session Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"{'='*80}\n")
        
        # Define categorical features (optional - will auto-detect object columns)
        categorical_features = []
        
        # ========================================================================
        # PHASE 1: SETUP BANKS
        # ========================================================================
        
        logger.log(f"\n{'='*80}")
        logger.log("PHASE 1: SETUP BANKS")
        logger.log(f"{'='*80}\n")
        
        banks = AMLModel.setup_federated_banks(
            filepaths=[
                'clean_data\large_bank_HI_transactions_preprocessed.parquet',
                'clean_data\medium_bank_LI_transactions_preprocessed.parquet',
                'clean_data\small_bank_HI_transactions_preprocessed.parquet'
            ],
            bank_names=['Large_Bank', 'Medium_Bank', 'Small_Bank'],
            categorical_features=categorical_features,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            random_state=42,
            verbose=False,
            logger=logger
        )
        
        # ========================================================================
        # PHASE 2: HYPERPARAMETER TUNING ON LARGEST BANK ONLY
        # ========================================================================
        
        logger.log(f"\n{'='*80}")
        logger.log("PHASE 2: HYPERPARAMETER TUNING ON LARGEST BANK")
        logger.log("Finding best hyperparameters using largest bank's data only")
        logger.log(f"{'='*80}\n")
        
        # Identify the largest bank by training set size
        largest_bank = max(banks, key=lambda b: len(b.X_train))
        logger.log(f"Largest bank identified: {largest_bank.bank_name} with {len(largest_bank.X_train)} training samples\n")
        
        # Grid search parameters
        param_grid = {
            'learning_rate': [0.01,0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [50, 100],
            'num_leaves': [31, 50]
        }
        
        # Perform grid search only on the largest bank
        best_params, _ = largest_bank.local_grid_search(param_grid, logger=logger)
        
        # ========================================================================
        # PHASE 3: TRAIN LOCAL MODELS WITH BEST PARAMETERS
        # ========================================================================
        
        logger.log(f"\n{'='*80}")
        logger.log("PHASE 3: TRAIN LOCAL MODELS")
        logger.log(f"Training local models for all banks using best params: {best_params}")
        logger.log(f"{'='*80}\n")
        
        for bank in banks:
            bank.best_params = best_params
            bank.train_final_local_model(logger=logger)
        
        # ========================================================================
        # PHASE 4: FEDERATED TRAINING (Sequential)
        # ========================================================================
        
        logger.log(f"\n{'='*80}")
        logger.log("PHASE 4: FEDERATED TRAINING (Sequential)")
        logger.log("Train model sequentially on each bank's data")
        logger.log(f"{'='*80}\n")
        
        global_model = AMLModel.federated_training_sequential(
            banks, best_params, logger=logger
        )
        
        # Save the global model
        global_model.save_model('best_global_federated_model.pkl', logger=logger)
        
        # ========================================================================
        # PHASE 5: COMPARISON
        # ========================================================================
        
        logger.log(f"\n{'='*80}")
        logger.log("PHASE 5: LOCAL vs GLOBAL COMPARISON")
        logger.log(f"{'='*80}\n")
        
        comparison = AMLModel.compare_local_vs_global(banks, global_model, logger=logger)
        
        # Log completion
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.log(f"\n{'='*80}")
        logger.log("✅ FEDERATED LEARNING COMPLETE!")
        logger.log(f"{'='*80}")
        logger.log(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.log(f"Duration: {duration}")
        logger.log(f"\nBest hyperparameters (from {largest_bank.bank_name}): {best_params}")
        logger.log(f"Global federated model saved to: best_global_federated_model.pkl")
        logger.log(f"All results saved to: results.txt")
        
    except Exception as e:
        logger.log(f"\n❌ ERROR: {str(e)}")
        import traceback
        logger.log(traceback.format_exc())
    finally:
        logger.close()