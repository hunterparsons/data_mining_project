from data.download import download
from data.traditional_processing import traditional_processer
from data.deep_model_processing import process_deep_network
from data.data_2026 import get_2026_inference_data, get_2026_tournament_teams
from data.team_mapping import get_team_mapping
from models.rf import RF
from models.xgb import XGB
from models.lr import LR
from models.deep_model import DeepModel
from engine.bracket_engine import BracketEngine

from sklearn.model_selection import train_test_split
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_traditional_data(data_path):
    X, X_scaled, y, scaler = traditional_processer(data_path)

    X_train_u, X_val_u, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_s, X_val_s, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    return X_train_u, X_train_s, X_val_u, X_val_s, y_train, y_val, scaler

def get_deep_data(data_path):
    X_tensor, y_tensor = process_deep_network(data_path)
    return train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

def train_traditional_models(X_train_u, X_train_s, X_val_u, X_val_s, y_train, y_val):
    print("Training Random Forest...")
    rf = RF(X_train_u, y_train, X_val_u, y_val)

    print("Training XGBoost...")
    xgb = XGB(X_train_u, y_train, X_val_u, y_val)

    print("Training Logistic Regression...")
    lr = LR(X_train_s, y_train, X_val_s, y_val)

    return lr, xgb, rf

def print_results(lr, xgb, rf, dm):
    results = {}

    results['Random Forest'] = rf.get_performance()
    results['XGBoost'] = xgb.get_performance()
    results['Logistic Regression'] = lr.get_performance()
    results['Deep Model'] = dm.get_performance()

    print("\n--- Model Comparison ---")
    for name, metrics in results.items():
        print(f"{name}: LogLoss = {metrics['LogLoss']:.4f} | Accuracy = {metrics['Acc']:.4f}")

if __name__ == '__main__':
    data_path = download()

    X_train_u, X_train_s, X_val_u, X_val_s, y_train, y_val, logistic_scaler = get_traditional_data(data_path)
    X_train_t, X_val_t, y_train_t, y_val_t = get_deep_data(data_path)

    lr, xgb, rf = train_traditional_models(X_train_u, X_train_s, X_val_u, X_val_s, y_train, y_val)

    print("Training Deep Model (1D CNN + Transformer)")
    dm = DeepModel(X_train_t, y_train_t, X_val_t, y_val_t)

    print_results(lr, xgb, rf, dm)

    print("\n" + "="*40)
    print(" INITIALIZING 2026 SIMULATIONS")
    print("="*40)

    stats_2026 = get_2026_inference_data(data_path)
    starting_64_teams = get_2026_tournament_teams(data_path)
    
    team_names = get_team_mapping(data_path)

    lr_engine = BracketEngine(model=lr.model, season_stats_2026=stats_2026, scaler=logistic_scaler, team_mapping=team_names)
    xgb_engine = BracketEngine(model=xgb.model, season_stats_2026=stats_2026, scaler=None, team_mapping=team_names)
    rf_engine = BracketEngine(model=rf.model, season_stats_2026=stats_2026, scaler=None, team_mapping=team_names)
    dm_engine = BracketEngine(model=dm.model, season_stats_2026=stats_2026, scaler=None, team_mapping=team_names)

    print("\n[LOGISTIC REGRESSION]")
    lr_engine.run_monte_carlo(starting_64_teams, num_simulations=100)

    print("\n[XGBOOST]")
    xgb_engine.run_monte_carlo(starting_64_teams, num_simulations=100)

    print("\n[RANDOM FOREST]")
    rf_engine.run_monte_carlo(starting_64_teams, num_simulations=100)

    print("\n[DEEP MODEL]")
    rf_engine.run_monte_carlo(starting_64_teams, num_simulations=100)