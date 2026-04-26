from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

from data.calculate_conference_strength import calculate_conference_strength

import pandas as pd
import numpy as np
import os

def extract_game_sequences(regular_season_df: pd.DataFrame, n_games: int = 15) -> pd.DataFrame:
    df1 = regular_season_df[['Season', 'DayNum', 'WTeamID', 'WScore', 'LScore']].copy()
    df1.columns = ['Season', 'DayNum', 'TeamID', 'Pts', 'OppPts']
    
    df2 = regular_season_df[['Season', 'DayNum', 'LTeamID', 'LScore', 'WScore']].copy()
    df2.columns = ['Season', 'DayNum', 'TeamID', 'Pts', 'OppPts']
    
    all_games = pd.concat([df1, df2]).sort_values(['Season', 'TeamID', 'DayNum'])
    all_games['PtDiff'] = all_games['Pts'] - all_games['OppPts']
    
    features = ['PtDiff', 'Pts']
    
    sequences = all_games.groupby(['Season', 'TeamID']).apply(
        lambda x: x.tail(n_games)[features].values.tolist()
    ).reset_index(name='Sequence')
    
    return sequences

def process_deep_network(path: str, n_games: int = 15):
    reg_season = pd.read_csv(os.path.join(path, 'MRegularSeasonDetailedResults.csv'))
    tourney_results = pd.read_csv(os.path.join(path, 'MNCAATourneyDetailedResults.csv'))

    sequences_df = extract_game_sequences(reg_season, n_games)
    conf_scores = calculate_conference_strength(path)

    df_win = tourney_results[['Season', 'WTeamID', 'LTeamID']].copy()
    df_win['Target'] = 1
    df_win.columns = ['Season', 'TeamA', 'TeamB', 'Target']

    df_lose = tourney_results[['Season', 'LTeamID', 'WTeamID']].copy()
    df_lose['Target'] = 0
    df_lose.columns = ['Season', 'TeamA', 'TeamB', 'Target']

    matches = pd.concat([df_win, df_lose]).sample(frac=1.0, random_state=42).reset_index(drop=True)

    matches = matches.merge(sequences_df, left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    matches = matches.rename(columns={'Sequence': 'Seq_A'}).drop('TeamID', axis=1)
    
    matches = matches.merge(sequences_df, left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    matches = matches.rename(columns={'Sequence': 'Seq_B'}).drop('TeamID', axis=1)

    matches = matches.merge(conf_scores, left_on=['Season', 'TeamA'], right_on=['Season', 'TeamID'])
    matches = matches.rename(columns={'ConfStrengthIndex': 'Conf_A'}).drop(['TeamID', 'ConfAbbrev'], axis=1)

    matches = matches.merge(conf_scores, left_on=['Season', 'TeamB'], right_on=['Season', 'TeamID'])
    matches = matches.rename(columns={'ConfStrengthIndex': 'Conf_B'}).drop(['TeamID', 'ConfAbbrev'], axis=1)

    X_tensor = []
    for _, row in matches.iterrows():
        seq_A = pad_sequences([row['Seq_A']], maxlen=n_games, dtype='float32', padding='pre')[0]
        seq_B = pad_sequences([row['Seq_B']], maxlen=n_games, dtype='float32', padding='pre')[0]
        
        diff_seq = seq_A - seq_B
        
        conf_diff = row['Conf_A'] - row['Conf_B']
        
        conf_diff_array = np.full((n_games, 1), conf_diff)
        
        combined_seq = np.hstack((diff_seq, conf_diff_array))

        X_tensor.append(combined_seq) #

    X_tensor = np.array(X_tensor)
    y = matches['Target'].values

    samples, timesteps, features = X_tensor.shape
    X_tensor_reshaped = X_tensor.reshape(-1, features)
    
    scaler = StandardScaler()
    X_scaled_reshaped = scaler.fit_transform(X_tensor_reshaped)
    X_final_tensor = X_scaled_reshaped.reshape(samples, timesteps, features)

    return X_final_tensor, y