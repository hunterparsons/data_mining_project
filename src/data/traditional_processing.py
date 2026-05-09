import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from data.calculate_conference_strength import calculate_conference_strength
from data.traditional_aggregate import aggregate
from data.data_2026 import get_pom_rankings, get_tournament_seeds

def traditional_processer(path):
    reg_season = pd.read_csv(os.path.join(path, 'MRegularSeasonDetailedResults.csv'))
    tourney_results = pd.read_csv(os.path.join(path, 'MNCAATourneyDetailedResults.csv'))

    season_stats = aggregate(reg_season)
    
    conf_scores = calculate_conference_strength(path)
    seeds_df = get_tournament_seeds(path)
    pom_df = get_pom_rankings(path)

    season_stats = pd.merge(season_stats, conf_scores, on=['Season', 'TeamID'], how='left')
    season_stats = pd.merge(season_stats, seeds_df, on=['Season', 'TeamID'], how='left')
    season_stats = pd.merge(season_stats, pom_df, on=['Season', 'TeamID'], how='left')

    season_stats['Seed'] = season_stats['Seed'].fillna(17) # Unseeded teams act as 17-seeds
    season_stats['Rank'] = season_stats['Rank'].fillna(362) # Unranked teams put at the bottom
    
    season_stats['AdjPointDiff'] = season_stats['PointDiff'] * season_stats['ConfStrengthIndex']
    season_stats['AdjWinPct'] = season_stats['WinPct'] * season_stats['ConfStrengthIndex']

    win_df = tourney_results[['Season', 'WTeamID', 'LTeamID']].copy()
    win_df['Target'] = 1
    win_df = win_df.merge(season_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID']).drop('TeamID', axis=1)
    win_df = win_df.merge(season_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], suffixes=('_TeamA', '_TeamB')).drop('TeamID', axis=1)

    lose_df = tourney_results[['Season', 'LTeamID', 'WTeamID']].copy()
    lose_df['Target'] = 0
    lose_df = lose_df.merge(season_stats, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID']).drop('TeamID', axis=1)
    lose_df = lose_df.merge(season_stats, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], suffixes=('_TeamA', '_TeamB')).drop('TeamID', axis=1)

    base_df = pd.concat([win_df, lose_df], ignore_index=True)

    features = [
        'AdjWinPct', 'FGPct', 'AdjPointDiff', 'AvgPoints', 'AvgOppPoints', 
        'AvgFGM', 'AvgFGA', 'AvgTO', 'ConfStrengthIndex', 'Seed', 'Rank'
    ]
    
    diff_cols = []
    for f in features:
        base_df[f'{f}_Diff'] = base_df[f'{f}_TeamA'] - base_df[f'{f}_TeamB']
        diff_cols.append(f'{f}_Diff')

    base_df = base_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    X = base_df[diff_cols].fillna(base_df[diff_cols].median())
    y = base_df['Target']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=diff_cols) 

    return X, X_scaled, y, scaler