from data.traditional_aggregate import aggregate
from data.calculate_conference_strength import calculate_conference_strength

import pandas as pd
import os

def get_tournament_seeds(data_path):
    seeds = pd.read_csv(os.path.join(data_path, 'MNCAATourneySeeds.csv'))
    seeds['Seed'] = seeds['Seed'].str.extract(r'(\d+)').astype(int)
    return seeds[['Season', 'TeamID', 'Seed']]

def get_pom_rankings(data_path):
    massey = pd.read_csv(os.path.join(data_path, 'MMasseyOrdinals.csv'))
    pom = massey[(massey['SystemName'] == 'POM') & (massey['RankingDayNum'] == 133)].copy()
    pom = pom.rename(columns={'OrdinalRank': 'Rank'})
    return pom[['Season', 'TeamID', 'Rank']]

def get_2026_inference_data(data_path):
    reg_season = pd.read_csv(os.path.join(data_path, 'MRegularSeasonDetailedResults.csv'))
    
    season_stats = aggregate(reg_season)
    
    conf_index = calculate_conference_strength(data_path)
    seeds_df = get_tournament_seeds(data_path)
    pom_df = get_pom_rankings(data_path)
    
    full_stats = pd.merge(season_stats, conf_index, on=['Season', 'TeamID'], how='left')
    full_stats = pd.merge(full_stats, seeds_df, on=['Season', 'TeamID'], how='left')
    full_stats = pd.merge(full_stats, pom_df, on=['Season', 'TeamID'], how='left')
    
    full_stats['Seed'] = full_stats['Seed'].fillna(17)
    full_stats['Rank'] = full_stats['Rank'].fillna(362)
    
    full_stats['AdjPointDiff'] = full_stats['PointDiff'] * full_stats['ConfStrengthIndex']
    full_stats['AdjWinPct'] = full_stats['WinPct'] * full_stats['ConfStrengthIndex']
    
    stats_2026 = full_stats[full_stats['Season'] == 2026]
    return stats_2026

def get_2026_tournament_teams(data_path):
    seeds = pd.read_csv(os.path.join(data_path, 'MNCAATourneySeeds.csv'))
    seeds_2026 = seeds[seeds['Season'] == 2026].copy()
    
    return seeds_2026['TeamID'].tolist()[:64]