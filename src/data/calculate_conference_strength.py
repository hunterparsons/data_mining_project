import pandas as pd
import os

def calculate_conference_strength(path):
    conferences = pd.read_csv(os.path.join(path, 'MTeamConferences.csv'))
    massey = pd.read_csv(os.path.join(path, 'MMasseyOrdinals.csv'))

    final_week_ranks = massey[massey['RankingDayNum'] >= 128]
    
    team_consensus = final_week_ranks.groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()

    team_conf_ranks = pd.merge(conferences, team_consensus, on=['Season', 'TeamID'], how='left')

    conf_strength = team_conf_ranks.groupby(['Season', 'ConfAbbrev'])['OrdinalRank'].mean().reset_index()
    conf_strength.rename(columns={'OrdinalRank': 'ConfAverageRank'}, inplace=True)

    team_conf_strength = pd.merge(conferences, conf_strength, on=['Season', 'ConfAbbrev'], how='left')

    team_conf_strength['ConfStrengthIndex'] = 365 - team_conf_strength['ConfAverageRank']

    return team_conf_strength[['Season', 'TeamID', 'ConfAbbrev', 'ConfStrengthIndex']]