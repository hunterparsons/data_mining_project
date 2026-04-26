import pandas as pd
import os

def get_team_mapping(data_path):
    teams_df = pd.read_csv(os.path.join(data_path, 'MTeams.csv'))
    return dict(zip(teams_df['TeamID'], teams_df['TeamName']))