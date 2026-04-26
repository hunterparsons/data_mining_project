import pandas as pd
def aggregate(season_stats_df: pd.DataFrame):
    # Break rows down into winners in losers
    winning_stats = season_stats_df.copy()
    winning_stats = winning_stats[['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WFGM', 'LFGM', 'WFGA', 'LFGA', 'WTO', 'LTO']]
    winning_stats.columns = ['Season', 'DayNum', 'TeamID', 'Points', 'OppTeamID', 'OppPoints', 'FGM', 'OppFGM', 'FGA', 'OppFGA', 'TO', 'OppTO']
    winning_stats['Wins'] = 1

    losing_stats = season_stats_df.copy()
    losing_stats = losing_stats[['Season', 'DayNum', 'LTeamID', 'LScore', 'WTeamID', 'WScore', 'LFGM', 'WFGM', 'LFGA', 'WFGA', 'LTO', 'WTO']]
    losing_stats.columns = ['Season', 'DayNum', 'TeamID', 'Points', 'OppTeamID', 'OppPoints', 'FGM', 'OppFGM', 'FGA', 'OppFGA', 'TO', 'OppTO']
    losing_stats['Wins'] = 0

    # Combine these back together
    all_games = pd.concat([winning_stats,losing_stats], ignore_index=True)

    # Season averages
    season_stats = all_games.groupby(['Season', 'TeamID']).agg(
        TotalGames=('TeamID', 'count'),
        TotalWins=('Wins', 'sum'),
        AvgPoints=('Points', 'mean'),
        AvgOppPoints=('OppPoints', 'mean'),
        AvgFGM=('FGM', 'mean'),
        AvgFGA=('FGA', 'mean'),
        AvgTO=('TO', 'mean')
    ).reset_index()

    # Advanced metrics
    season_stats['WinPct'] = season_stats['TotalWins'] / season_stats['TotalGames']
    season_stats['FGPct'] = season_stats['AvgFGM'] / season_stats['AvgFGA']
    season_stats['PointDiff'] = season_stats['AvgPoints'] - season_stats['AvgOppPoints']

    return season_stats