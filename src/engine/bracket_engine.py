import numpy as np
from collections import Counter

class BracketEngine:
    def __init__(self, model, season_stats_2026, scaler=None, team_mapping=None):
        self.model = model
        self.stats = season_stats_2026
        self.scaler = scaler
        self.team_mapping = team_mapping or {}
        self.features = [
            'AdjWinPct', 'FGPct', 'AdjPointDiff', 'AvgPoints', 'AvgOppPoints', 
            'AvgFGM', 'AvgFGA', 'AvgTO', 'ConfStrengthIndex', 'Seed', 'Rank'
        ]
        
    def get_name(self, team_id):
        return self.team_mapping.get(team_id, str(team_id))

    def get_win_probability(self, team_a, team_b):
        stats_a = self.stats[self.stats['TeamID'] == team_a].iloc[0]
        stats_b = self.stats[self.stats['TeamID'] == team_b].iloc[0]
        
        diff_row = [stats_a[f] - stats_b[f] for f in self.features]
        X_input = np.array([diff_row])
        
        if self.scaler is not None:
            X_input = self.scaler.transform(X_input)

        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_input)[0, 1]
        
        return self.model.predict(X_input, verbose=0)[0][0]
            

    def simulate_single_bracket(self, starting_teams, verbose=False):
        current_round = starting_teams.copy()
        
        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                team_a = current_round[i]
                team_b = current_round[i+1]
                
                prob = self.get_win_probability(team_a, team_b)
                
                winner = team_a if np.random.rand() < prob else team_b

                if verbose:
                    name_a = self.get_name(team_a)
                    name_b = self.get_name(team_b)
                    name_winner = self.get_name(winner)
                    print(f"{name_a} vs {name_b} | Winner: {name_winner} ({prob:.1%})")
                
                next_round.append(winner)
            
            current_round = next_round
            
        if verbose:
            print(f"Tournament Champion: {current_round[0]}")
            
        return current_round[0]

    def run_monte_carlo(self, starting_teams, num_simulations=100):
        print(f"\nRunning Monte Carlo Simulation ({num_simulations} runs)...")
        champions = []
        
        for _ in range(num_simulations):
            champ = self.simulate_single_bracket(starting_teams, verbose=False)
            champions.append(champ)
            
        champ_counts = Counter(champions)
        sorted_champs = champ_counts.most_common()
        
        print("--- Championship Probabilities ---")
        for team, count in sorted_champs:
            win_pct = (count / num_simulations) * 100
            print(f"{self.get_name(team)}: {win_pct:.1f}% ({count} simulated wins)")

        return sorted_champs