import numpy as np
from collections import Counter, defaultdict

ROUND_LABEL_BY_FIELD_SIZE = {
    32: "Round of 32",
    16: "Sweet 16",
    8: "Elite Eight",
    4: "Final Four",
    2: "Championship",
    1: "Champion",
}

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
            
        return self.model.predict_proba(X_input)[0, 1]

    def simulate_single_bracket(self, starting_teams, is_stochastic=True, verbose=False):
        """Simulates one full 64-team tournament. Returns a list of advancers per round
        (e.g. [round_of_32, sweet_16, elite_8, final_four, championship, champion])."""
        current_round = starting_teams.copy()
        rounds_advancers = []

        while len(current_round) > 1:
            next_round = []
            for i in range(0, len(current_round), 2):
                team_a = current_round[i]
                team_b = current_round[i+1]

                prob = self.get_win_probability(team_a, team_b)

                if is_stochastic:
                    # Roll a random number between 0 and 1
                    winner = team_a if np.random.rand() < prob else team_b
                else:
                    # Chalk: Highest probability always wins
                    winner = team_a if prob > 0.5 else team_b

                if verbose:
                    name_a = self.get_name(team_a)
                    name_b = self.get_name(team_b)
                    name_winner = self.get_name(winner)
                    print(f"{name_a} vs {name_b} | Winner: {name_winner} ({prob:.1%})")

                next_round.append(winner)

            rounds_advancers.append(next_round)
            current_round = next_round

        if verbose:
            print(f"Tournament Champion: {current_round[0]}")

        return rounds_advancers

    def run_monte_carlo(self, starting_teams, num_simulations=100):
        """Runs the stochastic bracket N times and reports per-round advancement frequencies."""
        print(f"\nRunning Monte Carlo Simulation ({num_simulations} runs)...")

        round_counts = defaultdict(Counter)

        for _ in range(num_simulations):
            rounds_advancers = self.simulate_single_bracket(starting_teams, is_stochastic=True, verbose=False)
            for advancers in rounds_advancers:
                label = ROUND_LABEL_BY_FIELD_SIZE.get(len(advancers), f"Round of {len(advancers)}")
                for team in advancers:
                    round_counts[label][team] += 1

        sorted_champs = round_counts["Champion"].most_common()

        print("--- Championship Probabilities ---")
        for team, count in sorted_champs:
            win_pct = (count / num_simulations) * 100
            print(f"{self.get_name(team)}: {win_pct:.1f}% ({count} simulated wins)")

        print("\n--- Per-Round Advancement Probabilities ---")
        for round_label in ["Round of 32", "Sweet 16", "Elite Eight", "Final Four", "Championship"]:
            if round_label not in round_counts:
                continue
            print(f"\n[{round_label}]")
            for team, count in round_counts[round_label].most_common():
                pct = (count / num_simulations) * 100
                print(f"  {self.get_name(team)}: {pct:.1f}% ({count})")

        return sorted_champs