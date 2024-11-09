import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import numpy as np

# Step 1: Load and Explore the Data
data = pd.read_csv('nhl_team_stats.csv')

# Step 2: Feature Engineering
team_pairs = list(itertools.combinations(data['team_name'], 2))

def create_match_features(team_a, team_b, data):
    team_a_stats = data[data['team_name'] == team_a].iloc[0]
    team_b_stats = data[data['team_name'] == team_b].iloc[0]
    
    match_features = {
        'team_a': team_a,
        'team_b': team_b,
        'games_played_diff': team_a_stats['games_played'] - team_b_stats['games_played'],
        'goals_for_diff': team_a_stats['goals_for'] - team_b_stats['goals_for'],
        'goals_against_diff': team_a_stats['goals_against'] - team_b_stats['goals_against'],
        'shots_for_per_game_diff': team_a_stats['shots_for_per_game'] - team_b_stats['shots_for_per_game'],
        'shots_against_per_game_diff': team_a_stats['shots_against_per_game'] - team_b_stats['shots_against_per_game'],
        'wins_diff': team_a_stats['wins'] - team_b_stats['wins'],
        'losses_diff': team_a_stats['losses'] - team_b_stats['losses'],
        'faceoff_win_pct_diff': team_a_stats['faceoff_win_pct'] - team_b_stats['faceoff_win_pct'],
        'power_play_pct_diff': team_a_stats['power_play_pct'] - team_b_stats['power_play_pct'],
        'penalty_kill_pct_diff': team_a_stats['penalty_kill_pct'] - team_b_stats['penalty_kill_pct'],
        'recent_performance_diff': (team_a_stats['points'] / team_a_stats['games_played']) - (team_b_stats['points'] / team_b_stats['games_played'])
    }
    
    return match_features

# Randomize the order of team pairs to balance any biases
randomized_team_pairs = team_pairs + [(b, a) for a, b in team_pairs]
np.random.shuffle(randomized_team_pairs)

matches = [create_match_features(team_a, team_b, data) for team_a, team_b in randomized_team_pairs]
matches_df = pd.DataFrame(matches)

# Step 3: Define Labels (1 for win, 0 for loss based on goals_for_diff)
def label_match(row):
    return 1 if row['goals_for_diff'] > 0 else 0  # Simplify to binary outcome

matches_df['label'] = matches_df.apply(label_match, axis=1)

# Step 4: Balance the Dataset by Resampling
majority = matches_df[matches_df.label == 1]
minority = matches_df[matches_df.label == 0]

minority_upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=42)
balanced_df = pd.concat([majority, minority_upsampled])

# Step 5: Prepare Features and Labels
scaler = RobustScaler()
X = pd.DataFrame(scaler.fit_transform(balanced_df.drop(['team_a', 'team_b', 'label'], axis=1)), columns=balanced_df.drop(['team_a', 'team_b', 'label'], axis=1).columns)
y = balanced_df['label']

# Step 6: Train-Test Split (Using larger test set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Step 7: Train a Simpler Logistic Regression Model
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)

# Step 8: Predict Betting Odds for a Specific Match with Temperature Scaling
def predict_match(team_a, team_b, data, model, temperature=5.0):
    match_features = create_match_features(team_a, team_b, data)
    match_features_df = pd.DataFrame([match_features]).drop(['team_a', 'team_b'], axis=1)
    probabilities = model.predict_proba(match_features_df)[0]
    
    # Print predicted probabilities for analysis
    print(f'Predicted Probabilities: {probabilities}')
    
    # Apply temperature scaling to soften the probabilities
    probabilities = np.power(probabilities, 1 / temperature)
    probabilities = np.clip(probabilities, 0.1, 0.9)  # Add minimum value to avoid extreme odds
    probabilities /= probabilities.sum()  # Normalize probabilities to sum to 1
    
    odds = {
        '1 (Team A Win)': round(1 / probabilities[1], 2),
        '2 (Team B Win)': round(1 / probabilities[0], 2)
    }
    return odds

# Example Prediction
team_a = 'Washington Capitals'
team_b = 'Philadelphia Flyers'
odds = predict_match(team_a, team_b, data, log_reg)
print(f'Predicted Odds for {team_a} vs {team_b}: {odds}')
