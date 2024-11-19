import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import resample

# Load the filtered dataset
file_path = './metro_teams_filtered.csv'  # Update with your file path if different
data = pd.read_csv(file_path)

# Create the 'outcome' column based on goalsFor and goalsAgainst
data['outcome'] = (data['goalsFor'] > data['goalsAgainst']).astype(int)

# Select relevant features for prediction
features = [
    'scoreAdjustedShotsAttemptsAgainst', 
    'unblockedShotAttemptsAgainst', 
    'dZoneGiveawaysAgainst',
    'xGoalsFromxReboundsOfShotsAgainst', 
    'reboundxGoalsAgainst'
]  # Modify as needed
X = data[features]
y = data['outcome']

# Combine features and target for balancing
combined_data = pd.concat([X, y], axis=1)

# Separate majority and minority classes
majority_class = combined_data[combined_data['outcome'] == 0]
minority_class = combined_data[combined_data['outcome'] == 1]

# Upsample the minority class
minority_upsampled = resample(
    minority_class,
    replace=True,  # Sample with replacement
    n_samples=len(majority_class),  # Match majority class size
    random_state=42  # For reproducibility
)

# Combine upsampled minority class with majority class
balanced_data = pd.concat([majority_class, minority_upsampled])

# Separate features and target from balanced data
X_balanced = balanced_data[features]
y_balanced = balanced_data['outcome']

# Split balanced data into training and testing sets
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(
    X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
)

# Train the model
model_balanced = RandomForestClassifier(random_state=42)
model_balanced.fit(X_train_balanced, y_train_balanced)

# Predict outcomes on the test set
y_pred_balanced = model_balanced.predict(X_test_balanced)

# Evaluate the model
accuracy_balanced = accuracy_score(y_test_balanced, y_pred_balanced)
report_balanced = classification_report(y_test_balanced, y_pred_balanced)

# Print results
print("Balanced Dataset Model Evaluation")
print(f"Accuracy: {accuracy_balanced:.2f}")
print("\nClassification Report:\n")
print(report_balanced)

# Function to predict matchup outcome
def predict_matchup_fixed(team1_name, team2_name, data, model):
    """
    Predict the outcome of a matchup between two teams using historical game data.

    Parameters:
        team1_name (str): Name of the first team.
        team2_name (str): Name of the second team.
        data (pd.DataFrame): Dataset containing historical stats for all teams.
        model: Trained machine learning model.

    Returns:
        str: Percentual prediction for each team's likelihood of winning, normalized to sum up to 100%.
    """
    # Extract all games for Team 1 and calculate average stats
    team1_stats = data[data['playerTeam'] == team1_name][features].mean()
    # Extract all games for Team 2 and calculate average stats
    team2_stats = data[data['playerTeam'] == team2_name][features].mean()
    
    # Ensure stats are found for both teams
    if team1_stats.empty or team2_stats.empty:
        return "One or both teams not found in the dataset."

    # Combine stats into a DataFrame for prediction
    matchup_data = pd.DataFrame([team1_stats, team2_stats])

    # Use the model to predict probabilities
    probabilities = model.predict_proba(matchup_data)

    # Extract probabilities for winning (class 1) for each team
    team1_win_prob = probabilities[0][1]
    team2_win_prob = probabilities[1][1]

    # Normalize probabilities to sum up to 100%
    total_prob = team1_win_prob + team2_win_prob
    team1_win_prob_normalized = team1_win_prob / total_prob * 100
    team2_win_prob_normalized = team2_win_prob / total_prob * 100

    return (
        f"Prediction:\n"
        f"{team1_name} Win Probability: {team1_win_prob_normalized:.2f}%\n"
        f"{team2_name} Win Probability: {team2_win_prob_normalized:.2f}%"
    )

# Example usage
team1_name = "NYI"  # Replace with the desired team name
team2_name = "WSH"  # Replace with the desired team name

print("\nPrediction for matchup:")
print(predict_matchup_fixed(team1_name, team2_name, data, model_balanced))
