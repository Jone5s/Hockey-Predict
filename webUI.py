import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

file_path = './metro_teams_merged_filtered_all.csv'
data = pd.read_csv(file_path)

# Create the 'outcome' column based on goalsFor and goalsAgainst
data['outcome'] = (data['goalsFor'] > data['goalsAgainst']).astype(int)

features = [
    'scoreAdjustedShotsAttemptsAgainst', 
    'unblockedShotAttemptsAgainst', 
    'dZoneGiveawaysAgainst',
    'xGoalsFromxReboundsOfShotsAgainst', 
    'reboundxGoalsAgainst',
    'xGoalsPercentage'
]

X = data[features]
y = data['outcome']

# Combine features and target for balancing
combined_data = pd.concat([X, y], axis=1)

# Separate majority and minority classes
majority_class = combined_data[combined_data['outcome'] == 0]
minority_class = combined_data[combined_data['outcome'] == 1]

minority_upsampled = resample(
    minority_class,
    replace=True,
    n_samples=len(majority_class),
    random_state=42
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

def predict_matchup_fixed_with_draw(team1_name, team2_name, data, model):
    team1_stats = data[data['playerTeam'] == team1_name][features].mean()
    team2_stats = data[data['playerTeam'] == team2_name][features].mean()
    
    if team1_stats.empty or team2_stats.empty:
        return "One or both teams not found in the dataset."
    
    matchup_data = pd.DataFrame([team1_stats, team2_stats])
    probabilities = model.predict_proba(matchup_data)
    
    team1_win_prob = probabilities[0][1]
    team2_win_prob = probabilities[1][1]
    
    total_prob = team1_win_prob + team2_win_prob
    draw_prob = (1 - total_prob) / 0.9  # Adjust draw probability
    
    team1_win_prob_normalized = team1_win_prob / total_prob * (1 - draw_prob) * 100
    team2_win_prob_normalized = team2_win_prob / total_prob * (1 - draw_prob) * 100
    draw_prob_normalized = draw_prob * 100
    
    # Calculate odds
    team1_decimal_odds = 1 / (team1_win_prob_normalized / 100)
    team2_decimal_odds = 1 / (team2_win_prob_normalized / 100)
    draw_decimal_odds = max(3.8, min(5, 1 / (draw_prob_normalized / 100)))
    
    # Return results as a dictionary
    return {
        "team1": {"name": team1_name, "prob": team1_win_prob_normalized, "odds": team1_decimal_odds},
        "team2": {"name": team2_name, "prob": team2_win_prob_normalized, "odds": team2_decimal_odds},
        "draw": {"prob": draw_prob_normalized, "odds": draw_decimal_odds},
    }


# Streamlit UI
st.title("NHL Matchup Betting Odds Predictor")

# Dropdowns for selecting teams
team_list = data['playerTeam'].unique()
team1_name = st.selectbox("Select Team 1", team_list)
team2_name = st.selectbox("Select Team 2", team_list)

if st.button("Calculate Outcome"):
    result = predict_matchup_fixed_with_draw(team1_name, team2_name, data, model_balanced)
    
    if isinstance(result, str):
        st.write(result)
    else:
        # Display odds in a simple way
        col1, col2, col3 = st.columns(3)
        with col1:
            st.button(f"{result['team1']['odds']:.2f}", disabled=True)
            st.caption(result['team1']['name'])
        with col2:
            st.button(f"{result['draw']['odds']:.2f}", disabled=True)
            st.caption("Draw")
        with col3:
            st.button(f"{result['team2']['odds']:.2f}", disabled=True)
            st.caption(result['team2']['name'])
            