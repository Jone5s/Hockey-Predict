# NHL Matchup Betting Odds Predictor

This project is a **Streamlit** application that predicts betting odds for NHL matchups using a **Random Forest Classifier** trained on historical game data.

## Features
- Uses a dataset (`metro_teams_merged_filtered_all.csv`) to train a **Random Forest** model.
- Balances training data using **upsampling** to ensure fairness.
- Allows users to select two teams and calculates betting odds for:
  - **Team 1 win**
  - **Team 2 win**
  - **Draw** (adjusted probability)
- Displays odds in a simple UI using **Streamlit**.

## Installation & Setup
### Requirements
Ensure you have Python installed along with the following dependencies:

```sh
pip install streamlit pandas scikit-learn
```

### Running the App
1. Clone this repository and navigate to the project folder.
2. Ensure the dataset (`metro_teams_merged_filtered_all.csv`) is in the root directory.
3. Run the Streamlit app:

```sh
streamlit run webUI.py
```

## How It Works
1. **Data Preparation**
   - Reads the CSV file and processes match statistics.
   - Creates an `outcome` column (1 = win, 0 = loss).
   - Extracts relevant features for training.
   - Balances the dataset by upsampling the minority class.

2. **Model Training**
   - Splits data into training and testing sets.
   - Trains a **Random Forest Classifier** to predict match outcomes.

3. **Making Predictions**
   - Takes team names as input.
   - Computes feature averages for selected teams.
   - Predicts win probabilities and calculates betting odds.

4. **User Interface (Streamlit)**
   - Dropdowns for selecting teams.
   - Button to trigger prediction.
   - Displays the odds in an interactive format.

## Example Output
```
Team 1: Boston Bruins, Probability: 42.3%, Odds: 2.36
Draw: Probability: 25.1%, Odds: 3.80
Team 2: New York Rangers, Probability: 32.6%, Odds: 3.07
```

## Author
- **Aleksi Leppinen**
