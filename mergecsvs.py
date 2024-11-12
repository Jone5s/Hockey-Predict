import pandas as pd
import glob

# List of Metropolitan Division team codes
metro_teams = ["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"]

# Path to your CSV files (update as necessary)
file_paths = glob.glob('./csvs/*.csv')  # Replace with the actual path

# Initialize an empty list to store DataFrames
all_data = []

# Loop over each file and process it
for file_path in file_paths:
    # Load the CSV file
    df = pd.read_csv(file_path)
    
    # Filter for games against Metropolitan Division teams
    df_filtered = df[df['opposingTeam'].isin(metro_teams)]
    
    # Append the filtered DataFrame to the list
    all_data.append(df_filtered)

# Concatenate all DataFrames into a single DataFrame
merged_data = pd.concat(all_data, ignore_index=True)

# Save the merged data
output_path = './metro_teams_merged.csv'  # Replace with your desired path
merged_data.to_csv(output_path, index=False)

print(f"Merged data saved to {output_path}")
