import requests
import pandas as pd

# Define a function to fetch team stats from the NHL API

def fetch_team_stats():
    url = "https://api.nhle.com/stats/rest/en/team/summary?sort=shotsForPerGame&cayenneExp=seasonId=20232024%20and%20gameTypeId=2"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching team stats: {e}")
        return None

# Function to parse team stats and extract key information
def parse_team_stats(team_data):
    teams_info = []
    if team_data and 'data' in team_data:
        for team in team_data['data']:
            team_info = {
                'team_name': team['teamFullName'],
                'games_played': team['gamesPlayed'],
                'goals_for': team['goalsFor'],
                'goals_against': team['goalsAgainst'],
                'shots_for_per_game': team['shotsForPerGame'],
                'shots_against_per_game': team['shotsAgainstPerGame'],
                'wins': team['wins'],
                'losses': team['losses'],
                'ot_losses': team['otLosses'],
                'points': team['points'],
                'faceoff_win_pct': team['faceoffWinPct'],
                'power_play_pct': team['powerPlayPct'],
                'penalty_kill_pct': team['penaltyKillPct']
            }
            teams_info.append(team_info)
    return teams_info

if __name__ == "__main__":
    # Fetch team stats for the 2023-2024 season
    team_data = fetch_team_stats()
    if team_data:
        # Parse the team data
        teams_info = parse_team_stats(team_data)
        
        # Convert collected data to a DataFrame and save to CSV for future analysis
        if teams_info:
            df = pd.DataFrame(teams_info)
            df.to_csv("nhl_team_stats.csv", index=False)
            print("Data collection complete, saved to nhl_team_stats.csv")
        else:
            print("No team stats found.")
    else:
        print("Failed to fetch team stats.")
