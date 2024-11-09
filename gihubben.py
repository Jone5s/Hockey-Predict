# -*- coding: utf-8 -*-
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time
import re
import pandas as pd

options = Options()
options.add_argument('disable-infobars')
options.add_argument('--incognito')
options.add_argument("start-maximized")

# Use Service to handle ChromeDriver path
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

URL = 'https://www.nhl.com/stats/teams?aggregate=0&reportType=game&seasonFrom=20232024&seasonTo=20242025&dateFromSeason&gameType=2&filter=gamesPlayed,gte,1&sort=a_gameDate&page=0&pageSize=100'
team = []
date = []
points = []
RW = []
ROW = []
SOW = []
goals = []
goals_against = []
power_play = []
penalty_kill = []
net_ppp = []
net_pkp = []
shots = []
shots_a = []
FOWp = []

for i in range(0, 29):
    driver.get(URL)
    time.sleep(5)  # Give time for the page to load fully

    # Locate rows using Selenium
    rows = driver.find_elements(By.CSS_SELECTOR, 'tr.sc-dSTloc.lpdNXV.rt-tr.null')

    for row in rows:
        all_data = row.find_elements(By.TAG_NAME, 'td')
        
        # Ensure there are enough columns to avoid index errors
        if len(all_data) > 23:
            team.append(all_data[1].text.strip())
            date.append(all_data[2].text.strip())
            points.append(all_data[8].text.strip())
            RW.append(all_data[10].text.strip())
            ROW.append(all_data[11].text.strip())
            SOW.append(all_data[12].text.strip())
            goals.append(all_data[13].text.strip())
            goals_against.append(all_data[14].text.strip())
            power_play.append(all_data[17].text.strip())
            penalty_kill.append(all_data[18].text.strip())
            net_ppp.append(all_data[19].text.strip())
            net_pkp.append(all_data[20].text.strip())
            shots.append(all_data[21].text.strip())
            shots_a.append(all_data[22].text.strip())
            FOWp.append(all_data[23].text.strip())
    
    # Update URL for pagination
    URL = re.sub(f'&page={i}', f'&page={i+1}', URL)

driver.quit()

# Create DataFrame and save to CSV
df = pd.DataFrame(zip(team, date, points, RW, ROW, SOW, goals, goals_against, power_play, penalty_kill, net_ppp, net_pkp, shots, shots_a, FOWp),
    columns=['team', 'date', 'points', 'RW', 'ROW', 'SOW', 'goals', 'goals_against', 'power_play', 'penalty_kill', 'net_ppp', 'net_pkp', 'shots', 'shots_against', 'FOWp'])

df.to_csv('nhl_stats.csv', index=False)
