# World Cup 2026 Predictor 🏆

A Monte Carlo simulation engine that estimates the probability 
of each nation winning the 2026 FIFA World Cup, built in Python 
and visualized through an interactive HTML dashboard.

## How it works

The simulator runs 1000 full tournament simulations, modelling 
the official FIFA 2026 format: 12 groups of 4 teams, 48 nations, 
104 matches total. Each match outcome is determined probabilistically 
based on weighted team strength scores and random variance.

**Team strength** is calculated from 5 metrics:
- FIFA ranking (25%)
- Average goals scored per match (25%)
- Average goals conceded per match (25%)
- Win rate over the last 12 months (15%)
- Knockout stage win rate (10%)

**Dynamic match factors** add realism to every simulation:
- Injury risk (5% per team per match → −10% strength)
- Red card risk (8% per team per match → −15% strength)
- Adverse weather (10% per match → −5% both teams)
- Home advantage for USA, Canada, Mexico (+8% strength)
- Inspired performance (3% per team per match → +12% strength)

## Dashboard

The interactive dashboard shows:
- **Group Stage** — final standings and match results for all 12 groups.
  Click on any group card to expand it and see the full standings 
  (points, GF, GS, qualification probability) and every match result 
  divided by matchday.
- **Knockout Bracket** — official FIFA 2026 bracket (M73–M104) with win probabilities
- **Win Probability** — bar chart of all nations ranked by probability of lifting the trophy

## Methodological Notes

**Group stage scores**
Match scores in the Group Stage tab are not real or predicted 
exact results — they are average goal tallies across all 1000 
simulations. Each simulation produces an integer scoreline 
(e.g. 1–2, 0–1, 3–0), sampled from a Poisson distribution 
calibrated on each team's historical goals-per-game average. 
The decimal figures shown are the mean of those 1000 outcomes, 
and should be read as expected goal output rather than a 
predicted final score.

**Knockout bracket percentages**
The percentage shown next to each team in the bracket represents 
the probability of that team reaching that specific match — 
i.e. in how many of the 1000 simulated tournaments that team 
made it to that round. It is not the head-to-head win probability 
for that fixture. Two teams in the same match can each show 
figures well below 50% because in many simulations a different 
pair of teams met in that slot entirely.

## Stack

Python 3 · pandas · numpy · vanilla HTML/CSS/JS

## Run locally

pip install pandas numpy
python predictor.py

Then open dashboard.html in your browser.

---

*Personal learning project built in a single day — the day before 
the 2026 World Cup kicked off — to explore Claude Code and Monte 
Carlo simulations. Data is estimated from publicly available FIFA 
rankings and recent international results. Not affiliated with FIFA or any football association*
