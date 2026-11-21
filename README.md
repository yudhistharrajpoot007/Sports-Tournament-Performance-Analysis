# Sports Tournament Match Statistics Analysis

An end-to-end Python data analysis pipeline exploring multi-season sports tournament match data. Built using **Pandas**, **NumPy**, and **Object-Oriented Programming (OOP)** to clean raw match records, engineer temporal features, evaluate team and player performance, and detect statistical outliers.

---

## Files Included

* **`28_tournament_match_stats.csv`** — Original raw dataset (609 rows × 10 columns).
* **`28_tournament_match_stats_extended.csv`** — Extended raw dataset with duplicates, missing values, and 5 additional feature columns (`Player_ID`, `Assists`, `Fouls`, `Venue`, `Attendance`) (1,009 rows × 15 columns).
* **`28_tournament_match_stats_extended_clean.csv`** — Cleaned, deduplicated, median-imputed, and chronologically ordered dataset (994 rows × 15 columns).
* **`tournament_analysis.py`** — Standalone Python script implementing the full 15-step analysis pipeline.
* **`tournament_analysis.ipynb`** — Jupyter / Google Colab notebook version of the analysis.
* **`requirements.txt`** — Required Python packages (`pandas`, `numpy`).

---

## Analysis Pipeline & Key Features

1. **Data Inspection & Cleaning:** Validates schemas, removes duplicate `Match_ID` records (`keep='first'`), and imputes missing values in `Player_Points` and `Match_Duration_Mins` using column medians.
2. **Temporal Feature Engineering:** Parses `Match_Date` into `datetime` objects and extracts `Month` and `Weekday` attributes.
3. **Tournament & Team Performance Metrics:** Aggregates total unique matches, total points, average match duration, team rankings, win percentages, and scoring averages across match types and tournament rounds.
4. **Python Data Structures & NumPy Statistics:** Uses `tuple`, `set`, and `dict` structures alongside NumPy sample statistics (`mean`, `median`, `std` with `ddof=1`).
5. **IQR Outlier Detection & Performance Tiering:** Flags match-level and player-level outliers via the $Q_3 + 1.5 \times \text{IQR}$ rule and classifies match performances into four tiers (`Low`, `Average`, `Strong`, `Elite`).
6. **Object-Oriented Modeling:** Implements a `Player` class to encapsulate player attributes and generate performance summaries.

---

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis script:
   ```bash
   python tournament_analysis.py
   # Or specify a different dataset file in the folder:
   python tournament_analysis.py 28_tournament_match_stats.csv
   python tournament_analysis.py 28_tournament_match_stats_extended_clean.csv
   ```
