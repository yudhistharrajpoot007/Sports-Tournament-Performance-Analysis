import sys
from pathlib import Path
import pandas as pd
import numpy as np


def calculate_win_percentage(wins: int, matches_played: int) -> float:
    """Calculate win percentage rounded to two decimal places."""
    if matches_played == 0:
        return 0.0
    return round((wins / matches_played) * 100, 2)


class Player:
    """Represents a tournament player and their match performance."""

    def __init__(self, name: str, team: str, points: float):
        self.name = name
        self.team = team
        self.points = points

    def summary(self) -> str:
        return f"Player: {self.name} | Team: {self.team} | Points: {self.points:.0f}"


def main():
    # Looks for the CSV in the same directory by default (or accepts a custom path via CLI)
    default_path = Path(__file__).resolve().parent / "28_tournament_match_stats_extended.csv"
    file_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path

    if not file_path.exists():
        print(f"Error: Could not find '{file_path}'.")
        sys.exit(1)

    print(f"Successfully loaded file from: {file_path.resolve()}\n")

    # 1. Load the tournament dataset and inspect dimensions, columns, and data types
    df = pd.read_csv(file_path)
    print("--- 1. DATASET OVERVIEW ---")
    print(f"Dimensions (Rows, Columns): {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print("\nData Types:\n", df.dtypes)

    # 2. Check for missing values and duplicate Match_ID records
    print("\n--- 2. MISSING VALUES & DUPLICATES ---")
    print("Missing values per column:\n", df.isnull().sum())
    duplicate_count = df["Match_ID"].duplicated().sum()
    print(f"\nDuplicate Match_ID records found: {duplicate_count}")
    df = df.drop_duplicates(subset=["Match_ID"], keep="first").copy()
    print(f"Dimensions after removing duplicates: {df.shape}")

    # 3. Handle missing Player_Points or Match_Duration_Mins using median imputation
    points_median = df["Player_Points"].median()
    duration_median = df["Match_Duration_Mins"].median()
    df["Player_Points"] = df["Player_Points"].fillna(points_median)
    df["Match_Duration_Mins"] = df["Match_Duration_Mins"].fillna(duration_median)
    print("\n--- 3. MISSING VALUE IMPUTATION ---")
    print(f"Imputed Player_Points with median ({points_median}) and Match_Duration_Mins with median ({duration_median}).")
    print("Remaining nulls:", df.isnull().sum().sum())

    # 4. Convert Match_Date to datetime and extract month and weekday
    df["Match_Date"] = pd.to_datetime(df["Match_Date"])
    df["Month"] = df["Match_Date"].dt.month_name()
    df["Weekday"] = df["Match_Date"].dt.day_name()
    print("\n--- 4. DATE FEATURE EXTRACTION ---")
    print(df[["Match_ID", "Match_Date", "Month", "Weekday"]].head())

    # 5. Calculate total matches played, total points scored, and average duration
    total_matches = df["Match_ID"].nunique()
    total_points = df["Player_Points"].sum()
    avg_duration = df["Match_Duration_Mins"].mean()
    print("\n--- 5. OVERALL TOURNAMENT SUMMARY ---")
    print(f"Total Matches Played: {total_matches}")
    print(f"Total Points Scored: {total_points:.0f}")
    print(f"Average Match Length: {avg_duration:.2f} mins")

    # 6 & 9. Rank teams by total points & wins, and calculate win percentage
    team_summary = df.groupby("Team", as_index=False).agg(
        Matches_Played=("Match_ID", "nunique"),
        Total_Points=("Player_Points", "sum"),
        Wins=("Result", lambda s: (s == "Win").sum()),
        Draws=("Result", lambda s: (s == "Draw").sum()),
        Losses=("Result", lambda s: (s == "Loss").sum()),
    )
    team_summary["Win_Percentage"] = team_summary.apply(
        lambda row: calculate_win_percentage(row["Wins"], row["Matches_Played"]), axis=1
    )
    team_ranking = team_summary.sort_values(
        by=["Total_Points", "Wins"], ascending=[False, False]
    ).reset_index(drop=True)
    team_ranking.index += 1
    print("\n--- 6 & 9. TEAM RANKINGS & WIN PERCENTAGE ---")
    print(team_ranking.to_string())

    # 7. Identify the top 10 players by total points
    top_10_players = (
        df.groupby("Player_Name", as_index=False)["Player_Points"]
        .sum()
        .rename(columns={"Player_Points": "Total_Points"})
        .sort_values(by="Total_Points", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top_10_players.index += 1
    print("\n--- 7. TOP 10 PLAYERS BY TOTAL POINTS ---")
    print(top_10_players.to_string())

    # 8. Compare average points scored across match types and tournament rounds
    avg_pts_match_type = (
        df.groupby("Match_Type", as_index=False)["Player_Points"]
        .mean()
        .rename(columns={"Player_Points": "Avg_Points"})
        .sort_values(by="Avg_Points", ascending=False)
    )
    avg_pts_round = (
        df.groupby("Tournament_Round", as_index=False)["Player_Points"]
        .mean()
        .rename(columns={"Player_Points": "Avg_Points"})
        .sort_values(by="Avg_Points", ascending=False)
    )
    print("\n--- 8. AVERAGE POINTS BY MATCH TYPE & TOURNAMENT ROUND ---")
    print("By Match Type:\n", avg_pts_match_type.round(2).to_string(index=False))
    print("\nBy Tournament Round:\n", avg_pts_round.round(2).to_string(index=False))

    # 10. Identify the longest and shortest matches
    max_dur = df["Match_Duration_Mins"].max()
    min_dur = df["Match_Duration_Mins"].min()
    longest_matches = df[df["Match_Duration_Mins"] == max_dur][
        ["Match_ID", "Match_Date", "Team", "Opponent", "Match_Type", "Match_Duration_Mins"]
    ]
    shortest_matches = df[df["Match_Duration_Mins"] == min_dur][
        ["Match_ID", "Match_Date", "Team", "Opponent", "Match_Type", "Match_Duration_Mins"]
    ]
    print(f"\n--- 10. LONGEST ({max_dur} mins) & SHORTEST ({min_dur} mins) MATCHES ---")
    print("Longest Matches:\n", longest_matches.to_string(index=False))
    print("\nShortest Matches:\n", shortest_matches.to_string(index=False))

    # 11. Tuple for fixed match-result labels & Set for all participating teams
    match_result_labels = ("Win", "Loss", "Draw")
    participating_teams = set(df["Team"].unique()).union(set(df["Opponent"].unique()))
    print("\n--- 11. TUPLE & SET STRUCTURES ---")
    print(f"Fixed Match-Result Labels (Tuple): {match_result_labels}")
    print(f"Participating Teams (Set, {len(participating_teams)} teams): {sorted(participating_teams)}")

    # 12. Dictionary for team-wise wins & function check
    team_wins_dict = dict(zip(team_ranking["Team"], team_ranking["Wins"]))
    team_matches_dict = dict(zip(team_ranking["Team"], team_ranking["Matches_Played"]))
    print("\n--- 12. TEAM-WISE WINS DICTIONARY & FUNCTION CHECK ---")
    print("Team Wins Dictionary:", team_wins_dict)
    for team, wins in team_wins_dict.items():
        win_pct = calculate_win_percentage(wins, team_matches_dict[team])
        print(f"  {team:<10}: {wins} wins / {team_matches_dict[team]} matches -> {win_pct}% win rate")

    # 13. Use NumPy to calculate mean, median, and standard deviation of player points
    points_array = df["Player_Points"].to_numpy()
    np_mean = np.mean(points_array)
    np_median = np.median(points_array)
    np_std = np.std(points_array, ddof=1)
    print("\n--- 13. NUMPY STATISTICAL SUMMARY (PLAYER POINTS) ---")
    print(f"Mean: {np_mean:.2f} | Median: {np_median:.2f} | Std Dev: {np_std:.2f}")

    # 14. IQR Outlier Detection & Performance Tiers
    q1, q3 = np.percentile(points_array, [25, 75])
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr
    match_outliers = df[df["Player_Points"] > upper_fence]

    player_totals = df.groupby("Player_Name", as_index=False)["Player_Points"].sum()
    pq1, pq3 = np.percentile(player_totals["Player_Points"], [25, 75])
    piqr = pq3 - pq1
    p_upper_fence = pq3 + 1.5 * piqr
    player_outliers = player_totals[player_totals["Player_Points"] > p_upper_fence]

    def classify_performance(pts: float) -> str:
        if pts < q1:
            return f"Low (<{q1:.0f} pts)"
        elif pts <= np_median:
            return f"Average ({q1:.0f}-{np_median:.0f} pts)"
        elif pts <= q3:
            return f"Strong ({np_median + 1:.0f}-{q3:.0f} pts)"
        else:
            return f"Elite (>{q3:.0f} pts)"

    df["Performance_Group"] = df["Player_Points"].apply(classify_performance)
    print("\n--- 14. IQR OUTLIERS & PERFORMANCE TIERS ---")
    print(f"Match-Level IQR: Q1={q1}, Q3={q3}, IQR={iqr}, Upper Fence={upper_fence}")
    print(f"Match-Level High Score Outliers (> {upper_fence} pts): {len(match_outliers)} records")
    print(f"\nCumulative Player-Level IQR: Q1={pq1}, Q3={pq3}, IQR={piqr}, Upper Fence={p_upper_fence}")
    print("Cumulative Outlier Players:\n", player_outliers.to_string(index=False))
    print("\nMatch Performance Group Distribution:\n", df["Performance_Group"].value_counts())

    # 15. Player Class & Dynamic Observations
    top_row = df.loc[df["Player_Points"].idxmax()]
    sample_player = Player(top_row["Player_Name"], top_row["Team"], top_row["Player_Points"])
    top_team = team_ranking.iloc[0]
    top_player = top_10_players.iloc[0]

    print("\n--- 15. PLAYER CLASS & FIVE TOURNAMENT OBSERVATIONS ---")
    print("Sample Player Object ->", sample_player.summary())

    observations = [
        f"1. Top Scoring Team: {top_team['Team']} leads the tournament with {top_team['Total_Points']:.0f} total points and {top_team['Wins']} wins ({top_team['Win_Percentage']}% win rate).",
        f"2. Top Cumulative Scorer: {top_player['Player_Name']} leads all players with {top_player['Total_Points']:.0f} cumulative points; {len(match_outliers)} single-match IQR outliers exceeded {upper_fence:.1f} points.",
        f"3. Stage Scoring Trends: Highest average points occur in {avg_pts_match_type.iloc[0]['Match_Type']} matches ({avg_pts_match_type.iloc[0]['Avg_Points']:.2f} avg pts).",
        f"4. Match Duration Spread: Match lengths span from {min_dur:.0f} minutes ({len(shortest_matches)} matches) to {max_dur:.0f} minutes ({len(longest_matches)} matches), averaging {avg_duration:.2f} minutes.",
        f"5. Data Integrity Summary: Removed {duplicate_count} duplicate Match_ID records and imputed missing numerical entries using medians ({points_median:.1f} pts, {duration_median:.1f} mins).",
    ]
    print("\nFive Key Observations:")
    for obs in observations:
        print(obs)


if __name__ == "__main__":
    main()
