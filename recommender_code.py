"""
recommender.py

A content-based game recommendation engine.

HOW IT WORKS (the actual concept, not just the code)
------------------------------------------------------
Every game has a `combined_features` text field (genres + categories + tags +
developer + description, built in clean_steam_data.py). We convert each
game's text into a vector of numbers using TF-IDF, then measure how "close"
two games are by the angle between their vectors (cosine similarity).

TF-IDF (Term Frequency - Inverse Document Frequency):
  - Term Frequency: how often a word appears in THIS game's text.
  - Inverse Document Frequency: how RARE that word is across ALL games.
  - A word like "Action" appears in thousands of games, so it gets a low
    weight. A word like "Souls-like" appears in relatively few, so it gets
    a higher weight. This means rare, distinctive tags matter more than
    generic ones when comparing games -- which is exactly what you want.

Cosine similarity:
  - Every game becomes a point in a high-dimensional space (one dimension
    per unique word across the whole dataset).
  - Cosine similarity measures the angle between two games' vectors, not
    their distance. This matters because it ignores length differences
    (a game with a long description isn't unfairly judged "more different"
    just because it has more words) and focuses purely on whether the
    same words show up in the same proportions.
  - Score ranges from 0 (nothing in common) to 1 (identical text).

USAGE
-----
    python3 recommender.py                  # interactive mode, type game names
    python3 recommender.py "God of War"     # one-off lookup
"""

import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class GameRecommender:
    def __init__(self, csv_path: str):
        print("Loading data...")
        self.df = pd.read_csv(csv_path)
        self.df["combined_features"] = self.df["combined_features"].fillna("")

        print(f"Vectorizing {len(self.df):,} games with TF-IDF...")
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=15000)
        self.matrix = self.vectorizer.fit_transform(self.df["combined_features"])
        print("Ready.\n")

        # NOTE: We deliberately do NOT precompute a full NxN similarity
        # matrix here. For 30,624 games that matrix would need ~7GB of RAM
        # (30624 * 30624 * 8 bytes), and 99.99% of those pairwise scores
        # would never even get looked at. Instead, _similarity_for_row()
        # below does a single row's worth of comparison (1 x N) on demand --
        # a few milliseconds instead of a memory crash. This is the same
        # reason real recommendation systems (Netflix, Spotify) don't do
        # naive full pairwise comparison once catalogs get large.

    def _similarity_for_row(self, idx: int):
        """Cosine similarity between one game and every other game."""
        return cosine_similarity(self.matrix[idx], self.matrix).flatten()

    def find_matches(self, query: str, limit: int = 5) -> pd.DataFrame:
        """Partial, case-insensitive name search. Returns candidate rows."""
        mask = self.df["Name"].str.contains(query, case=False, na=False, regex=False)
        matches = self.df[mask]
        # Show the most-reviewed matches first, since that's most likely
        # what the person meant (e.g. "Cyberpunk 2077" over "Cyberpunk Horror").
        matches = matches.sort_values("Positive", ascending=False)
        return matches.head(limit)

    def recommend(self, game_name: str, n: int = 5):
        """
        Returns (matched_game_name, DataFrame of top-n similar games)
        or (None, candidates_df) if the name is ambiguous/not found.
        """
        exact = self.df[self.df["Name"].str.lower() == game_name.lower()]

        if len(exact) == 1:
            idx = exact.index[0]
        else:
            candidates = self.find_matches(game_name)
            if len(candidates) == 0:
                return None, None
            if len(candidates) > 1 and len(exact) == 0:
                # Ambiguous -- let the caller decide what to show the user.
                return None, candidates
            idx = candidates.index[0]

        matched_name = self.df.loc[idx, "Name"]
        sims = self._similarity_for_row(idx)
        scores = list(enumerate(sims))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:n]  # exclude itself

        result_idx = [i for i, _ in scores]
        result_scores = [round(s, 3) for _, s in scores]

        result = self.df.loc[result_idx, ["Name", "Genres", "Tags"]].copy()
        result["similarity"] = result_scores
        return matched_name, result.reset_index(drop=True)


def _print_recommendation(rec: GameRecommender, query: str):
    matched, result = rec.recommend(query)

    if matched is None and result is None:
        print(f'  No games found matching "{query}".\n')
        return

    if matched is None:
        print(f'  "{query}" is ambiguous. Did you mean one of these?')
        for name in result["Name"].head(5):
            print(f"    - {name}")
        print()
        return

    print(f'  Because you liked: "{matched}"')
    print(f"  {'-'*60}")
    for _, row in result.iterrows():
        print(f"  {row['similarity']:.3f}  {row['Name']}")
        print(f"         Genres: {row['Genres']}")
    print()


if __name__ == "__main__":
    rec = GameRecommender(r"C:\Users\shaha\OneDrive\Desktop\Projects\Game Recommendation Engine\steam_games_clean.csv")

    if len(sys.argv) > 1:
        _print_recommendation(rec, " ".join(sys.argv[1:]))
    else:
        print("Type a game name to get recommendations (or 'quit' to exit).\n")
        while True:
            query = input("Game name: ").strip()
            if query.lower() in ("quit", "exit", ""):
                break
            _print_recommendation(rec, query)
