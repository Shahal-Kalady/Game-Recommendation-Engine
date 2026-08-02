# Steam Game Recommendation Engine

A content-based recommendation engine that suggests similar Steam games based on genre, category, tags, developer, and description — built with Python, pandas, and scikit-learn.

Given a game you like (e.g. *Cyberpunk 2077*), it recommends similar games (e.g. *The Witcher 3*) using TF-IDF text vectorization and cosine similarity — no external ML APIs, no black-box models, just a transparent, explainable pipeline.

## Why this project

Netflix, Spotify, and Steam all use recommendation systems to help users discover content. This project builds a simplified but genuinely functional version of the same core idea — content-based filtering — on real, messy, public data (125,000+ games from the [Steam Games Dataset](https://www.kaggle.com/datasets/fronkonstin/steam-games)), including the actual data-cleaning and debugging work that comes with using real-world data.

## What it does

1. **Fixes a real data bug.** The raw CSV's header row is missing one column name, silently shifting every field from "About the Game" onward by one position. `clean_steam_data.py` traces and corrects this before anything else touches the data.
2. **Filters out low-signal entries.** ~76% of Steam's catalog has fewer than 50 total reviews (shovelware, dead projects, test builds). These are filtered out so the engine doesn't recommend games nobody has played.
3. **Builds a weighted text profile per game**, combining genres, categories, tags, developer, and a trimmed description — with tunable weights so each field's influence on similarity can be adjusted.
4. **Removes noisy metadata.** Steam's "Categories" field mixes real gameplay descriptors (Co-op, PvP) with platform features nearly every game has (Steam Achievements, Family Sharing, Steam Cloud). These are filtered out since they were making unrelated games look artificially similar.
5. **Vectorizes with TF-IDF** and computes cosine similarity **on demand, per query** — not as a full precomputed matrix, which would need several GB of RAM for this dataset size. This keeps the approach viable at larger catalog sizes too.

## Tech stack

- Python 3
- pandas — data loading and cleaning
- scikit-learn — `TfidfVectorizer`, `cosine_similarity`

## Project structure

```
├── clean_steam_data.py     # Fixes header bug, filters low-signal games, builds weighted text field
├── recommender.py          # Loads cleaned data, vectorizes, serves recommendations
├── steam_games_clean.csv   # Generated output (not included — see Setup)
└── README.md
```

## Setup

```bash
git clone https://github.com/YOUR-GITHUB/steam-game-recommender.git
cd steam-game-recommender
pip install pandas scikit-learn
```

Download the raw dataset from [Kaggle](https://www.kaggle.com/datasets/fronkonstin/steam-games) and place `steam_games_dataset.csv` in this folder. (Not included in this repo — it's ~400MB, too large for GitHub.)

Clean the data:

```bash
python clean_steam_data.py steam_games_dataset.csv steam_games_clean.csv --min-reviews 50
```

Run the recommender:

```bash
python recommender.py "Stardew Valley"
```

or launch interactive mode:

```bash
python recommender.py
```

## Example output

```
Because you liked: "Cyberpunk 2077"
------------------------------------------------------------
0.893  The Witcher 3: Wild Hunt
       Genres: RPG
...
```

## How the similarity works

Each game's text (genres, tags, categories, etc.) is converted into a TF-IDF vector — a way of weighting words so common ones (like "Action") count less and rare, distinctive ones (like "Souls-like") count more. Comparing two games' vectors by the angle between them (cosine similarity) gives a score from 0 (nothing in common) to 1 (identical). Field weights (e.g. Genres weighted 3x more than the description) are configurable constants in `clean_steam_data.py`.

## Known limitations

- **Cold start problem**: a game with a sparse or missing description/tags gets weak recommendations, since there's little text to compare.
- **No personalization**: this recommends *similar games*, not games tailored to a specific user's play history — that would require collaborative filtering and user-level data, which this dataset doesn't include.
- **Tag noise**: Steam tags are community-submitted and occasionally inconsistent (~34% of games in the raw dataset have no tags at all).
- **No live data**: recommendations are only as fresh as the last time the dataset was downloaded and cleaned.

## Possible next steps

- Add a simple web interface (Streamlit) for live demos
- Add popularity-based tie-breaking using review counts
- Experiment with collaborative filtering if a ratings-per-user dataset becomes available

## Credits

Built as a learning project to practice data cleaning, feature engineering, and content-based recommendation systems using Python.
