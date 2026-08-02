"""
clean_steam_data.py

Cleans the raw Steam games CSV for use in a content-based recommendation engine.

WHY THIS SCRIPT EXISTS
-----------------------
The raw CSV's header row is missing one column name. Every field from
"About the game" onward is shifted one position to the right of where its
header name says it is. Reading it with plain pd.read_csv() would silently
put Tags data into the "Genres" column, playtime numbers into "Developers",
etc. This script rebuilds the header correctly before doing anything else.

WHAT IT DOES
------------
1. Fixes the header misalignment.
2. Drops columns that are irrelevant for recommendations (URLs, images,
   support emails, etc.) to keep the file small and manageable.
3. Filters out games with too few reviews (default: fewer than 50 total
   Positive + Negative reviews). Steam is full of shovelware/test/dead
   entries -- ~76% of the raw dataset has fewer than 50 reviews. Keeping
   them makes your recommender suggest games nobody has ever played.
4. Builds a single `combined_features` text column (genres + categories +
   tags + developer + description) that you feed straight into
   TfidfVectorizer for the recommender.
5. Parses "Estimated owners" (e.g. "0 - 20000") into a numeric midpoint,
   useful later for popularity-based ranking or tie-breaking.
6. Saves the result to steam_games_clean.csv.

USAGE
-----
    python3 clean_steam_data.py input.csv output.csv --min-reviews 50
"""

import argparse
import pandas as pd
import csv


def fix_header(raw_path: str) -> list[str]:
    """Rebuild the header to match the actual (shifted) data columns."""
    with open(raw_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        orig_header = next(reader)

    # There's one extra data field after "DiscountDLC count" that has no
    # header name of its own -- insert a placeholder there and split the
    # merged label.
    fixed = orig_header[:8] + ["DLC count"] + orig_header[8:]
    fixed[7] = "Discount"
    return fixed


def parse_owners_midpoint(owners_range: str) -> int:
    """'0 - 20000' -> 10000. Returns 0 if unparseable."""
    try:
        low, high = owners_range.split(" - ")
        return (int(low) + int(high)) // 2
    except Exception:
        return 0


# Steam "Categories" mixes genuinely useful gameplay descriptors (Co-op,
# PvP, VR Only) with Steam *platform* features that almost every game has
# and say nothing about the game itself. Left in, these act as noise that
# makes unrelated games look similar just because they both have
# achievements or cloud saves. We filter them out before vectorizing.
NOISE_CATEGORIES = {
    "Family Sharing", "Steam Achievements", "Steam Cloud",
    "Full controller support", "Partial Controller Support",
    "Steam Trading Cards", "Steam Leaderboards", "Steam Workshop",
    "Stats", "Remote Play on TV", "Remote Play Together",
    "Remote Play on Phone", "Remote Play on Tablet",
    "Tracked Controller Support", "Custom Volume Controls",
    "In-App Purchases", "Cross-Platform Multiplayer", "Captions available",
    "Commentary available", "SteamVR Collectibles",
}


def clean_categories(categories: str) -> str:
    if not categories or pd.isna(categories):
        return ""
    kept = [c for c in categories.split(",") if c.strip() not in NOISE_CATEGORIES]
    return ",".join(kept)


# --- FIELD WEIGHTS ---
# How many times each field's text gets repeated before being joined into
# combined_features. More repeats = more occurrences of those words = more
# influence on the TF-IDF vector for that game. This is a blunt but
# effective way to say "Genres matter more than the free-text description."
# Tune these and re-run this script to see how recommendations change.
WEIGHT_GENRES = 3
WEIGHT_CATEGORIES = 2
WEIGHT_TAGS = 2
WEIGHT_DEVELOPER = 0
WEIGHT_DESCRIPTION = 1

# How many words of "About the game" to keep. Higher = more descriptive
# text influences the vector (both good context and more noise words).
DESCRIPTION_WORD_LIMIT = 30


def build_combined_features(row: pd.Series) -> str:
    """
    Combine genre/category/tag/developer/description text into one field
    for TF-IDF. See the WEIGHT_* constants above to control how much each
    field influences the final similarity score.
    """
    genres = str(row["Genres"]) if pd.notna(row["Genres"]) else ""
    categories = clean_categories(row["Categories"])
    tags = str(row["Tags"]) if pd.notna(row["Tags"]) else ""
    developer = str(row["Developers"]) if pd.notna(row["Developers"]) else ""
    about = str(row["About the game"]) if pd.notna(row["About the game"]) else ""

    # Keep only the first N words of the description so a long wall of
    # marketing text doesn't drown out the genre/tag signal.
    about_short = " ".join(about.split()[:DESCRIPTION_WORD_LIMIT])

    parts = [
        (genres + " ") * WEIGHT_GENRES,
        (categories + " ") * WEIGHT_CATEGORIES,
        (tags + " ") * WEIGHT_TAGS,
        (developer + " ") * WEIGHT_DEVELOPER,
        (about_short + " ") * WEIGHT_DESCRIPTION,
    ]
    return " ".join(parts).strip()


def clean(input_path: str, output_path: str, min_reviews: int) -> None:
    fixed_header = fix_header(input_path)

    print("Reading raw CSV...")
    df = pd.read_csv(input_path, skiprows=1, names=fixed_header, index_col=False)
    print(f"  Loaded {len(df):,} rows")

    # Keep only columns relevant to a recommendation engine.
    keep_cols = [
        "AppID", "Name", "Release date", "Estimated owners", "Price",
        "Metacritic score", "Positive", "Negative", "Developers",
        "Publishers", "Categories", "Genres", "Tags", "About the game",
    ]
    df = df[keep_cols].copy()

    # Filter out low-signal / shovelware / dead entries.
    total_reviews = df["Positive"].fillna(0) + df["Negative"].fillna(0)
    before = len(df)
    df = df[total_reviews >= min_reviews].copy()
    print(f"  Filtered to games with >= {min_reviews} reviews: "
          f"{len(df):,} rows kept ({len(df) / before * 100:.1f}%)")

    # Drop rows with no name (shouldn't exist, but just in case).
    df = df[df["Name"].notna() & (df["Name"].str.strip() != "")]

    # Numeric popularity proxy.
    df["owners_estimate"] = df["Estimated owners"].fillna("0 - 0").apply(parse_owners_midpoint)

    # Review score (useful for ranking ties later).
    df["review_score"] = (
        df["Positive"] / (df["Positive"] + df["Negative"]).replace(0, 1)
    ).round(3)

    # Build the text field the recommender will actually vectorize.
    print("Building combined text field...")
    df["combined_features"] = df.apply(build_combined_features, axis=1)

    # Drop any rows where we ended up with essentially no text signal.
    before2 = len(df)
    df = df[df["combined_features"].str.len() > 10]
    if before2 != len(df):
        print(f"  Dropped {before2 - len(df)} rows with no usable text signal")

    df = df.reset_index(drop=True)

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset to {output_path}  ({len(df):,} rows, {len(df.columns)} columns)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--min-reviews", type=int, default=50,
                         help="Minimum total (positive+negative) reviews to keep a game. Default: 50")
    args = parser.parse_args()
    clean(args.input_csv, args.output_csv, args.min_reviews)
