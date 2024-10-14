import re
import json 
from pathlib import Path
import pandas as pd
import polars as pl


def fix_shifted_data(movies: pl.DataFrame) -> pl.DataFrame:
    is_shifted = pl.col("Release date").str.replace(r'(\[\d+\])+', '').str.strip_chars().str.strptime(dtype=pl.Datetime, format='%B %d, %Y', strict=False).is_null()
    shifted_data = movies.filter(is_shifted).to_pandas()
    not_shifted_data = movies.filter(~is_shifted).to_pandas()

    shifted_data.iloc[:, 2:-3] = shifted_data.iloc[:, 1:-4]
    shifted_data.iloc[:, 1] = None

    return pl.from_pandas(pd.concat([
        shifted_data,
        not_shifted_data
    ]))


if __name__ == "__main__":

    # Process movies titles

    path = Path("/Users/tamara/Documents/Projects/movie-chatbot/data/raw")
    movies = pl.read_csv(path / "movies_context_*.csv").drop("")

    runtime_pattern = "(?:(\d+)\s?h)?(?:\s?(\d+)\s?min)?"

    movies_clean = fix_shifted_data(
        movies.filter(pl.col("Title") != "Awaiting release")
    ).with_columns(
        pl.col("Release date").str.replace(r'(\[\d+\])+', '').str.strip_chars().str.strptime(dtype=pl.Datetime, format='%B %d, %Y', strict=False).alias("release_date"),
        pl.col("parse_ts").str.to_datetime().alias("parse_ts"),
        pl.col("SourceYear").str.replace("since_", "").cast(pl.Int32).alias("year"),
        pl.col('Title').str.replace(r'(\[\d+\])+', '').str.strip_chars(),
        (pl.col("Runtime").str.extract_groups(runtime_pattern).struct["1"].cast(pl.Int32) * 60 +
        pl.col("Runtime").str.extract_groups(runtime_pattern).struct["2"].cast(pl.Int32).fill_null(0)).alias("runtime_min"),
    ).sort(["year", "TitleId"]).with_columns(
        pl.col("release_date").forward_fill()
    ).with_columns(
        (pl.col('Title') + pl.lit(" is a ") + pl.col('Genre') + pl.lit(" Netflix movie in ") + pl.col('Language') +
        + pl.lit(" language that was released on ") + pl.col('release_date').dt.strftime("%Y-%m-%d")).alias('text'),
    )

    # Process movies data

    movies_data_l = []
    meta_cols = ["parse_ts", "TitleId", "SourceYear", "Title"]

    years = ["2021", "2022", "2023", "since_2024"]
    for year in years:
        with open(path / f"movies_data_{year}.json", "r") as f:
            movies_data = json.load(f)

        for movie_info in movies_data:
            for topic, topic_text in movie_info.items():
                if topic not in meta_cols:
                    movies_data_l.append({
                        "parse_ts": movie_info["parse_ts"],
                        "TitleId": movie_info["TitleId"],
                        "SourceYear": movie_info["SourceYear"],
                        "topic": topic,
                        "text": topic_text
                    })


    movies_clean_cols = ["TitleId", "SourceYear", "Title", "Genre", "runtime_min", "Language", "release_date"]
    movies_data_df = pl.DataFrame(movies_data_l).with_columns(
        pl.col("parse_ts").str.to_datetime(),
    ).join(movies_clean.select(movies_clean_cols), on=["TitleId", "SourceYear"], how="left")

    # Save processed data

    save_to = Path("/Users/tamara/Documents/Projects/movie-chatbot/data/processed")
    movies_clean.select(["Title", "Genre", "Language", "SourceYear", "TitleId", "year", "release_date", 
                         "runtime_min", "text", "parse_ts"]).write_csv(save_to / "movies.csv")
    movies_data_df.write_csv(save_to / "movie_data.csv")

