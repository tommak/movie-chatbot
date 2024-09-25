import re
from pathlib import Path
import pandas as pd


if __name__ == "__main__":

    path = Path("/Users/tamara/Documents/Projects/movie-chatbot/data/raw")
    movies = pd.read_parquet(path / "movies_context_2024_with_meta.parquet")
    movies_cast = pd.read_parquet(path / "movie_cast_2024_with_meta.parquet")

    movies = movies[movies["Title"] != "Awaiting release"].copy()
    movies["release_date"] = pd.to_datetime(movies['Release date'].str.replace(r'\[\d+\]', '', regex=True).str.strip(), format='%B %d, %Y', errors='coerce')
    movies["Title"] = movies['Title'].str.replace(r'\[\d+\]', '', regex=True).str.strip()
    movies["text"] = movies.apply(lambda r: f"'{r.Title}' is a {r.Genre} Netflix movie in {r.Language} language that was released on {r.release_date}", axis=1)
    movies = movies.set_index("TitleId")
    movies['release_date_ts'] = movies['release_date'].astype('int64') // 10**9

    movies_cast = movies_cast.set_index("TitleId")[["CastInfo"]].join(movies[["Title", "release_date", "release_date_ts"]], how="left")
    movies_cast["text"] = movies_cast.apply(lambda r: f"Netflix movie '{r.Title}' cast: {r.CastInfo}", axis=1)

    save_to = Path("/Users/tamara/Documents/Projects/movie-chatbot/data/processed")
    movies.to_parquet(save_to / "movies_context_2024_with_meta.parquet")
    movies_cast.to_parquet(save_to / "movie_cast_2024_with_meta.parquet")

