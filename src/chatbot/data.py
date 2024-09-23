import requests
from bs4 import BeautifulSoup
import pandas as pd
from pathlib import Path

wiki_params = lambda wiki_page :  {
        "action": "parse",
        "prop": "text",
        "page": wiki_page,
        "exlimit": 1,
        "explaintext": 1,
        "formatversion": 2,
        "format": "json"
}
WIKI_API_URL = "https://en.wikipedia.org/w/api.php"


def get_movie_list(wiki_page):

    params = wiki_params(wiki_page)
    resp = requests.get(WIKI_API_URL, params=params)
    response_dict = resp.json()

    html_content = response_dict["parse"]["text"]
    soup = BeautifulSoup(html_content, 'html.parser')

    tables = soup.find_all("table")

    movie_list = []
    table = tables[0]

    for row in table.find_all('tr'):
        cols = row.find_all('th')
        if cols:
            cols_text = [ele.text.strip() for ele in cols]
            cols_text += ["Link"]

        cols = row.find_all('td')
        if cols:
            cols_text = [ele.text.strip() for ele in cols]

            movie_links = cols[0].find_all("a")
            l = movie_links[0].get("href") if movie_links and movie_links[0].parent != "sup" else ""
            cols_text += [l]

        movie_list.append(cols_text)

    df = pd.DataFrame(movie_list)
    df.columns = df.iloc[0]
    df = df.iloc[1:, :]
    df["TitleId"] = range(len(df))

    return df


def get_cast_data(wiki_page):
    params = wiki_params(wiki_page)
    resp = requests.get(WIKI_API_URL, params=params)
    response_dict = resp.json()

    html_content = response_dict["parse"]["text"]
    soup = BeautifulSoup(html_content, 'html.parser')

    sections = [el.text for el in soup.find_all("h2")]
    cast_section_id = None
    for i, val in enumerate(sections):
        if "cast" in val.lower():
            cast_section_id = i
            break

    if cast_section_id is not None:
        cast_list = [el.text for el in soup.find_all("h2")[cast_section_id].find_next('ul').find_all("li")]
    else:
        cast_list = []

    return cast_list

def get_movies_cast(movies_df):
    cast_data_list = []
    for index, row in movies_df.iterrows():
        if row.Link:
            try:
                wiki_page = row.Link.replace("/wiki/", "")
                cast_data = get_cast_data(wiki_page)
                if cast_data:
                    cast_data_list.append([row.TitleId, row.Title, "; ".join(cast_data) ])
            except Exception:
                print(f"Failed to process the title {row.Title} and link {row.Link}")

    return pd.DataFrame(cast_data_list, columns=["TitleId", "Title", "CastInfo"])


if __name__ == "__main__":

    wiki_page = "List_of_Netflix_original_films_(since_2024)"
    movies = get_movie_list(wiki_page)
    movies_cast = get_movies_cast(movies)

    # Add natural language texts
    # movies["text"] = movies.apply(lambda r: f"'{r.Title}' is a {r.Genre} Netflix movie in {r.Language} language that was released on {r['Release date']}", axis=1)
    movies_cast["text"] = movies_cast.apply(lambda r: f"Netflix movie '{r.Title}' cast: {r.CastInfo}", axis=1)

    save_to = Path("./data/raw")
    movies.to_csv(save_to / "movies_context_2024_with_meta.csv")
    movies_cast.to_csv(save_to / "movie_cast_2024_with_meta.csv")




