from collections import defaultdict
import json
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

    parse_ts = pd.Timestamp.now().floor("D")

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
    df["parse_ts"] = parse_ts
    df["TitleId"] = range(len(df))
    
    return df


def get_movie_data(wiki_page):

    params = wiki_params(wiki_page)
    resp = requests.get(WIKI_API_URL, params=params)
    response_dict = resp.json()

    html_content = response_dict["parse"]["text"]
    soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')

    content_div = soup.find("div", {"class": "mw-content-ltr"})
    data = defaultdict(list)
    default_header = "General"
    current_header = None
    data[default_header] = []

    # Find the infobox vevent element and remove it
    infobox = content_div.find("table", {"class": "infobox vevent"})
    if infobox:
        infobox.decompose()

    exclude_headers = ["external links", "references", "see also", "Further reading", "notes", "bibliography"]

    for element in content_div.descendants:
        if element.name in ['h2']:
            current_header = element.text.strip()
            meta_header = current_header
        elif element.name in ['h3']:
            current_header = element.text.strip()
            if meta_header:
                current_header = meta_header + "-" + current_header
        elif element.name in ['p', 'ul', 'ol']:
            header_name = current_header or default_header
            if header_name.lower() in exclude_headers:
                continue
            
            text = element.get_text(strip=True, separator=" ")
            decoded_text = text.encode('utf-8').decode('unicode-escape')
            if decoded_text:
                data[header_name].append(decoded_text)

    # Flatten the lists and join text passages
    for key in data:
        data[key] = " ".join(data[key])

    return data




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


def get_movie_details(movies_df):
    movie_data_list = []
    for _, row in movies_df.iterrows():
        if row.Link:
            try:
                wiki_page = row.Link.replace("/wiki/", "")
                parse_ts = pd.Timestamp.now().floor("D").strftime("%Y-%m-%d")
                movie_data = get_movie_data(wiki_page)
                if movie_data:
                    movie_data["parse_ts"] = parse_ts
                    movie_data["TitleId"] = row.TitleId
                    movie_data["SourceYear"] = row.SourceYear
                    movie_data["Title"] = row.Title
                    movie_data_list.append(movie_data)
            except Exception:
                print(f"Failed to process the title {row.Title} and link {row.Link}")

    return movie_data_list


if __name__ == "__main__":

    years = ["2021", "2022", "2023", "since_2024"]
    save_to = Path("./data/raw")

    for year in years:
        wiki_page = f"List_of_Netflix_original_films_({year})"
        
        movies = get_movie_list(wiki_page)
        movies["SourceYear"] = year
        movies_data = get_movie_details(movies)

        print(f"Parsed {len(movies)} movies and {len(movies_data)} movie-data for year {year}")
        
        # movies.to_parquet(save_to / f"movies_context_{year}.parquet")
        movies.to_csv(save_to / f"movies_context_{year}.csv")
        with open(save_to / f"movies_data_{year}.json", "w") as f:
            json.dump(movies_data, f, indent=4)
        



