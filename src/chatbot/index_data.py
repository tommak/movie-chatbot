"""
This script indexes a database of movies using the Chroma vector store and OpenAI embeddings.
"""

import os
from typing import List, Optional
import pandas as pd
import polars as pl
from pathlib import Path
import datetime as dt

from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_core.pydantic_v1 import BaseModel, Field


def compile_docs(path: Path, data_version: str):
    
    docs = []

    data_paths = [
        (path / f"movies_{data_version}.parquet", False),
        (path / f"movies_data_{data_version}.parquet", True)
    ]

    for data_path, data_with_topic in data_paths:
        movies_data = pl.read_parquet(data_path)
        movies_data = movies_data.with_columns(
            (pl.col("release_date").dt.timestamp("us") // 1e6).alias("release_date_ts")
        )
        meta_names = ["release_date_ts", "Title", "runtime_min", "Genre", "Language"]
        for row in movies_data.iter_rows(named=True):
            meta = {nm: row[nm] for nm in meta_names}
            meta["source"]= f"https://en.wikipedia.org/wiki/List_of_Netflix_original_films_({row['SourceYear']})"
            meta["topic"] = row["topic"] if data_with_topic else "Movie-Info"
            document = Document(
                    page_content=row["text"],
                    metadata=meta
                )
            docs.append(document)

    return docs
     

class Search(BaseModel):
    """Search over a database of movies."""

    query: str = Field(
        ...,
        description="Similarity search query applied to movies database.",
    )
    # release_date_min_ts: Optional[int] = Field(None, description="The earliest release date of a movies to consider, in the form of UNIX timestamp")
    release_date_min: Optional[str] = Field(None, description="The earliest release date of a movies to consider, use the format '2024-09-11'. ")


class ChromaMoviesContext:
    """
    A class to manage the context for movie-related data using Chroma for vector storage.
    Attributes:
    -----------
    vectorstore_ : Chroma
        An instance of Chroma used for storing and retrieving document vectors.
    Methods:
    --------
    __init__(context_group_name: str, model_name: str, embedding_function: Embeddings, path: Path)
        Initializes the ChromaMoviesContext with the specified parameters.
    index_docs(docs)
        Indexes the provided documents into the vector store.
    vectorstore
        Returns the vector store instance.
    as_search_retriever(search: Search) -> List[Document]
        Retrieves documents from the vector store based on the search query and optional release date filter.
    """

    def __init__(self, context_group_name: str, model_name: str, embedding_function: Embeddings, path: Path):
    
        self.vectorstore_ = Chroma(
                    collection_name=context_group_name,
                    embedding_function=embedding_function,
                    persist_directory=os.path.join(path, f"{model_name}/meta_chroma_langchain_db")
                )
    
    def index_docs(self, docs):
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # splits = text_splitter.split_documents(docs)
        self.vectorstore_.add_documents(documents=docs)

    @property
    def vectorstore(self):
        return self.vectorstore_

    def as_search_retriever(self, search: Search) -> List[Document]:
        if search.release_date_min is not None:
            print(f"Detected release_date bound {search.release_date_min}")
            ts = pd.Timestamp(search.release_date_min).timestamp()
            _filter = {"release_date_ts": {"$gte": ts}}
        else:
            _filter = None
        return self.vectorstore_.similarity_search(search.query, filter=_filter)



if __name__ == "__main__":

    context_group_name = "movies"
    embedding_function = OpenAIEmbeddings()

    path = Path("data/processed/")
    data_version = "dver01"
    docs = compile_docs(path, data_version)
    
    cache_path = "./cache/context"
     
    index_version = f"openai_{data_version}_indver01"
    context = ChromaMoviesContext(context_group_name, index_version, embedding_function=embedding_function, path=cache_path)
    context.index_docs(docs)

    # Test semantic search
    ts = dt.datetime(2024, 5, 1).timestamp()
    search_res = context.vectorstore.similarity_search_with_score(query="Japanese movie", filter={"release_date_ts": {"$gte": ts}}, k=5)
    for doc, score in search_res:
        print(f"[{score}] " + doc.page_content)