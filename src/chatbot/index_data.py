"""
This script indexes a database of movies using the Chroma vector store and OpenAI embeddings.
"""

import os
from typing import List, Optional
import pandas as pd
from pathlib import Path

from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_core.pydantic_v1 import BaseModel, Field


def compile_docs(path):
    
    movies_data = pd.read_csv(path)
    docs = []
    for _, row in movies_data.iterrows():
        meta = {"source": "https://en.wikipedia.org/wiki/List_of_Netflix_original_films_(since_2024)",
                "release_date_ts": row["release_date_ts"]}
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
    release_date_min: Optional[str] = Field(None, description="The earliest release date of a movies to consider, use the format '2024-09-11' ")


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
    data_path = "data/processed/movies_context_2024_with_meta_cleaned.csv"
    docs = compile_docs(data_path)
    path = "./cache/context"
     
    context = ChromaMoviesContext(context_group_name, "openai", embedding_function=embedding_function, path = path)
    context.index_docs(docs)

    # Test semantic search
    ts = pd.Timestamp("2024-05-01").timestamp()
    search_res = context.vectorstore.similarity_search_with_score(query="Japanese movie", filter={"release_date_ts": {"$gte": ts}}, k=5)
    for doc, score in search_res:
        print(f"[{score}] " + doc.page_content)