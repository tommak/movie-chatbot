from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from index_data import ChromaMoviesContext, Search
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.output_parsers import StrOutputParser

from prompts import retriever_query_analyzer_prompt, contextualize_qa_system_prompt


def get_query_analyzer(llm):
    # system = """You are an expert at converting user questions into database queries. \
    # You have access to a database of movies. You need to understand if user is defining any
    # time limits for the movies he is interested in and formulate the earliers acceptable date. 
    # Today is 2024-09-12.
    # Given a question, return a list of database queries optimized to retrieve the most relevant results."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", retriever_query_analyzer_prompt),
            ("human", "{input}"),
        ]
    )

    structured_llm = llm.with_structured_output(Search)
    query_analyzer = {"input": RunnablePassthrough(), "current_date": RunnablePassthrough()} | prompt | structured_llm
    return query_analyzer


def create_retriever(llm, context_store):
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    retriever = get_query_analyzer(llm) | context_store.as_search_retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever


if __name__ == "__main__":
    model = "gpt-3.5-turbo"
    llm = ChatOpenAI(model=model, temperature=0)

    context_group_name = "movies"
    embedding_function = OpenAIEmbeddings()
    path = "./cache/context"
    context = ChromaMoviesContext(context_group_name, "openai", embedding_function=embedding_function, path=path)

    history_aware_retriever = create_retriever(llm, context)
    res = history_aware_retriever.invoke({"input": "Any new movies this year?", "chat_history": [("human", "Hi")]})
    print(res)
