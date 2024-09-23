from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_message_histories.file import FileChatMessageHistory

import uvicorn
from fastapi import FastAPI, Request, Form
from typing import Any, Dict

from index_data import ChromaMoviesContext
from retriever import create_retvriever

import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "movies-chatbot"
    

class RAGChatBot:
    """
    RAGChatBot is a class that represents a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about movies and TV series.
    Attributes:
        llm: The language model used for generating responses.
        retriever: The retriever used for fetching relevant documents.
    Methods:
        __init__(llm, retriever):
            Initializes the RAGChatBot with the given language model and retriever.
        get_session_history(session_id: str) -> BaseChatMessageHistory:
            Retrieves the chat history for a given session ID from the cache.
        create_qa_chain():
            Creates a question-answering chain using the language model and retriever, and wraps it with a conversational chain that maintains session history.
    """
    def __init__(self, llm: ChatOpenAI, retriever: Any) -> None:
        self.llm = llm
        self.retriever = retriever
        
    @staticmethod
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        path = f"./cache/chat_history/{session_id}"
        return FileChatMessageHistory(path)

    def create_qa_chain(self) -> RunnableWithMessageHistory:
        system_prompt = (
            """
            You are an assistant answering questions about movies and tv series. Below you are given a context about the latest movies.
            Answer the question giving priority to the context below, but also use your general knowledge and the information about the movies released earlier.
            If there are multiple movies with the same name, give priority to the latest one. If needed, mention all existing movies with the given name.
            Today is 2024-09-12.

            {context}
        """
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self.get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )
        return conversational_rag_chain


def setup_chain() -> RunnableWithMessageHistory:
    llm_config: Dict[str, Any] = {
        "model": "gpt-3.5-turbo",
        "temperature": 0
    }

    context_group_name = "movies"
    llm = ChatOpenAI(**llm_config)
    embedding_function = OpenAIEmbeddings()
    context_path = "./cache/context"
    
    context_store = ChromaMoviesContext(context_group_name, "openai", embedding_function, context_path)
    retriever = create_retvriever(llm, context_store)
    
    chatbot = RAGChatBot(llm, retriever)
    return chatbot.create_qa_chain()


def run_console_chat(stream: bool = False, session_id: str = "default_id") -> None:
    chain = setup_chain()
    
    print("Movie-Chatbot: Hi! I'm a Movie Chatbot. How can I assist you today?")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye", "q"]:
            print("Movie-Chatbot: Goodbye! Have a great day!")
            break
        else:
            print(f"Movie-Chatbot: ")
            if not stream:
                response = chain.invoke({"input": user_input},
                                        config={"configurable": {"session_id": session_id}})["answer"]
                if "context" in response:
                    print("My context: ")
                    for doc in response["context"]:
                        print(doc.page_content)
                print("-"*50)
                print(response)
            else:
                for r in chain.stream({"input": user_input},
                                    config={"configurable": {"session_id": session_id}}):
                    if "context" in r:
                        print("My context: ")
                        for doc in r["context"]:
                            print(doc.page_content)
                    if "answer" in r:
                        print(r["answer"], end="")
                print("\n")


def serve_api_endpoint() -> None:
    app = FastAPI()
    chain = setup_chain()

    @app.get("/")
    def read_root(request: Request) -> Dict[str, str]:
        return {"message": "Welcome to the Movies ChatBot API!"}

    @app.post("/prompt")
    def process_prompt(prompt: str = Form(...), session_id: str = Form(...)) -> Dict[str, str]:
        response = chain.invoke({"input": prompt},
                                config={"configurable": {"session_id": session_id}})["answer"]
        return {"response": response}

    uvicorn.run(app, host="localhost", port=8000)


if __name__ == "__main__":
    session_id = "session_1"
    run_console_chat(stream=False, session_id=session_id)
    # serve_api_endpoint()
