# Movie Chatbot

This chatbot is designed to answer questions about movies and TV series using the latest context and general knowledge.

## Features

- Answers questions about movies and TV series.
- Prioritizes the latest movie information.
- Uses context from previous chat history to provide accurate responses.
- Can be run in console mode or served as an API endpoint.


### Environment Variables

Set the following environment variables:

```bash
export LANGCHAIN_API_KEY="your_langchain_api_key"
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
export LANGCHAIN_PROJECT="movies-chatbot"
export OPENAI_API_KEY="your_openai_api_key"
```


## Usage

### Console Mode

In console mode, you can interact with the chatbot directly in the terminal. Type your questions and get responses about movies and TV series.

### API Mode

In API mode, you can send POST requests to the `/prompt` endpoint with the following parameters:

- `prompt`: The question you want to ask.
- `session_id`: The session ID for maintaining chat history.

Example request:

```bash
curl -X POST "http://localhost:8000/prompt" -d "prompt=Tell me about the latest Marvel movie" -d "session_id=session_1"
```

## License

This project is licensed under the Apache-2.0 license.

## Acknowledgements

- [LangChain](https://python.langchain.com/v0.2/docs/tutorials/)
- [OpenAI](https://www.openai.com/)
