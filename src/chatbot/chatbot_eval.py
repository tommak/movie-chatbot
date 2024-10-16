from pathlib import Path
import datetime as dt
from langchain import hub
from chatbot import setup_chain
from langchain_openai import ChatOpenAI

from langsmith.evaluation import evaluate


import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "movies-chatbot"

# Grade prompt
grade_prompt_answer_accuracy = prompt = hub.pull("langchain-ai/rag-answer-vs-reference")

def answer_vs_ref_evaluator(run, example) -> dict:
    """
    A simple evaluator for RAG answer accuracy
    """

    # Get question, ground truth answer, RAG chain answer
    input_question = example.inputs["question"]
    reference = example.outputs["answer"]
    prediction = run.outputs["answer"]

    # LLM grader
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)

    # Structured prompt
    answer_grader = grade_prompt_answer_accuracy | llm

    # Run evaluator
    score = answer_grader.invoke({"question": input_question,
                                  "correct_answer": reference,
                                  "student_answer": prediction})
    score = score["Score"]

    return {"key": "answer_v_reference_score", "score": score}


def chatbot_app(example: dict):

    chain_config = {
        "llm_config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0
        },
        "RAG": {
            "context_group_name": "movies",
            "context_path": "./cache/context",
            "context_version": "openai_dver01_indver01"

        }
    }
    current_date = dt.datetime.now().strftime("%Y-%m-%d")
    chain = setup_chain(chain_config)
    output = chain.invoke({"input": example["question"], "current_date": current_date},
                          config={"configurable": {"session_id": None}})#["answer"]
    return output

def run_evaluation(chain_config: dict, dataset_name:str):

    experiment_results = evaluate(
        chatbot_app,
        data=dataset_name,
        evaluators=[answer_vs_ref_evaluator],
        experiment_prefix="rag-answer-v-reference",
        metadata=chain_config,
    )
    return experiment_results

    

if __name__ == "__main__":

    chain_config = {
        "llm_config": {
            "model": "gpt-3.5-turbo",
            "temperature": 0
        },
        "RAG": {
            "context_group_name": "movies",
            "context_path": "./cache/context",
            "context_version": "openai_dver01_indver01"

        }
    }

    # dataset_name = "Movies QA"
    # experiment_results = run_evaluation(chain_config, dataset_name)

    dataset_name = "Movies Data QA"
    experiment_results = run_evaluation(chain_config, dataset_name)

    # save_path = Path("/Users/tamara/Documents/Projects/movie-chatbot/output/eval_runs")
    # save_name = "rag_movies_qa"
    # with open(save_path / f"{save_name}.pkl", "wb") as f:
    #     pickle.dump(experiment_results, f)


    
