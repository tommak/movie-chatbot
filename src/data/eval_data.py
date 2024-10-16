from langsmith import Client
import json


if __name__ == "__main__":

    dataset_name = "Movies QA"
    path = "./data/eval/movies_qa.json"
    with open(path, 'r') as file:
        data = json.load(file)
    
    client = Client()
    dataset = client.create_dataset(dataset_name)
    client.create_examples(
        inputs=[{"question": q["Question"]} for q in data],
        outputs=[{"answer": ans["Answer"]} for ans in data],
        dataset_id=dataset.id,
    )

    dataset_name = "Movies Data QA"
    path = "./data/eval/movies_data_qa.json"
    with open(path, 'r') as file:
        data = json.load(file)
    
    client = Client()
    dataset = client.create_dataset(dataset_name)
    client.create_examples(
        inputs=[{"question": q["Question"]} for q in data],
        outputs=[{"answer": ans["Answer"]} for ans in data],
        dataset_id=dataset.id,
    )