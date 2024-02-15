"""
merge evaluation results and split by key
"""
from utils import *


def split_json_files(file: List[dict], directory: str, model_name: str):
    question = []
    response_chatdoctor = []
    response_baize = []
    response_reference = []
    evaluation = []
    for ds in file:
        question.append(ds["input"])
        response_chatdoctor.append(ds["response_chatdoctor"])
        response_baize.append(ds["response_baize"])
        response_reference.append(ds["reference"])
        evaluation.append(ds[model_name])
    save_json_file(os.path.join(directory, f"{model_name}_question.json"), question)
    save_json_file(os.path.join(directory, f"{model_name}_response_chatdoctor.json"), response_chatdoctor)
    save_json_file(os.path.join(directory, f"{model_name}_response_baize.json"), response_baize)
    save_json_file(os.path.join(directory, f"{model_name}_response_reference.json"), response_reference)
    save_json_file(os.path.join(directory, f"{model_name}_evaluation.json"), evaluation)


if __name__ == "__main__":
    files_path = r"Dataset_Construction/Results/Evaluation_Results/gpt-35-turbo/gpt-35-turbo_chain-of-thought_10000"
    files_name = find_json_files(files_path)
    data = union_json_files(files_name)
    split_json_files(data,
                     r".\Dataset_Construction\Results\Evaluation_Results\gpt-35-turbo\Original_Split_Data",
                     "gpt-35-turbo")