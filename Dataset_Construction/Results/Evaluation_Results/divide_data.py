"""
divide evaluated data to train set and test set
"""

from utils import *

directory = r"Dataset_Construction/Results/Evaluation_Results/gpt-35-turbo/Original_Split_Data/chain-of-thought/"
question_file_path = directory + r"gpt-35-turbo_question.json"
response_chatdoctor_file_path = directory + r"gpt-35-turbo_response_chatdoctor.json"
response_baize_file_path = directory + r"gpt-35-turbo_response_baize.json"
response_reference_file_path = directory + r"gpt-35-turbo_response_reference.json"
evaluation_file_path = directory + r"gpt-35-turbo_evaluation.json"
question = load_json_file(question_file_path)
response_chatdoctor = load_json_file(response_chatdoctor_file_path)
response_baize = load_json_file(response_baize_file_path)
response_reference = load_json_file(response_reference_file_path)
evaluation = load_json_file(evaluation_file_path)

dataset = []
for i in range(len(question)):
    dataset.append({
        "question": question[i],
        "response_chatdoctor": response_chatdoctor[i],
        "response_baize": response_baize[i],
        "response_reference": response_reference[i],
        "evaluation": evaluation[i]
    })
train_dataset = dataset[:9000]
test_dataset = dataset[9000:]
train_file_path = r"Model_Training/Dataset/Train/gpt-35-turbo-chain-of-thought-9000.json"
test_file_path = r"Model_Training/Dataset/Test/gpt-35-turbo-chain-of-thought-1000.json"
save_json_file(train_file_path, train_dataset)
save_json_file(test_file_path, test_dataset)