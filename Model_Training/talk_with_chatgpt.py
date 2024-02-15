import time
import openai

from tqdm import tqdm
from Model_Training.prompt import PROMPT_DICT
from Model_Training.Evaluate.evaluate import extract_evaluation_result_from_json, check_result_correctness_3
from utils import *

openai.api_key = ""


def completion(messages):
    completions = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=256,
        temperature=0.7,
    )
    return completions.choices[0]['message']['content']


# 获取chatgpt的建议
def get_gpt_suggestions(filepath: str, original_filepath: str):
    prompt = "You are an expert in the medical field. You know the causes and symptoms of most diseases. You know what " \
             "treatment should be used for each disease. You also know the effects and side effects of various medicines and treatments." \
             "The following article includes a medical related question, the responses of three doctors to this " \
             "medical question, as well as the analysis, evaluation, and scoring of the responses of the three " \
             "doctors. But the answer in the evaluation may be inaccurate. Therefore, after reading the following article," \
             " you need to provide improvement suggestions for each answer in the Evaluation. ### Your output has to follow the following " + \
            'format: "Suggestions for comments on Doctor 1: xxx. ' \
            'Suggestions for comments on Doctor 2: xxx. ' \
            'Suggestions for comments on Doctor 3: xxx.'
    cases = load_json_file(filepath)
    original_file = load_json_file(original_filepath)
    result_file = []
    for i in range(len(cases)):
        original_file[i]["output"] = cases[i]["output"]
    score, evaluation = extract_evaluation_result_from_json(original_file)
    for i, case in tqdm(enumerate(original_file), total=len(original_file)):
        if len(check_result_correctness_3([score[i]], [evaluation[i]])) == 0:
            continue
        inputs = PROMPT_DICT["prompt_for_suggestions"].format_map(original_file[i])
        msg = [{"role": "system", "content": prompt}, {"role": "user", "content": inputs}]
        while True:
            try:
                response = completion(msg)
                suggestion = get_suggestion(response)
                if suggestion is None:
                    continue
                sug1, sug2, sug3 = suggestion
                original_file[i]["suggestion1"] = sug1
                original_file[i]["suggestion2"] = sug2
                original_file[i]["suggestion3"] = sug3
                result_file.append(original_file[i].copy())
                break
            except Exception as e:
                print(str(e))
                print("Retrying...")
                time.sleep(2)
    return original_file


# 从文本中提取对模型的建议
def get_suggestion(text):
    try:
        sug1 = text.split('Suggestions for comments on Doctor 1: ')[-1].split('Suggestions for comments on Doctor 2:')[0]
        sug2 = text.split('Suggestions for comments on Doctor 2: ')[-1].split('Suggestions for comments on Doctor 3:')[0]
        sug3 = text.split('Suggestions for comments on Doctor 3: ')[-1]
    except:
        return None
    return [sug1.strip(), sug2.strip(), sug3.strip()]


def get_advice_and_save(file_path: str, original_file_path: str, save_path: str):
    file_with_suggestions = get_gpt_suggestions(file_path, original_file_path)
    save_json_file(save_path, file_with_suggestions)


if __name__ == "__main__":
