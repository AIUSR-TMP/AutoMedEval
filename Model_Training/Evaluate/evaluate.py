import itertools
import bert_score
import statistics

import numpy as np

from Model_Training.Evaluate.BARTScore.bart_score import BARTScorer
from utils import *


def get_bert_score_from_json_result(json_result: dict):
    candidates = []
    references = []
    for item in json_result:
        candidates.append(item["output"])
        references.append(item["ground_truth"])
    P, R, F1 = bert_score.score(candidates, references, lang="en")
    mean_P, mean_R, mean_F1 = statistics.mean(P.tolist()), statistics.mean(R.tolist()), statistics.mean(F1.tolist())
    print(f"mean bert score: P: {mean_P}, R: {mean_R}, F1: {mean_F1}")


def get_bart_score_from_json_result(json_result: dict):
    candidates = []
    references = []
    for item in json_result:
        candidates.append(item["output"])
        references.append(item["ground_truth"])
    bart_scorer = BARTScorer()
    score = bart_scorer.score(candidates, references)
    mean_score = statistics.mean(score)
    print(f"mean bart score: {mean_score}")


def calculate_format_correct_result(file: List[dict]):
    r"""
    files_path = r"Model_Training/Inference/Results/medllama_new-gpt4-search-in-chain-9000_checkpoint-211_split-gpt4-search-in-chain-1w_1000.json"
    calculate_format_correct_result(load_json_file(files_path))
    """
    more = 0
    less = 0
    for ds in file:
        ouptut = ds["output"]
        score = [int(s.split(" ")[-1]) for s in re.findall("Response \d: \d", ouptut)]
        if len(score) > 3:
            more += 1
        elif len(score) < 3:
            less += 1
    print(f"correct: {len(file) - more - less}, more: {more}, less: {less}")


def extract_output_and_evaluation(data: dict, key_name: str):
    return [list(map(int, s.split(": "))) for s in re.findall("Doctor (\d: \d)", data[key_name])]


def extract_evaluation_result_from_json(file: List[dict]):
    final_score = []
    answers = []
    more = 0
    less = 0
    final_score_less_list = []
    answer_less_list = []
    for ds in file:
        score = extract_output_and_evaluation(ds, "output")
        if "ground_truth" in ds:
            answer = extract_output_and_evaluation(ds, "ground_truth")
        else:
            answer = extract_output_and_evaluation(ds, "evaluation")
        flag1 = False
        flag2 = False
        for i in range(len(score)):
            if score[i][0] != i + 1:
                break
            if i == 2:
                flag1 = True
                break
        for i in range(len(answer)):
            if answer[i][0] != i + 1:
                break
            if i == 2:
                flag2 = True
                break
        if flag1 and flag2:
            final_score.append(list(np.array(score)[:3, 1]))
            answers.append(list(np.array(answer)[:3, 1]))
            if len(score) > 3:
                more += 1
        elif flag1 and not flag2:
            final_score.append(list(np.array(score)[:3, 1]))
            answers.append([])
            answer_less_list.append(ds)
            less += 1
        elif not flag1 and flag2:
            final_score.append([])
            answers.append(list(np.array(answer)[:3, 1]))
            final_score_less_list.append(ds)
            less += 1
        else:
            final_score.append([])
            answers.append([])
            answer_less_list.append(ds)
            final_score_less_list.append(ds)
            less += 1

    # for ds in less_list:
    #     print(ds["output"])
    print(f"more : {more}, less : {less}")
    return final_score, answers


def check_result_correctness_3(final_score: List[List[int]], answers: List[List[int]]):
    incorrect = []
    correct = []
    for i in range(len(final_score)):
        if len(answers[i]) != 3:
            print(f"answer {i} is not correct")
            continue
        if len(final_score[i]) != 3:
            incorrect.append(i)
            continue
        d, e, f = final_score[i][0], final_score[i][1], final_score[i][2]
        a, b, c = answers[i][0], answers[i][1], answers[i][2]
        if a == b or a == c or b == c:
            flag = True
            if a == b and b == c:
                if d != e or e != f:
                    flag = False
            elif a == b:
                if c > a:
                    if d != e or d >= f:
                        flag = False
                else:
                    if d != e or d <= f:
                        flag = False
            elif a == c:
                if b > a:
                    if d != f or d >= e:
                        flag = False
                else:
                    if d != f or d <= e:
                        flag = False
            elif b == c:
                if a > b:
                    if e != f or e >= d:
                        flag = False
                else:
                    if e != f or e <= d:
                        flag = False
        else:
            a = [0, 1, 2]
            perms = list(itertools.permutations(a))
            flag = False
            for perm in perms:
                if final_score[i][perm[0]] > final_score[i][perm[1]] > final_score[i][perm[2]] and answers[i][perm[0]] > answers[i][perm[1]] > answers[i][perm[2]]:
                    flag = True
                    break
        if not flag:
            incorrect.append(i)
        else:
            correct.append(i)
    print(f"correct: {len(correct)}, total: {len(correct) + len(incorrect)}")
    return incorrect


def compare_2(a, b, c, d):
    if a == b:
        if c == d:
            return True
        else:
            return False
    elif a < b:
        if c < d:
            return True
        else:
            return False
    else:
        if c > d:
            return True
        else:
            return False


def check_result_correctness_2(evaluation: List[List[int]], answers: List[List[int]]):
    incorrect = 0
    correct = 0
    for i in range(len(evaluation)):
        if len(answers[i]) != 3:
            print(f"answer {i} is not correct")
            continue
        if len(evaluation[i]) != 3:
            incorrect += 3
            continue
        a, b, c = evaluation[i][0], evaluation[i][1], evaluation[i][2]
        d, e, f = answers[i][0], answers[i][1], answers[i][2]
        if compare_2(a, b, d, e):
            correct += 1
        else:
            incorrect += 1
        if compare_2(a, c, d, f):
            correct += 1
        else:
            incorrect += 1
        if compare_2(b, c, e, f):
            correct += 1
        else:
            incorrect += 1
    print(f"correct: {correct}, total: {correct + incorrect}")
    return incorrect


def do_evaluate(path: str):
    if os.path.isfile(path):
        paths = [path]
    else:
        paths = find_json_files(path)
    for path in paths:
        # if "-1.json" not in path:
        #     continue
        print(path)
        json_result = load_json_file(path)
        get_bert_score_from_json_result(json_result)
        get_bart_score_from_json_result(json_result)
        final_score, answers = extract_evaluation_result_from_json(json_result)
        check_result_correctness_3(final_score, answers)
        check_result_correctness_2(final_score, answers)


def extract_incorrect_cases(file: List[dict], incorrect: List):
    incorrect_file = []
    for idnex in incorrect:
        incorrect_file.append(file[idnex])
    return incorrect_file


def merge_incorrect_from_evaluation_result_and_original_file(path_result: str, path_original: str):
    file_result = load_json_file(path_result)
    evaluation, answers = extract_evaluation_result_from_json(file_result)
    incorrect = check_result_correctness_3(evaluation, answers)
    original_file = load_json_file(path_original)
    incorrect_file = []
    for i in range(len(incorrect)):
        incorrect_file.append(original_file[incorrect[i]])
        incorrect_file[-1]["output"] = file_result[incorrect[i]]["output"]
    return incorrect_file


def cal_abs_diff_score(filepath: str):
    file_result = load_json_file(filepath)
    evaluation, answers = extract_evaluation_result_from_json(file_result)
    score = 0
    for i in range(len(evaluation)):
        for j in range(3):
            score += abs(evaluation[i][j] - answers[i][j])
    print(f"模型输出与标准答案分差为: {score}")
    return score


def evaluate_pandalm(score_path: str, test_path: str):
    score = load_json_file(score_path)
    test_data = load_json_file(test_path)
    ref_score = []
    for ds in test_data:
        answer = [list(map(int, s.split(": "))) for s in re.findall("Doctor (\d: \d)", ds["evaluation"])]
        flag = False
        for i in range(len(answer)):
            if answer[i][0] != i + 1:
                break
            if i == 2:
                flag = True
                break
        if flag:
            ref_score.append(list(np.array(answer)[:3, 1]))
        else:
            ref_score.append([])
    """
    chatdoctor，baize
    chatdoctor，reference
    baize，reference
    """
    final_score = []
    for i in range(len(score[0])):
        score_chatdoctor = 0
        score_baize = 0
        score_reference = 0
        if score[0][i] == 0:
            score_chatdoctor += 1
            score_baize += 1
        elif score[0][i] == 1:
            score_chatdoctor += 1
        elif score[0][i] == 2:
            score_baize += 1
        if score[1][i] == 0:
            score_chatdoctor += 1
            score_reference += 1
        elif score[1][i] == 1:
            score_chatdoctor += 1
        elif score[1][i] == 2:
            score_reference += 1
        if score[2][i] == 0:
            score_baize += 1
            score_reference += 1
        elif score[2][i] == 1:
            score_baize += 1
        elif score[2][i] == 2:
            score_reference += 1
        final_score.append([score_chatdoctor, score_baize, score_reference])
    for i in range(len(final_score)):
        flag = True
        if score[0][i] == 0 and final_score[i][0] != final_score[i][1]:
            flag = False
        if score[1][i] == 0 and final_score[i][0] != final_score[i][2]:
            flag = False
        if score[2][i] == 0 and final_score[i][1] != final_score[i][2]:
            flag = False
        if score[0][i] == 1 and final_score[i][0] <= final_score[i][1]:
            flag = False
        if score[1][i] == 1 and final_score[i][0] <= final_score[i][2]:
            flag = False
        if score[2][i] == 1 and final_score[i][1] <= final_score[i][2]:
            flag = False
        if score[0][i] == 2 and final_score[i][0] >= final_score[i][1]:
            flag = False
        if score[1][i] == 2 and final_score[i][0] >= final_score[i][2]:
            flag = False
        if score[2][i] == 2 and final_score[i][1] >= final_score[i][2]:
            flag = False
        if not flag:
            final_score[i] = []
            print(i)
    check_result_correctness_3(final_score, ref_score)
    check_result_correctness_2(final_score, ref_score)
    return final_score


if __name__ == "__main__":
    do_evaluate(r"Model_Training/Inference/Results/gpt-35-turbo_search-in-chain_9000/medllama-gpt-35-turbo-search-in-chain-9000_gpt-4-search-in-chain-1000_prompt_no_input_-1.json")
