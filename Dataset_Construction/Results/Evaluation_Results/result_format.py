"""
unify the format of the result file
"""

import json
import os
import re


def filereader(filepath, model_name):
    with open(filepath, "r", encoding='utf-8') as f:
        file = json.load(f)
        scores = []
        for i in range(len(file)):
            result = file[i][model_name]
            score = [int(s.split(" ")[-1]) for s in re.findall("Doctor \d: \d", result)]
            scores.append(score)
    return scores


def result_format(filepath):
    path = filepath
    filename = path.split("/")[-1].split(".")[0]
    model_name = filename.split("_")[-1]
    st = int(filename.split("_")[-3])
    result = filereader(os.path.realpath(path), model_name)
    result_path = os.path.join(os.path.dirname(path), filename+".txt")
    with open(result_path, "w", encoding='utf-8') as f:
        for i in range(len(result)):
            score = result[i]
            if len(score) != 3:
                f.write(f"Case{i+st} Error: " + str(score) + "\n")
                continue
            f.write(f"Case{i+st}: {score[0]} {score[1]} {score[2]}\n")


if __name__ == "__main__":
    result_format("./Dataset_Construction/Results/Evaluation_Results/chatGPT/result_rank_common_prompt_50_chatGPT.json")

