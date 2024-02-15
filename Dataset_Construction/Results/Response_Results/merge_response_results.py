"""
merge response results from chatdoctor and baize
"""

import json

from tqdm import tqdm


def get_data(path: str):
    with open(path, "r", encoding='utf-8') as f:
        file = json.load(f)
    return file


def merge_data():
    data = get_data('Dataset_Construction/Results/Response_Results/medical_meadow/medical_meadow_wikidoc.json')
    response1 = get_data('Dataset_Construction/Results/Response_Results/medical_meadow/response_chatdoctor.json')
    response2 = get_data('Dataset_Construction/Results/Response_Results/medical_meadow/response_baize.json')
    reference = get_data('Dataset_Construction/Results/Response_Results/medical_meadow/reference.json')
    merged_datas = []
    for i, data in enumerate(tqdm(data)):
        del data["output"]
        response1[i] = response1[i].strip()
        response2[i] = response2[i].strip()
        reference[i] = reference[i].strip()
        if response1[i].endswith("\n"):
            response1[i] = response1[i][:-1]
        if response2[i].endswith("\n"):
            response2[i] = response2[i][:-1]
        if reference[i].endswith("\n"):
            reference[i] = reference[i][:-1]
        response1[i] = response1[i].replace("\n", " ")
        response2[i] = response2[i].replace("\n", " ")
        reference[i] = reference[i].replace("\n", " ")
        data["response_chatdoctor"] = response1[i][:1000]
        data["response_baize"] = response2[i][:1000]
        data["reference"] = reference[i][:1000]
        merged_datas.append(data)
    with open('Dataset_Construction/Results/Response_Results/merged_response_data_medical_meadow.json', "w", encoding='utf-8') as f:
        json.dump(merged_datas, f)


if __name__ == '__main__':
    merge_data()

