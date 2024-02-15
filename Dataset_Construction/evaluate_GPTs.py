import json
import yaml

from tqdm import tqdm
from Dataset_Construction.msg_builder import build_message
from Dataset_Construction.chat_complete import request
from Dataset_Construction.Results.Evaluation_Results.result_format import result_format


def save_and_format_file(model_name, msg_method, st, ed, data):
    sst = "%05d" % st
    eed = "%05d" % ed
    filepath = f"./Dataset_Construction/Results/Evaluation_Results/{model_name}/result_rank_{msg_method}_{sst}_{eed}_{model_name}.json"
    data = data[st:ed]
    with open(filepath, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False)
    result_format(filepath)


def main():
    with open('./Dataset_Construction/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        model_name = config['openai']['generate_model']['model_name']
        generate_choices = config['openai']['generate_choices']
        st, ed = config['openai']['generate_ids']
        msg_methods = config['openai']['prompt_method']
    openai_api_key = ''
    for generate_choice in generate_choices:
        if generate_choice['model_name'] == model_name:
            openai_api_key = generate_choice['api_key']
    generate_data_name = config['openai']['generate_data_name']
    with open(f"./Dataset_Construction/Results/Response_Results/merged_response_data_{generate_data_name}.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    lt = st
    for msg_method in msg_methods:
        for i, ds in enumerate(tqdm(data.copy())):
            if i < st:
                continue
            if i == ed:
                break
            message, res_instruction, res_input = build_message(ds["input"], ds["response_chatdoctor"], ds["response_baize"], ds["reference"], msg_method)
            if msg_method != "search_in_chain":
                response = request(
                    model=model_name,
                    messages=message,
                    api_key=openai_api_key,
                )
                data[i][model_name] = response[0]
            else:
                data[i][model_name] = message
            if (i+1) % 100 == 0:
                save_and_format_file(model_name, msg_method, lt, i+1, data)
                lt = i+1
        if lt != ed:
            save_and_format_file(model_name, msg_method, lt, ed, data)


if __name__ == '__main__':
    main()
