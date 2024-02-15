"""
constructs four different types of instructions and dialogues
1: common_prompt
2: chain_of_thought
3: chain_of_keyword
4: search_in_chain
"""

import json
import re

import yaml
from tqdm import tqdm
from Dataset_Construction.PDF.knowledge_extractor import KnowledgeExtractor
from Dataset_Construction.chat_complete import request

knowledge_extractor = KnowledgeExtractor()


def common_prompt(query, response_chatdoctor, response_baize, reference):

    """
    instruction = "You are an evaluator of conversations in the medical field. " + \
                  "After reading a conversation between the patient and the robots, " + \
                  "you need to score each robot's response from 1 point to 5 points " + \
                  "based on the medical knowledge you learned. And then " + \
                  "give reasons for scoring. The output format should be like this:\n" + \
                  "Response 1: X points. Response 2: Y points. Response 3: Z points. \n" + \
                  "Reason:xxxxxxxxx.\n"
    """
    """
    instruction = "You are an expert in the medical field. You know the causes and symptoms of most diseases. You " + \
                  "know what treatment should be used for each disease. You also know the effects and side effects " + \
                  "of various medicines and treatments. After reading a conversation between the patient and " + \
                  "doctors, you need to score each doctor's response from 1 point to 5 points based on the medical " + \
                  "knowledge you learned. And then give reasons for scoring. The output format should be like this:\n" + \
                  "Doctor 1: X points. Doctor 2: Y points. Doctor 3: Z points.\n" + \
                  "Reason:xxxxxxxxx."
    """
    instruction = "You are an expert in the medical field. You know the causes and symptoms of most diseases. You " + \
                  "know what treatment should be used for each disease. You also know the effects and side effects " + \
                  "of various medicines and treatments. After reading a conversation between the patient and " + \
                  "doctors, you need to analyze the quality of each doctor's response based on the medical knowledge you learned. " + \
                  "After analyzing, score each doctor's response from 1 point to 5 points based on the analyze. " + \
                  "The output format should be like this:\n\nAnalyze: ......\n" + \
                  "Score: Doctor 1: X points. Doctor 2: Y points. Doctor 3: Z points."
    dialogue = f"Patient: \"{query}\"\nDoctor 1: \"{response_chatdoctor}\"\nDoctor 2: \"{response_baize}\"\nDoctor 3: \"{reference}\"\n\n"
    msg = [{"role": "system", "content": instruction}, {"role": "user", "content": dialogue}]
    return msg, instruction, dialogue


def chain_of_thought(query, response_chatdoctor, response_baize, reference):
    instruction = "You are an expert in the medical field. You know the causes and symptoms of most diseases. You " + \
                  "know what treatment should be used for each disease. You also know the effects and side effects " + \
                  "of various medicines and treatments. After reading a conversation " + \
                  "between the patient and doctors, you need to understand the patient's question and analyze " + \
                  "doctors' responses step by step based on what you have learned. " + \
                  "After analyzing, score each doctor's response from 1 point to 5 points" + \
                  " based on the analyze. " + \
                  "The output format should be like this:\n\nAnalyze: \nStep 1: ... \nStep 2: ... \nStep 3: ......\n" + \
                  "Score: Doctor 1: X points. Doctor 2: Y points. Doctor 3: Z points.\n"
    dialogue = "Here is an example:\n\nPatient: \"Can you provide an overview of the lung's squamous cell " + \
               "carcinoma?\"\nDoctor 1: \"Squamous cell carcinoma is a type of cancer that develops in the " + \
               "cells lining the bronchi and other areas of the lung. It is usually caused by smoking or exposure " + \
               "to other risk factors such as asbestos, radiation, and air pollution. Symptoms may include " + \
               "coughing, difficulty breathing, chest pain, weight loss, and fatigue. Treatment options include " + \
               "surgery, chemotherapy, and radiation therapy.\"\nDoctor 2: \"Squamous cell carcinoma is a type " + \
               "of cancer that begins in the squamous cells, which are the thin, flat cells that line the inside " + \
               "of the lungs. It is the second most common type of lung cancer, and it is more common in smokers." + \
               "\"\nDoctor 3: \"Squamous cell carcinoma of the lung may be classified according to the WHO " + \
               "histological classification system into 4 main types: papillary, clear cell, small cell, and " + \
               "basaloid.\"\nAnalyze: Step 1: The patient is asking for an overview of lung's squamous cell carcinoma.\n" + \
               "Step 2: Doctor 1 provides a detailed overview of squamous cell carcinoma, including its causes, " + \
               "symptoms, and treatment options. It also mentions that it develops in the cells lining the bronchi " + \
               "and other areas of the lung.\nStep 3: Doctor 2 also provides an overview of squamous cell " + \
               "carcinoma, but it focuses more on the type of cells it affects and its prevalence in smokers.\n" + \
               "Step 4: Doctor 3 provides information on the classification of squamous cell carcinoma according " + \
               "to the WHO histological classification system.\nScore: Doctor 1: 5 points. Doctor 2: 4 points. " + \
               "Doctor 3: 3 points. \n\n"
    dialogue += f"Here are the conversations you need to evaluate:\nPatient: \"{query}\"\nDoctor 1: \"{response_chatdoctor}\"\nDoctor 2: \"{response_baize}\"\nDoctor 3: \"{reference}\"\n"
    msg = [{"role": "system", "content": instruction}, {"role": "user", "content": dialogue}]
    return msg, instruction, dialogue


def chain_of_keyword(query, response_chatdoctor, response_baize, reference, model_name, api_key):
    prompt = f"<{query}>What is the keywords of this sentence? please answer with the format of [keyword 1]、" + \
             "[keyword 2]、[keyword 3], for example: [keyword 1]: Knowledge graph, [keyword 2]: semantics net.\n" + \
             "Attention：The question you answer must be a certain entity (Entity), for example: Knowledge graph, " + \
             "semantics net, but not: building, related, etc."
    res = request(model_name, prompt, api_key=api_key)[0].replace("\n", ", ")
    keywords = re.findall(r'\[keyword \d+\]: (.*?)(?:,|$)', res)
    instruction = "You are an expert in the medical field. You know the causes and symptoms of most diseases. You " + \
                  "know what treatment should be used for each disease. You also know the effects and side effects " + \
                  "of various medicines and treatments. After reading a conversation between the patient and " + \
                  "doctors, you need to analyze the quality of each doctor's response " + \
                  "based on the medical knowledge you learned and the [Keyword]-[Infos] pairs " + \
                  "of chain of keyword that I provided for you. After analyzing, score each doctor's response from " + \
                  "1 point to 5 points based on the analyze. The output format should be like this:\n\n" + \
                  "Analyze: ......\n" + \
                  "Score: Doctor 1: X points. Doctor 2: Y points. Doctor 3: Z points."

    dialogue = ""
    for i in range(len(keywords)):
        q = f"What is {keywords[i]}?"
        dialogue += f"[Keyword {i+1}]: {q}\n"
        knowledge = knowledge_extractor.get_related_knowledge(keywords[i], "medical-knowledge-pdf")
        information = ""
        for j in range(len(knowledge)):
            information += f"[Info of Keyword <{keywords[i]}> {j + 1}]: {knowledge[j]}\n"
        dialogue += f"[Infos {i+1}]:\n{information}\n"
    dialogue += f"\nHere are the conversations you need to evaluate:\nPatient: \"{query}\"\nDoctor 1: \"{response_chatdoctor}\"\nDoctor 2: \"{response_baize}\"\nDoctor 3: \"{reference}\"\n"
    msg = [{"role": "system", "content": instruction}, {"role": "user", "content": dialogue}]
    return msg, instruction, dialogue


def search_in_chain(query, response_chatdoctor, response_baize, reference, model_name, api_key):
    limit = 3
    instruction = "You are an expert in the medical field. You know the causes and symptoms of most diseases. You " + \
                  "know what treatment should be used for each disease. You also know the effects and side effects " + \
                  "of various medicines and treatments. After reading a conversation between the patient and " + \
                  "doctors, you need to understand the patient's question and analyze the quality of each " + \
                  "doctor's response step by step based on what you have learned. Before generating the final score, " + \
                  "if you are unsure about something, you can ask questions in the format of \"[Question]: What is " + \
                  "...?\", so that I can provide extra knowledge for the thing you don't sure. \nAttention: After asking the question starts with [Question] or generating the final score, you " + \
                  "need to stop generation immediately." + \
                  "Your output should follow the format like this:\n\nAnalyze:\nStep 1: ...\nStep 2: ...\n" + \
                  "Step 3: ...\nScore: Doctor 1: X points. Doctor 2: Y points. " + \
                  "Doctor 3: Z points.\n"
    dialogue = "Here is an example:\nPatient: \"Can you provide an overview of the lung's squamous cell " + \
               "carcinoma?\"\nDoctor 1: \"Squamous cell carcinoma is a type of cancer that develops in the " + \
               "cells lining the bronchi and other areas of the lung. It is usually caused by smoking or exposure " + \
               "to other risk factors such as asbestos, radiation, and air pollution. Symptoms may include " + \
               "coughing, difficulty breathing, chest pain, weight loss, and fatigue. Treatment options include " + \
               "surgery, chemotherapy, and radiation therapy.\"\nDoctor 2: \"Squamous cell carcinoma is a type " + \
               "of cancer that begins in the squamous cells, which are the thin, flat cells that line the inside " + \
               "of the lungs. It is the second most common type of lung cancer, and it is more common in smokers." + \
               "\"\nDoctor 3: \"Squamous cell carcinoma of the lung may be classified according to the WHO " + \
               "histological classification system into 4 main types: papillary, clear cell, small cell, and " + \
               "basaloid.\"\n\nAnalyze:\nStep 1: The patient is asking for an overview of lung's squamous cell carcinoma.\n" + \
               "Step 2: Doctor 1 provides a detailed overview of squamous cell carcinoma, including its causes, " + \
               "symptoms, and treatment options. It also mentions that it develops in the cells lining the bronchi " + \
               "and other areas of the lung.\nStep 3: Doctor 2 also provides an overview of squamous cell " + \
               "carcinoma, but it focuses more on the type of cells it affects and its prevalence in smokers.\n" + \
               "Step 4: Doctor 3 provides information on the classification of squamous cell carcinoma according " + \
               "to the WHO histological classification system.\nScore: Doctor 1: 5 points. Doctor 2: 4 points. " + \
               "Doctor 3: 3 points. \n\n"
    dialogue += f"Here are the conversations you need to evaluate:\nPatient: \"{query}\"\nDoctor 1: \"{response_chatdoctor}\"\nDoctor 2: \"{response_baize}\"\nDoctor 3: \"{reference}\"\n\n"
    msg = [{"role": "system", "content": instruction}, {"role": "user", "content": dialogue}]
    final_answer = ""
    for i in range(limit):
        res = request(model_name, msg, api_key=api_key)[0]
        score = [int(s.split(" ")[-1]) for s in re.findall("Doctor \d: \d", res)]
        if "[Question]: " in res and len(score) == 0:
            print(f"detect question {i+1} times.")
            question = res.split('[Question]: ')[-1].strip()
            knowledge = knowledge_extractor.get_related_knowledge(question, "medical-knowledge-pdf", top_k=1)[0]
            dialogue += f"According to the Reference, the related knowledge for \"{question}\" is \"{knowledge}\", " + \
                        "you can give your answer and continue constructing the reasoning chain.\n"
            msg = [{"role": "system", "content": instruction}, {"role": "user", "content": dialogue}]
        else:
            final_answer = res
            break
    return final_answer, instruction, dialogue


def build_message(query, response_chatdoctor, response_baize, reference, method, model_name=None):
    with open('./Dataset_Construction/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        if model_name is None:
            model_name = config['openai']['generate_model']['model_name']
        generate_choices = config['openai']['generate_choices']
    openai_api_key = ""
    for generate_choice in generate_choices:
        if generate_choice['model_name'] == model_name:
            openai_api_key = generate_choice['api_key']
    message = ""
    instruction = None
    dialogue = None
    if method == "common_prompt":
        message, instruction, dialogue = common_prompt(query, response_chatdoctor, response_baize, reference)
    elif method == "chain_of_thought":
        message, instruction, dialogue = chain_of_thought(query, response_chatdoctor, response_baize, reference)
    elif method == "chain_of_keyword":
        message, instruction, dialogue = chain_of_keyword(query, response_chatdoctor, response_baize, reference, model_name, openai_api_key)
    elif method == "search_in_chain":
        message, instruction, dialogue = search_in_chain(query, response_chatdoctor, response_baize, reference, model_name, api_key=openai_api_key)
    return message, instruction, dialogue


if __name__ == '__main__':
    with open("./Dataset_Construction/Results/Response_Results/merged_response_data.json", "r", encoding='utf-8') as f:
        data = json.load(f)
    for i, ds in enumerate(tqdm(data.copy())):
        if i == 1000:
            break
        msg, ins, dia = build_message(ds["input"], ds["response_chatdoctor"], ds["response_baize"], ds["reference"], "search_in_chain")

