PROMPT_DICT = {
    "prompt_input": (
        "Below are some responses for a given task. The task is defined by the Instruction with an input that provides further context. "
        "Evaluate the responses and generate a score for each response.\n\n"
        "### Instruction:\n{question}\n\n### Input:\n{input}\n\n### Response 1:\n{response_chatdoctor}\n\n"
        "### Response 2:\n{response_baize}\n\n### Response 3:\n{response_reference}\n\n"
        "### Evaluation:\n"
    ),
    "prompt_no_input": (
        "Below are some responses for a given task. The task is defined by the Instruction. "
        "Evaluate the responses and generate a score for each response.\n\n"
        "### Instruction:\n{question}\n\n### Response 1:\n{response_chatdoctor}\n\n"
        "### Response 2:\n{response_baize}\n\n### Response 3:\n{response_reference}\n\n"
        "### Evaluation:\n"
    ),
    "prompt_suggestion": (
        "Below are some responses for a given task. The task is defined by the Instruction. "
        "Evaluate the responses with the suggestions and generate a score for each response.\n\n"
        "### Instruction:\n{question}\n\n### Response 1:\n{response_chatdoctor}\n\n"
        "### Response 2:\n{response_baize}\n\n### Response 3:\n{response_reference}\n\n"
        "### Suggestions:\n{suggestion}\n\n### Evaluation:\n"
    ),
    "prompt_suggestion_seperated": (
        "Below are some responses for a given task. The task is defined by the Instruction. "
        "Evaluate the responses and generate a score for each response. "
        "And the suggestion followed by each response is for you to refer.\n\n"
        "### Instruction:\n{question}\n\n### Response 1:\n{response_chatdoctor}\n\n### Suggestion for response 1:\n{suggestion1}\n\n"
        "### Response 2:\n{response_baize}\n\n### Suggestion for response 2:\n{suggestion2}\n\n"
        "### Response 3:\n{response_reference}\n\n### Suggestion for response 3:\n{suggestion3}\n\n"
        "### Evaluation:\n"
    ),
    "prompt_no_input1": (
        "Below are some responses for a given task. The task is defined by the Instruction. "
        "Evaluate the responses with the knowledge you learned from the perspective of the response's relevance to the "
        "task, correctness of the knowledge in response, and understandability of the response's content."
        "Then generate a score for each response.\n\n"
        "### Instruction:\n{question}\n\n### Response 1:\n{response_chatdoctor}\n\n"
        "### Response 2:\n{response_baize}\n\n### Response 3:\n{response_reference}\n\n"
        "### Evaluation:\n"
    ),
    "prompt_no_input2": (
        "Below are some responses for a given task. The task is defined by the Instruction. "
        "Evaluate the doctor's responses to the task with the knowledge you learned from the perspective "
        "of the response's relevance to the "
        "task, correctness of the knowledge in response, and understandability of the response's content. "
        "Then generate a score for each response.\n\n"
        "### Instruction:\n{question}\n\n### Doctor 1:\n{response_chatdoctor}\n\n"
        "### Doctor 2:\n{response_baize}\n\n### Doctor 3:\n{response_reference}\n\n"
        "### Evaluation:\n"
    ),
    "prompt_for_suggestions": (
        "### Instruction:\n{question}\n\n### Response 1:\n{response_chatdoctor}\n\n"
        "### Response 2:\n{response_baize}\n\n### Response 3:\n{response_reference}\n\n"
        "### Evaluation:\n{output}\n\n"
    ),
}
