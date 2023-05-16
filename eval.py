
import os
import random
import numpy as np
import torch
import datetime
import evaluate

from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.llama_utils import LlamaUtils

from transformers.utils import logging
from datasets import DatasetDict, Dataset, load_from_disk, load_metric
from typing import Optional, Dict, List
from loguru import logger


import argparse

format_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


# piqa eval
def eval_answer(output_filepath, eval_filepath, dev_textpath, dataset_name: str = 'piqa'):
    with open(output_filepath, encoding="utf-8") as output_file:
        outputs_json = output_file.read()
    with open(eval_filepath, encoding="utf-8") as eval_file:
        standard = eval_file.read().splitlines()
    with open(dev_textpath, encoding="utf-8") as dev_text:
        text = dev_text.read().splitlines()
    
    if dataset_name == 'piqa':
        output_dict = json.loads(outputs_json)
        for idx, _dict in enumerate(output_dict):
            output = _dict['answer']
            if output == '0' or output == '1':
                continue
            elif ("sol1" in output) or ("sol 1" in output) or ("Option 1" in output):
                output == '0'
            elif ("sol2" in output) or ("sol 2" in output) or ("Option 2" in output):
                output == '1'
            else:
                data = json.loads(text[idx])
                sol1 = data["sol1"]
                sol2 = data["sol2"]
                if (data["sol1"][:64] in output) or (output in sol1):
                    output == '0'
                elif (data["sol2"][:64] in output) or (output in sol2):
                    output == '1'
                else:
                    output == '0'
    elif dataset_name == 'hellaswag':
        for idx, output in enumerate(outputs):
            if len(output) == 1:
                continue
            elif output == "0":
                output == '0'
            elif output == "1":
                output == '1'
            elif output == "2":
                output == '2'
            elif output == "3":
                output == '3'
            else:
                data = json.loads(text[idx])
                sol1 = data["activity_label"] + data["ending_options"][0]
                sol2 = data["activity_label"] + data["ending_options"][1]
                sol3 = data["activity_label"] + data["ending_options"][2]
                sol4 = data["activity_label"] + data["ending_options"][3]
                if (data["ending_options"][0] in output) or (output in sol1):
                    output == '0'
                elif (data["ending_options"][1] in output) or (output in sol2):
                    output == '1'
                elif (data["ending_options"][2] in output) or (output in sol3):
                    output == '2'
                elif (data["ending_options"][3] in output) or (output in sol3):
                    output == '3'
                else:
                    output == '0'
    elif dataset_name == 'winogrande':
        for idx, output in enumerate(outputs):
            if output == '2' or output == '1':
                continue
            elif "option1" in output:
                output == '1'
            elif "option2" in output:
                output == '2'
            else:
                data = json.loads(text[idx])
                if data["option1"] in output:
                    output == '1'
                elif data["option2"][:64] in output:
                    output == '2'
                else:
                    output == '1'
    # print(standard[:10],outputs)
    print("the rating in the {} is {}".format(dataset_name, accuracy_score(standard, outputs)))

if __name__=='__main__':
    
    
    output_filepath = 'answer.jsonl'
    eval_filepath = './metric/piqa/label/valid-labels.lst'
    dev_textpath = './metric/piqa/inputs/valid.jsonl'
    dataset_name = "piqa"
    
    # output_filepath = 'answer.txt'
    # eval_filepath = './metric/hellaswag-train-dev/valid-labels.lst'
    # dev_textpath = './metric/hellaswag-train-dev/valid.jsonl'
    # dataset_name = "hellaswag"
    
    # output_filepath = 'answer.txt'
    # eval_filepath = './metric/winogrande_1.1/dev-labels.lst'
    # dev_textpath = './metric/winogrande_1.1/dev.jsonl'
    # dataset_name = "winogrande"
    
    eval_answer(output_filepath, eval_filepath, dev_textpath, dataset_name)
    

if __name__ == '__main__':
    run()
