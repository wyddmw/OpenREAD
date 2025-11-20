import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
import os
import json
import re

# set gpt for scoring
from openai import OpenAI
API_KEY=os.getenv("OPENAI_API_KEY")

client = OpenAI(
        api_key=API_KEY,
    )

def eval_gpt_score(args):
    eval_json = json.load(open(args.eval_file, 'r'))
    corr_num = 0
    for idx, eval_sample in enumerate(eval_json):
        question = eval_sample['question']
        answer_gt = eval_sample['answer_gt']
        answer = eval_sample['answer']
        content = []
        task_prompt = f"Given the question: '{question}', the reference answer: '{answer_gt.strip()}', and the predicted answer {answer}. Please evaluate the correctness of the predicted answer and assign a continuous score between 0 and 1. Please give me the score only."
        content.append({"type": "input_text", "text": task_prompt})

        response = client.responses.create(
                    model='gpt-4.1',
                    input=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                )
        rationale = response.output_text.strip()
        rationale = float(rationale)
        if rationale >= 0.5:
            corr_num += 1
        print(rationale)
    avg_score = corr_num / len(eval_json)
    eval_json.append({'gpt_score': avg_score})
    os.makedirs(args.output_path, exist_ok=True)
    with open(args.eval_file, 'w') as f:
        json.dump(eval_json, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_file", type=str, default='/home/spyder/project/e2e_driving/SmartAD/test.json')
    parser.add_argument("--output_path", type=str, default='cvpr_results/')
    parser.add_argument("--training_type", type=str, default='grpo')
    args = parser.parse_args()
    eval_gpt_score(args)
