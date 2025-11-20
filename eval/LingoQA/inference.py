import sys
import torch
import argparse
from pathlib import Path
import os
import pandas as pd
from tqdm import tqdm
import json
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from datasets import Dataset
from functools import partial 
from benchmark.constants import Keys
from benchmark.judge import LingoJudge
import evaluate
from evaluate import load
from pycocoevalcap.meteor.meteor import Meteor

CIDEr = load("Kamichanw/CIDEr")
bleu = evaluate.load("bleu")
sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

def evaluate_metric(predictions, lingojudge_pretrained, lingo_test_parquet, batch_size=1):
    """
    Simple script for running evaluation on the LingoQA benchmark.

    Args:
        predictions_path: Path to a .csv file containing the model predictions.
        batch_size: Batch size for evaluation.
    """
    # Load references
    references = pd.read_parquet(lingo_test_parquet)
    references = references[[Keys.question_id, Keys.question, Keys.answer]]
    references = references.groupby([Keys.question_id, Keys.question]).agg({Keys.answer: list}).reset_index()
    references = references.rename({Keys.answer: Keys.references}, axis=1)
    # Load predictions
    predictions = predictions.rename({"answer": Keys.prediction}, axis=1)
    # Merge predictions and references
    merged = pd.merge(predictions, references, on=[Keys.question_id])
    if len(merged) != 500:
        print("WARNING! You are evaluating on a subset of the LingoQA benchmark. Please check your input file for missing or mis-matched examples.")

    # # Create dataset from merged data
    dataset = Dataset.from_pandas(merged)
    # Lingo-Judge evaluation
    judge = LingoJudge(lingojudge_pretrained).eval().cuda()
    dataset_evaluated = dataset.map(partial(evaluate_question, judge), batched=True, batch_size=batch_size)
    dataset_filtered = dataset_evaluated.filter(select_correct)
    benchmark_score = dataset_filtered.num_rows/dataset_evaluated.num_rows
    print(f"The overall benchmark score is {benchmark_score*100}%")
    # obtain other eval metrics
    predictions = [pred.strip() for pred in dataset['Keys.prediction']]
    references = [[ref.strip() for ref in refs] for refs in dataset['Keys.references']]
    assert len(predictions) == len(references), "Predictions and references must have the same length."
    pred_dict = {idx: [pred] for idx, pred in enumerate(predictions)}
    ref_dict = {idx: refs for idx, refs in enumerate(references)}
    score = CIDEr.compute(predictions=dataset['Keys.prediction'], references=dataset['Keys.references'])
    print('CIDEr score',score['CIDEr'])
    results = bleu.compute(predictions=dataset['Keys.prediction'], references=dataset['Keys.references'])
    print("BLEU score:", results)
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(ref_dict, pred_dict)
    print('meteor_score',meteor_score)
    return {'LingoJudge': benchmark_score, 'CIDEr': score['CIDEr'], 'BLEU': results, 'METEOR': meteor_score}

def evaluate_question(metric: LingoJudge, data_dict: dict) -> dict:
    """
    Run evaluation for a batch of questions.

    Args:
        metric: the evaluation metric for computing the scores.
        data_dict: the data dictionary containing questions, references, and predictions.

    Out:
        data_dict: updated data dictionary containing information such as
        the maximum score, the probability of correctness, and a boolean
        indicating whether the prediction is correct or not.
    """
    if 'question' in data_dict:
        questions = data_dict['question']
    else:
        questions = data_dict['question_x']         # a list of questions │['Are you legally allowed to make a right turn at this intersection?']
    references = data_dict['Keys.references']
    prediction = data_dict['Keys.prediction']

    scores = metric.compute(questions, references, prediction)
    data_dict[Keys.score] = scores
    data_dict[Keys.probability] = torch.sigmoid(scores)
    data_dict[Keys.correct] = scores > 0.0
    data_dict[Keys.incorrect] = scores <= 0.0
    return data_dict

def select_correct(data_dict: dict) -> bool:
    """
    Filtering function for selecting the predictions classified as correct.
    """
    return data_dict[Keys.correct]

def select_incorrect(data_dict: dict) -> bool:
    """
    Filtering function for selecting the predictions classified as incorrect.
    """
    return data_dict[Keys.incorrect]

def EvalLingoQA(args):
    # read evaluation json file
    eval_file = os.path.join(args.LINGOQA_DATA_PATH, 'evaluation_data.json')
    eval_json = json.load(open(eval_file, 'r'))
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model_id)

    result = []
    if args.output_path is not None:
        os.makedirs(args.output_path, exist_ok=True)
        output_file = os.path.join(args.output_path, f'lingoqa_results_{args.model_id.split("/")[-1]}.json')
        f = open(output_file, 'a', encoding='utf-8')
    else:
        f = open('results.json', 'a', encoding='utf-8')
    
    for index, data in tqdm(enumerate(eval_json)):
        # obtain the inference results
        segment_id = data['image_id']
        question_id = data['question_id']
        question = data['conversations'][0]['value'].strip()
        answer_gt = data['conversations'][1]['value'].strip()
        content = []
        messages = []
        if isinstance(data['image'], list):
            image_lists = sorted(data['image'])
            for image in image_lists:
                content.append(
                    {
                        'type': 'image',
                        'image': image,
                    }
                )
        else:
            content.append(
                    {
                        'type': 'image',
                        'image': image,
                    }
                )
        content.append({"type": "text", "text": question})
        if args.training_type == 'grpo':
            # add system prompt
            messages = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': 'You are a mature and professional driver. Given a driving related question, please think about the reasoning process in mind and then provide the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'}]
                },
                {
                    "role": "user",
                    "content": content,
                }
            ]
        elif args.training_type == 'sft':
            messages = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text': 'You are a mature and professional driver.'}]
                },
                {
                    "role": "user",
                    "content": content,
                }
            ]
        else:
            # default system for vanilla Qwen3-VL
            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference: Generation of the output
        generated_ids = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        if '<answer>' in output_text:
            pattern = r'<answer>(.*?)</answer>'
            match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
            answer = match.group(1).strip() if match else output_text.strip()
            pattern = r'<think>(.*?)</think>'
            match = re.search(pattern, output_text, re.DOTALL | re.IGNORECASE)
            thinking = match.group(1).strip() if match else output_text.strip()
            result.append(
                {
                    'question_id': question_id,
                    'question': question,
                    'segment_id': segment_id,
                    'answer': f'{answer}',
                    'answer_gt': f'{answer_gt.strip()}',
                    'thinking': f'{thinking.strip()}'
                }
            )
        else:
            result.append(
                {
                    'question_id': question_id,
                    'question': question,
                    'segment_id': segment_id,
                    'answer': f'{output_text.strip()}',
                    'answer_gt': f'{answer_gt.strip()}',
                }
            )
        break
    df = pd.DataFrame(result)
    # compute the evaluation metrics
    LingoQA_parquet = os.path.join(args.LINGOQA_DATA_PATH, 'evaluation/val.parquet')
    eval_results = evaluate_metric(df, args.lingojudge_pretrained, LingoQA_parquet)
    csv_file_name = output_file.replace('.json', '.csv')
    result.append({'LingoJudge': eval_results['LingoJudge']})
    result.append({'CIDEr': eval_results['CIDEr']})
    result.append({'BLEU': eval_results['BLEU']})
    result.append({'METEOR': eval_results['METEOR']})
    json_data = json.dumps(result, indent=4)
    f.write(json_data)
    f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="qwen3-vl-8b")
    parser.add_argument("--eval_file", type=str, default='./data/LingoQA/evaluation_data.json')
    parser.add_argument("--output_path", type=str, default='./')
    parser.add_argument("--training_type", type=str, default='grpo')
    parser.add_argument("--lingojudge_pretrained", type=str)
    parser.add_argument("--batch_size", type=int, default=1)    
    parser.add_argument('--LINGOQA_DATA_PATH', type=str, help='Path to the LingoQA data directory containing test parquet file.')
    args = parser.parse_args()
    EvalLingoQA(args)