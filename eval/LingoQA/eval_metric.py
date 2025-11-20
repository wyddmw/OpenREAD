import click
import torch
import pandas as pd
from datasets import Dataset
from functools import partial 
from pathlib import Path
from benchmark.constants import Keys
from benchmark.judge import LingoJudge

import evaluate
from evaluate import load
from pycocoevalcap.meteor.meteor import Meteor
CIDEr = load("Kamichanw/CIDEr")
bleu = evaluate.load("bleu")


def evaluate(predictions, lingojudge_pretrained, lingo_test_parquet, batch_size=1):
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
    dataset_incorrect = dataset_evaluated.filter(select_incorrect)
    incorrect_examples = dataset_incorrect.to_pandas()
    incorrect_examples.to_csv("lingoqa_incorrect_examples.csv", index=False)
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


