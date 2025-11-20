import sys
import torch
import argparse
from PIL import Image
from pathlib import Path
import os
import json
import re
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

import torch.distributed as dist

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

def is_dist():
    return dist.is_available() and dist.is_initialized()

def init_dist():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank() if is_dist() else 0
    world_size = dist.get_world_size() if is_dist() else 1
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

def barrier():
    if is_dist():
        dist.barrier()

def EvalLingoQA(args):
    rank, world_size, local_rank = init_dist()

    # read evaluation json file
    eval_json = json.load(open(args.eval_file, 'r'))

    # -------- Model & Processor --------
    device = torch.device(f"cuda:{local_rank}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_id)

    # -------- Output path & temp files --------
    os.makedirs(args.output_path, exist_ok=True)
    shard_path = os.path.join(
        args.output_path,
        f'lingoqa_results_{args.model_id.split("/")[-1]}__rank{rank}.json'
    )
    final_path = os.path.join(
        args.output_path,
        f'lingoqa_results_{args.model_id.split("/")[-1]}_{args.num_val_samples}.json'
    )

    result = []

    # -------- Shard samples by rank --------
    max_n = len(eval_json) if args.num_val_samples is None else min(args.num_val_samples, len(eval_json))

    for index, data in enumerate(eval_json[:max_n]):
        if index % world_size != rank:
            continue

        # obtain the inference results
        question = data['conversations'][0]['value'].strip()
        answer_gt = data['conversations'][1]['value'].strip()
        content = []
        if isinstance(data['image'], list):
            image_lists = sorted(data['image'])
            for image in image_lists:
                content.append({'type': 'image', 'image': image})
        else:
            content.append({'type': 'image', 'image': data['image']})
        content.append({"type": "text", "text": question})

        if args.training_type == 'grpo':
            messages = [
                {
                    'role': 'system',
                    'content': [{'type': 'text', 'text':
                        'You are a mature and professional driver. Given a driving related question, please think about the reasoning process in mind and then provide the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>'}]
                },
                {"role": "user", "content": content}
            ]
        else:
            messages = [
                {'role': 'system', 'content': [{'type': 'text', 'text': 'You are a mature and professional driver.'}]},
                {"role": "user", "content": content}
            ]

        # Preparation for inference
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        )
        inputs = inputs.to(device)

        # Inference: Generation of the output
        with torch.inference_mode():
            generated_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
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
            result.append({
                'index': index,
                'token': data['scene_token'],
                'question': question,
                'answer': f'{answer}',
                'answer_gt': f'{answer_gt.strip()}',
                'thinking': f'{thinking.strip()}'
            })
        else:
            result.append({
                'index': index,
                'token': data['scene_token'],
                'question': question,
                'answer': f'{output_text.strip()}',
                'answer_gt': f'{answer_gt.strip()}',
            })

        if rank == 0:
            print(f"[rank {rank}] index: {index}")
            print(f"🚀 Qwen3VL: {output_text.strip()}\n")
            print(f"Ground Truth: {answer_gt.strip()}\n")

    with open(shard_path, 'w', encoding='utf-8') as fshard:
        json.dump(result, fshard, indent=2)

    barrier()

    if rank == 0:
        merged = []
        shard_files = []
        for r in range(world_size):
            rp = os.path.join(
                args.output_path,
                f'lingoqa_results_{args.model_id.split("/")[-1]}__rank{r}.json'
            )
            if os.path.exists(rp):
                shard_files.append(rp)
                with open(rp, 'r', encoding='utf-8') as fr:
                    merged.extend(json.load(fr))
        merged.sort(key=lambda x: x['index'])
        for x in merged:
            x.pop('index', None)

        with open(final_path, 'w', encoding='utf-8') as ffinal:
            json.dump(merged, ffinal, indent=2)

        for sf in shard_files:
            os.remove(sf)
        print(f"[rank 0] ✅ merged results -> {final_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, require=True)
    parser.add_argument("--eval_file", type=str, require=True)
    parser.add_argument("--num_val_samples", type=int, default=6019)
    parser.add_argument("--output_path", type=str, default='traj_eval_results/')
    parser.add_argument("--training_type", type=str, default='sft')
    args = parser.parse_args()
    EvalLingoQA(args)
