import base64
import os
import argparse
import json
from openai import OpenAI
from tqdm import tqdm

 
def main():
    parser = argparse.ArgumentParser(description="Batch run Qwen VL model on images in a directory.")
    parser.add_argument(
        "--model",
        default="gpt-4.1",
        help="Model name (default: qwen2.5-vl-72b-instruct)",
    )
    parser.add_argument(
        "--data_json_file", default="/data2/WiseAD_v2_qwen/cot_data/sampled_40k_no_overlap.json", help="Path to the data JSON file (default: data.json)"
    )
    parser.add_argument(
        "--output_file", default="gpt_output.json", help="Output file name (default: gpt_output.json)"
    )
    parser.add_argument(
        "--sample_size", default=0, type=int, help="Sample numbers (default: 1024)"
    )
    parser.add_argument(
        "--start_index", default=0, type=int, help="Start index (default: 0)"
    )
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    # load json
    data_json_files = json.load(open(args.data_json_file, "r"))
    if args.sample_size > 0:
        if args.start_index > 0:
            data_json_files = data_json_files[args.start_index: args.start_index + args.sample_size]
        else:  
            data_json_files = data_json_files[:args.sample_size]
    # data_json_files = data_json_files[:2]  # for testing
    for item in tqdm(data_json_files):
        image_list = item.get("images", [])
        question = prompt = item['conversations'][0]['value'].strip()
        # answer_gt = item['answer_gt'].strip()
        answer_gt = item['conversations'][1]['value'].strip()
        
        # update prompt for the rationale
        task_prompt = f"""Suppose you are the driver behind the wheel. Here is what you see along with a question and its reference answer. Please analyse images, question and answer first.
                          The question is {question} and the corresponding answer is {answer_gt}.
                          Please provide a rationale of how you think and reason to obtain the answer from a driver's perspective.
                          If the question involves scenery understanding, give a concise and natural rationale in one sentence.
                          For action related questions, give a concise and natural rationale in no more than two sentences.
                          Avoid any unnecessary descriptions like 'From a driver's perspective', 'As a driver', etc.
                          Avoid any redundant conclusions like 'confirming the answer', 'the answer is true', etc."""
        
        # Prepare base64 encoded images
        b64_img_list = []
        for img in image_list:
            with open(img, "rb") as image_file:
                b64_image = base64.b64encode(image_file.read()).decode("utf-8")
                b64_img_list.append(b64_image)
        
        # Prepare content for API call
        content = []
        for b64_image in b64_img_list:
            content.append({"type": "input_image", "image_url": f"data:image/png;base64,{b64_image}"})
        content.append({"type": "input_text", "text": task_prompt})
        
        response = client.responses.create(
                    model=args.model,
                    input=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                )
        rationale = response.output_text.strip()
        print(rationale)
        item['conversations'][1]['value'] = f'<think> {rationale} </think>\n<answer> {answer_gt} </answer>'

    # replace the original json file
    json.dump(data_json_files, open(args.output_file, "w"), indent=4)

if __name__ == "__main__":
    main()

