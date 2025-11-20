import os
import pandas as pd
import json
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, required=True)

    return parser

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    os.makedirs(args.output_folder_path, exist_ok=True)
    for file_name in os.listdir(args.input_folder_path):
        if file_name.endswith('.json'):
            json_file_path = os.path.join(args.input_folder_path, file_name)
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            df = pd.DataFrame(data)
            csv_file_name = file_name.replace('.json', '.csv')
            df.to_csv(csv_file_name, index=False)
            print(f"Converted {json_file_path} to {csv_file_name}")

    print("All JSON files have been converted to CSV files and saved in the new folder.")
