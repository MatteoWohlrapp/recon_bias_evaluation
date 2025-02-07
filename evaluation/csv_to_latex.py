import argparse
import yaml
import pandas as pd
import os

def main():
    # important that naming of columns is in the same order as in the latex file
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to options YAML file.')
    args = parser.parse_args()
    
    # Load configuration
    with open(args.opt, 'r') as f:
        config = yaml.safe_load(f)

    latex_path = config['latex_path']
    csv_path = config['csv_path']
    results_path = config['results_path']
    df = pd.read_csv(csv_path)

    # get name of latex file
    latex_name = latex_path.split("/")[-1]

    with open(latex_path, 'r') as f:
        latex_content = f.read()

    # get all available columns
    available_columns = df.columns.tolist()
    for row in df.iterrows():
        replacement_key = ""
        for column in available_columns:
            if column != "value":
                #check if last column
                replacement_key += f"{row[1][column]}-"

        replacement_key = replacement_key[:-1] # remove last dash
        # round to 2 decimal places
        latex_content = latex_content.replace(replacement_key, f"{row[1]['value']:.2f}")

    with open(os.path.join(results_path, latex_name), 'w') as f:
        f.write(latex_content)


if __name__ == '__main__':
    main()