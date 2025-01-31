import subprocess
import pandas as pd
import os

"""
Deprecated
This script runs the AutoPCP model to calculate the PCP score for the input audio files.
10.31.2024 - run_auto_pcp() is not needed for the pipeline, but it is kept for future use.
"""

def run_auto_pcp():
    command = [
        "python", "-m", "stopes.modules",
        "+compare_audios=AutoPCP_multilingual_v2",
        "launcher.cluster=local",
        "+compare_audios.input_file=/home/wallacelab/LLM_BJT/pipeline/input.tsv",
        "compare_audios.src_audio_column=src_audio",
        "compare_audios.tgt_audio_column=tgt_audio",
        "+compare_audios.named_columns=true",
        "+compare_audios.output_file=/home/wallacelab/LLM_BJT/pipeline/output.txt"
    ]

    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print("Output:", result.stdout)
        print("Error (if any):", result.stderr)
    except subprocess.CalledProcessError as e:
        print("An error occurred:", e)

def process_output(output_path, input_path, output_csv):
    with open(output_path, 'r') as f:
        scores = f.read().strip().splitlines()
    
    scores = [float(score.strip()) for score in scores]
    df = pd.read_csv(input_path, sep="\t")

    df['PCP Score'] = scores
    df['Original'] = df['src_audio'].apply(lambda x: x.split('/')[-1]) 
    df['Retelling'] = df['tgt_audio'].apply(lambda x: x.split('/')[-1])

    df = df[["Original", "Retelling", "PCP Score", "src_audio", "tgt_audio"]]

    df.to_csv(output_csv, sep="\t",index=False)

    if os.path.exists(output_path):
        os.remove(output_path)