import os
import pandas as pd
import numpy as np

# Retrieve existing participants function
def get_existing_participants(retelling_folder):
    participants = set()
    
    # Ensures no duplicate participant directories exist or are accounted for
    for participant_dir in os.listdir(retelling_folder):
        participant_path = os.path.join(retelling_folder, participant_dir)
        if os.path.isdir(participant_path) and participant_dir.startswith('p'):
            participants.add(participant_dir)
    return participants

def save_comparisons_to_csv(comparisons, output_csv, desired_order=None):
    # Convert list to DataFrame; this will include the union of all keys.
    df = pd.DataFrame(comparisons)
    
    # If a desired order is provided, filter out columns not in df.
    if desired_order is not None:
        df = df[[col for col in desired_order if col in df.columns]]
    
    # Write to CSV.
    df.to_csv(output_csv, index=False)
    print(f"Comparison results saved to: {output_csv}")

