import json
import pathlib
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_metadata(metadata_path):
    """
    Loads metadata from CasualConversations_clean_transcriptions.json.
    Creates a compound key from the metadata key by removing the extension,
    while preserving the original case.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    file_to_age = {}
    for full_key, entry in metadata.items():
        # Remove the extension (e.g., from "CasualConversationsA/1174/1174_11.MP4" to "CasualConversationsA/1174/1174_11")
        key_path = pathlib.Path(full_key)
        compound_key = str(key_path.with_suffix(''))
        age = entry.get("age", None)
        file_to_age[compound_key] = age
    return file_to_age

def get_valid_audio_files(audio_root, metadata_mapping):
    """
    Walk through audio_root to find .wav files.
    For each audio file, compute a compound key from its relative path
    (using the last three parts, with the extension removed) and preserving case.
    If an exact match is not found, try to find any metadata key ending with that key.
    Only return files for which a valid age is found in the metadata mapping.
    """
    valid_files = []
    root_path = pathlib.Path(audio_root)
    all_files = list(root_path.rglob("*.wav"))
    for audio_file in tqdm(all_files, desc="Collecting audio files", unit="file"):
        try:
            rel = audio_file.relative_to(audio_root)
        except ValueError:
            continue
        # Use the last three parts of the relative path.
        if len(rel.parts) >= 3:
            compound_parts = rel.parts[-3:]
        else:
            compound_parts = rel.parts
        # Remove the extension from the last part.
        compound_parts = list(compound_parts)
        compound_parts[-1] = pathlib.Path(compound_parts[-1]).stem
        file_key = "/".join(compound_parts)
        
        # Try to find an exact match.
        age = metadata_mapping.get(file_key)
        # If not found, try to find a metadata key that ends with our file_key.
        if age is None:
            for meta_key, meta_age in metadata_mapping.items():
                if meta_key.endswith(file_key):
                    age = meta_age
                    break

        if age is None or (isinstance(age, str) and age.upper() == "N/A"):
            print(f"Skipping {audio_file}: no valid age for key '{file_key}'.")
            continue
        valid_files.append((audio_file, age))
    return valid_files

def visualize_unique_age_distance_matrix(ages):
    """
    Extracts unique ages (sorted) and computes a matrix of absolute differences.
    """
    unique_ages = sorted(set(float(a) for a in ages))
    unique_ages_arr = np.array(unique_ages)
    # Compute absolute differences.
    age_distance_matrix = np.abs(np.subtract.outer(unique_ages_arr, unique_ages_arr))
    
    plt.figure(figsize=(8, 6))
    plt.imshow(age_distance_matrix, interpolation='nearest', cmap='viridis')
    plt.title("Unique Age Distance Matrix")
    plt.xlabel("Age")
    plt.ylabel("Age")
    plt.colorbar(label="Age Difference")
    
    tick_positions = list(range(len(unique_ages)))
    plt.xticks(tick_positions, unique_ages, rotation=45, fontsize=10)
    plt.yticks(tick_positions, unique_ages, fontsize=10)
    plt.tight_layout()
    
    save_path = os.path.join("casual_conversations", "matrices", "rdms", "unique_age_matrix.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Unique age distance matrix saved to {save_path}")
    plt.show()

def main():
    metadata_path = "casual_conversations/metadata/CasualConversations_clean_transcriptions.json"
    audio_root = "casual_conversations/rawdata/audios"
    
    # Load metadata mapping using compound keys (preserving case).
    metadata_mapping = load_metadata(metadata_path)
    
    # Get all valid audio files (only those with a valid age)
    valid_audio_files = get_valid_audio_files(audio_root, metadata_mapping)
    if not valid_audio_files:
        print("No valid audio files found!")
        return
    
    # Extract ages from valid files.
    ages = [age for _, age in valid_audio_files]
    
    # For a simpler overview, compute unique ages and visualize their distance matrix.
    visualize_unique_age_distance_matrix(ages)

if __name__ == "__main__":
    main()
