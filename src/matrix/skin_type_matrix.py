import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle

def load_clean_transcriptions(clean_transcriptions_path):
    with open(clean_transcriptions_path, "r") as f:
        return json.load(f)

def extract_skin_data(clean_transcriptions):
    """
    Traverse the clean_transcriptions JSON and extract:
      - file_keys (using the video_path's stem)
      - file_paths (the original video_path, or dictionary key if missing)
      - corresponding skin_type values (converted to float)
      - corresponding ages (converted to float)
      
    Handles both "skin_type" and "skin-type" keys.
    """
    file_keys = []
    file_paths = []
    skin_types = []
    ages = []
    
    if isinstance(clean_transcriptions, dict):
        # Iterate over key-value pairs.
        for key, entry in clean_transcriptions.items():
            # Use the entry's "video_path" if available; otherwise, use the key.
            video_path = entry.get("video_path", key)
            if not video_path:
                continue
            # Handle both naming conventions.
            skin = entry.get("skin_type") or entry.get("skin-type")
            if skin is None or skin == "":
                continue
            try:
                skin_val = float(skin)
            except ValueError:
                print(f"Skipping {video_path} because skin_type is not a valid number: {skin}")
                continue
            
            age = entry.get("age")
            if age is None or (isinstance(age, str) and age.upper() == "N/A"):
                continue
            try:
                age_val = float(age)
            except ValueError:
                print(f"Skipping {video_path} because age is not a valid number: {age}")
                continue
            
            file_stem = pathlib.Path(video_path).stem
            file_keys.append(file_stem)
            file_paths.append(video_path)
            skin_types.append(skin_val)
            ages.append(age_val)
    elif isinstance(clean_transcriptions, list):
        for entry in clean_transcriptions:
            video_path = entry.get("video_path")
            if not video_path:
                continue
            skin = entry.get("skin_type") or entry.get("skin-type")
            if skin is None or skin == "":
                continue
            try:
                skin_val = float(skin)
            except ValueError:
                print(f"Skipping {video_path} because skin_type is not a valid number: {skin}")
                continue
            
            age = entry.get("age")
            if age is None or (isinstance(age, str) and age.upper() == "N/A"):
                continue
            try:
                age_val = float(age)
            except ValueError:
                print(f"Skipping {video_path} because age is not a valid number: {age}")
                continue
            
            file_stem = pathlib.Path(video_path).stem
            file_keys.append(file_stem)
            file_paths.append(video_path)
            skin_types.append(skin_val)
            ages.append(age_val)
    return file_keys, file_paths, skin_types, ages

def compute_skin_matrix(skin_types_list):
    """
    Given a list of skin type values, compute an NxN matrix where each entry
    is the absolute difference between two skin type values.
    """
    skin_array = np.array(skin_types_list, dtype=float)
    skin_matrix = np.abs(skin_array[:, None] - skin_array[None, :])
    return skin_matrix

def visualize_skin_matrix(skin_matrix, ages, save_path="skin_matrix.png"):
    """
    Visualize the skin type difference matrix as a heatmap.
    
    Instead of labeling axes with file identifiers, we now label them by age.
    Tick labels (derived from the age values) are sampled to avoid clutter.
    """
    num_labels = len(ages)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(skin_matrix, interpolation="nearest", cmap="viridis")
    plt.title("Skin Type Difference Matrix (Labeled by Age)")
    plt.xlabel("Samples (Age)")
    plt.ylabel("Samples (Age)")
    plt.colorbar(im, label="Absolute Difference in Skin Type")
    
    # Sample tick marks to avoid clutter.
    tick_interval = max(1, num_labels // 10)
    tick_positions = list(range(0, num_labels, tick_interval))
    tick_labels = [str(ages[i]) for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)
    plt.yticks(tick_positions, tick_labels, fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Skin matrix visualization saved to {save_path}")
    plt.show()

def main():
    clean_transcriptions_path = "casual_conversations/metadata/CasualConversations_clean_transcriptions.json"
    output_image_path = "casual_conversations/matrices/rdms/skin_matrix.png"
    output_pkl_path = "casual_conversations/matrices/rdms/skin_matrix.pkl"
    
    clean_transcriptions = load_clean_transcriptions(clean_transcriptions_path)
    file_keys, file_paths, skin_types, ages = extract_skin_data(clean_transcriptions)
    
    if not file_keys or not skin_types or not ages:
        print("No valid skin type data found!")
        return
    
    # Sort all data by age (lowest to highest)
    sorted_indices = sorted(range(len(ages)), key=lambda i: ages[i])
    file_keys = [file_keys[i] for i in sorted_indices]
    file_paths = [file_paths[i] for i in sorted_indices]
    skin_types = [skin_types[i] for i in sorted_indices]
    ages = [ages[i] for i in sorted_indices]
    
    skin_matrix = compute_skin_matrix(skin_types)
    visualize_skin_matrix(skin_matrix, ages, save_path=output_image_path)
    
    # Create a results dictionary.
    results = {
        "file_keys": file_keys,
        "file_paths": file_paths,
        "ages": ages,
        "skin_types": skin_types,
        "skin_matrix": skin_matrix
    }
    
    # Save the results dictionary to a pickle file.
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Skin matrix data saved to {output_pkl_path}")

if __name__ == "__main__":
    main()
