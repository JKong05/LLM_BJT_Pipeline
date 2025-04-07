import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import pickle

def load_clean_transcriptions(clean_transcriptions_path):
    with open(clean_transcriptions_path, "r") as f:
        return json.load(f)

def extract_gender_data(clean_transcriptions):
    """
    Traverse the clean_transcriptions JSON and extract:
      - file_keys (using the video_path's stem)
      - file_paths (the original video_path, or dictionary key if missing)
      - corresponding gender values (as strings, skipping "N/A")
      - corresponding ages (converted to float)
      
    Handles both dictionary and list formats.
    """
    file_keys = []
    file_paths = []
    genders = []
    ages = []
    
    if isinstance(clean_transcriptions, dict):
        for key, entry in clean_transcriptions.items():
            # Use entry's "video_path" if available; otherwise, use the key.
            video_path = entry.get("video_path", key)
            if not video_path:
                continue
            
            # Extract gender; skip if missing or "N/A".
            gender = entry.get("gender")
            if not gender or gender.strip().upper() == "N/A":
                continue
            gender = gender.strip()
            
            # Extract age.
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
            genders.append(gender)
            ages.append(age_val)
    elif isinstance(clean_transcriptions, list):
        for entry in clean_transcriptions:
            video_path = entry.get("video_path")
            if not video_path:
                continue
            
            gender = entry.get("gender")
            if not gender or gender.strip().upper() == "N/A":
                continue
            gender = gender.strip()
            
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
            genders.append(gender)
            ages.append(age_val)
    return file_keys, file_paths, genders, ages

def compute_gender_matrix(gender_list):
    """
    Given a list of gender values (strings), compute an NxN matrix where each entry is:
      0 if the two genders are the same, and 1 if they are different.
    """
    genders_arr = np.array(gender_list)
    # Create an outer comparison matrix.
    equal_matrix = np.equal.outer(genders_arr, genders_arr)
    # Convert boolean (True if equal) to int (0 if equal, 1 if different).
    gender_matrix = np.where(equal_matrix, 0, 1)
    return gender_matrix

def visualize_gender_matrix(gender_matrix, ages, save_path="gender_matrix.png"):
    """
    Visualize the gender difference matrix as a heatmap.
    The x- and y-axes are labeled with ages (sorted from lowest to highest).
    Tick labels (derived from the age values) are sampled to avoid clutter.
    """
    num_labels = len(ages)
    plt.figure(figsize=(12, 10))
    im = plt.imshow(gender_matrix, interpolation="nearest", cmap="coolwarm")
    plt.title("Gender Difference Matrix (Labeled by Age)")
    plt.xlabel("Samples (Age)")
    plt.ylabel("Samples (Age)")
    plt.colorbar(im, label="Gender Difference (0=same, 1=different)")
    
    tick_interval = max(1, num_labels // 10)
    tick_positions = list(range(0, num_labels, tick_interval))
    tick_labels = [str(ages[i]) for i in tick_positions]
    plt.xticks(tick_positions, tick_labels, rotation=45, fontsize=8)
    plt.yticks(tick_positions, tick_labels, fontsize=8)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Gender matrix visualization saved to {save_path}")
    plt.show()

def main():
    clean_transcriptions_path = "casual_conversations/metadata/CasualConversations_clean_transcriptions.json"
    output_image_path = "casual_conversations/matrices/rdms/gender_matrix.png"
    output_pkl_path = "casual_conversations/matrices/gender_matrix.pkl"
    
    clean_transcriptions = load_clean_transcriptions(clean_transcriptions_path)
    file_keys, file_paths, genders, ages = extract_gender_data(clean_transcriptions)
    
    if not file_keys or not genders or not ages:
        print("No valid gender data found!")
        return
    
    # Sort the data by age (lowest to highest)
    sorted_indices = sorted(range(len(ages)), key=lambda i: ages[i])
    file_keys = [file_keys[i] for i in sorted_indices]
    file_paths = [file_paths[i] for i in sorted_indices]
    genders = [genders[i] for i in sorted_indices]
    ages = [ages[i] for i in sorted_indices]
    
    gender_matrix = compute_gender_matrix(genders)
    visualize_gender_matrix(gender_matrix, ages, save_path=output_image_path)
    
    # Save results to a pickle file.
    results = {
        "file_keys": file_keys,
        "file_paths": file_paths,
        "ages": ages,
        "genders": genders,
        "gender_matrix": gender_matrix
    }
    os.makedirs(os.path.dirname(output_pkl_path), exist_ok=True)
    with open(output_pkl_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Gender matrix data saved to {output_pkl_path}")

if __name__ == "__main__":
    main()
