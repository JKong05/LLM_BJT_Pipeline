import json
import pathlib
import torch
import pickle
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from ..utils.prosody_processing import get_prosodic_embeddings

def load_metadata(metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    file_to_age = {}
    for entry in metadata.values():
        age = entry.get("label", {}).get("age", None)
        for file_path in entry.get("files", []):
            base_stem = pathlib.Path(file_path).stem
            file_to_age[base_stem] = age
    return file_to_age

def extract_expressive_embeddings(root_dir, metadata_mapping, batch_size=10, device="cpu"):
    root_path = pathlib.Path(root_dir)
    chunk = []
    all_files = list(root_path.rglob("*.wav"))
    
    with tqdm(total=len(all_files), desc="Processing audio files", unit="file") as pbar:
        for audio_file in all_files:
            # Look up the age using the file stem
            file_stem = audio_file.stem
            age = metadata_mapping.get(file_stem, None)
            # Skip files with missing age or age equals "N/A"
            if age is None or age.upper() == "N/A":
                print(f"N/A age in {audio_file}.")
                pbar.update(1)
                continue
                
            try:
                with torch.no_grad():
                    emb = get_prosodic_embeddings(str(audio_file))
                    emb = emb.to(device, non_blocking=True)
                    
                    if emb.dim() == 2 and emb.shape[0] == 1:
                        emb = emb.squeeze(0)
                    
                    chunk.append((audio_file.name, emb, age))
                    
                    if len(chunk) >= batch_size:
                        yield chunk
                        chunk = []
                        if device == "cuda":
                            torch.cuda.empty_cache()
                pbar.update(1)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                pbar.update(1)
    if chunk:
        yield chunk

def visualize_expressive_rdm(embeddings, age_labels):
    # Compute pairwise distances between individual embeddings
    rdm = torch.cdist(embeddings, embeddings)
    rdm_np = rdm.cpu().numpy()

    np.fill_diagonal(rdm_np, np.nan)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(rdm_np, interpolation='nearest', cmap='viridis')
    plt.title("RDM: Expressivity Based on Age")
    plt.ylabel("Age")
    plt.colorbar(label="Dissimilarity")
    
    ax = plt.gca()
    ax.xaxis.tick_top()                   # Move tick labels to top
    ax.xaxis.set_label_position('top')    # Move x-axis label to top
    plt.xlabel("Age")
    
    # Create tick positions: only label the first embedding of each contiguous age group.
    group_ticks = []
    group_labels = []
    prev_age = None
    for i, age in enumerate(age_labels):
        if age != prev_age:
            group_ticks.append(i)
            group_labels.append(age)
            prev_age = age
    
    # Set ticks only at the computed positions.
    plt.xticks(group_ticks, group_labels, rotation=45)
    plt.yticks(group_ticks, group_labels)
    plt.tight_layout()
    
    save_path = "casual_conversations/matrices/rdms/expressive_rdm.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Individual RDM visualization saved to {save_path}")
    plt.show()

def main():
    root_dir = "casual_conversations/rawdata/audios"
    metadata_path = "casual_conversations/metadata/CasualConversations_mini.json"
    batch_size = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load metadata mapping (file stem -> age)
    metadata_mapping = load_metadata(metadata_path)
    
    # Collect data: audio file names, ages, and individual embeddings.
    audio_files_list = []
    ages_list = []
    emb_list = []
    
    for batch in extract_expressive_embeddings(root_dir, metadata_mapping, batch_size, device):
        file_names, embeddings, ages = zip(*batch)
        batch_tensor = torch.stack(embeddings, dim=0)
        if device == "cuda":
            batch_tensor = batch_tensor.cpu()
        print("Processed a batch with expressive embeddings tensor shape:", batch_tensor.shape)
        audio_files_list.extend(file_names)
        ages_list.extend(ages)
        emb_list.append(batch_tensor)
    
    if emb_list:
        # Concatenate all individual embeddings into one tensor
        all_embeddings = torch.cat(emb_list, dim=0)
        print("Total embeddings tensor shape:", all_embeddings.shape)
        
        # Save full results (optional)
        results = {"audio_files": audio_files_list, "ages": ages_list, "embeddings": all_embeddings}
        save_dir = os.path.join("casual_conversations", "matrices")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "age_expressive_matrix.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved results to {save_path}")
        
        # Compute and visualize the individual-level RDM with age labels
        visualize_expressive_rdm(all_embeddings, ages_list)

if __name__ == "__main__":
    main()
