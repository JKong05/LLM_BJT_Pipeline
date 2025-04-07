import os
import json
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pathlib
import matplotlib.gridspec as gridspec

from ..utils.semantic_processing import semantic_seamless_search

def load_clean_transcriptions(clean_transcriptions_path):
    with open(clean_transcriptions_path, "r") as f:
        return json.load(f)

def get_audio_files(audio_root):
    audio_files = []
    for root, dirs, files in os.walk(audio_root):
        for file in files:
            if file.lower().endswith(".wav"):
                audio_files.append(os.path.join(root, file))
    return audio_files

def match_transcription(audio_file, clean_transcriptions):
    """
    Matches the audio file with its transcription and age by checking if the file stem
    is a substring of any key in the transcription JSON.
    """
    file_stem = pathlib.Path(audio_file).stem  # e.g. "0069_05"
    for key, value in clean_transcriptions.items():
        if file_stem in key:
            if isinstance(value, dict):
                transcription = value.get("cleaned_transcription", "")
                age = value.get("age", None)
            else:
                transcription = value
                age = None
            return key, transcription, age
    return None, None, None

def extract_semantic_embeddings_from_audio(audio_root, clean_transcriptions, batch_size=10, device="cpu"):
    audio_files = get_audio_files(audio_root)
    chunk = []
    total = len(audio_files)
    pbar = tqdm(total=total, desc="Processing audio files", unit="file")
    for audio_file in audio_files:
        matched_key, transcription, age = match_transcription(audio_file, clean_transcriptions)

        if transcription is None:
            print(f"No matching transcription found for {audio_file}.")
            pbar.update(1)
            continue
        if age is None or (isinstance(age, str) and age.upper() == "N/A"):
            print(f"N/A age in {audio_file}.")
            pbar.update(1)
            continue
                
        try:
            emb = semantic_seamless_search(transcription)
            emb = emb.to(device, non_blocking=True)

            if emb.dim() == 2 and emb.shape[0] == 1:
                emb = emb.squeeze(0)

            # Append tuple including the age.
            chunk.append((audio_file, emb, age))

            if len(chunk) >= batch_size:
                yield chunk
                chunk = []
                if device == "cuda":
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
        pbar.update(1)
    pbar.close()
    
    if chunk:
        yield chunk
        if device == "cuda":
            torch.cuda.empty_cache()

def compute_age_matrix(ages_list):
    """
    Given a list of ages, return an NxN matrix of absolute age differences.
    """
    ages_array = np.array([float(a) for a in ages_list], dtype=float)
    age_matrix = np.abs(ages_array[:, None] - ages_array[None, :])
    return age_matrix

def visualize_combined_rdms(embeddings, ages, file_stems, 
                            semantic_title="Semantic RDM", age_title="Age Matrix RDM",
                            save_path="casual_conversations/matrices/rdms/combined_rdm.png"):
    """
    Compute and visualize side-by-side:
    - A semantic RDM (pairwise dissimilarity of embeddings) without tick marks.
    - An Age RDM (pairwise absolute differences between ages) with sampled tick labels.
    
    Tick labels on the age RDM are sampled (at most one every 10 samples) for clarity.
    """
    # Compute semantic RDM.
    semantic_rdm = torch.cdist(embeddings, embeddings)
    semantic_rdm_np = semantic_rdm.cpu().numpy()
    np.fill_diagonal(semantic_rdm_np, np.nan)
    
    # Compute age matrix RDM.
    age_matrix = compute_age_matrix(ages)
    
    # Create figure with two subplots.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left subplot: Semantic RDM without tick marks.
    im1 = ax1.imshow(semantic_rdm_np, interpolation='nearest', cmap='viridis')
    ax1.set_title(semantic_title)
    ax1.set_xlabel("Samples")
    ax1.set_ylabel("Samples")
    ax1.xaxis.tick_top()
    # Remove tick marks on semantic RDM.
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Dissimilarity", rotation=270, labelpad=15)
    
    # Right subplot: Age Matrix RDM with sampled tick labels.
    im2 = ax2.imshow(age_matrix, interpolation='nearest', cmap='viridis')
    ax2.set_title(age_title)
    ax2.set_xlabel("Samples")
    ax2.set_ylabel("Samples")
    ax2.xaxis.tick_top()
    
    num_labels = len(file_stems)
    tick_interval = max(1, num_labels // 10)
    tick_positions = list(range(0, num_labels, tick_interval))
    tick_labels = [file_stems[i] for i in tick_positions]
    
    ax2.set_xticks(tick_positions)
    ax2.set_yticks(tick_positions)
    ax2.set_xticklabels(tick_labels, rotation=45, fontsize=8)
    ax2.set_yticklabels(tick_labels, fontsize=8)
    
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Absolute Age Difference", rotation=270, labelpad=15)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Combined RDM visualization saved to {save_path}")
    plt.show()

def main():
    clean_transcriptions_path = "casual_conversations/metadata/CasualConversations_clean_transcriptions.json"
    audio_root = "casual_conversations/rawdata/audios"
    batch_size = 10
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    clean_transcriptions = load_clean_transcriptions(clean_transcriptions_path)
    
    audio_files_list = []
    ages_list = []
    emb_list = []
    
    for batch in extract_semantic_embeddings_from_audio(audio_root, clean_transcriptions, batch_size, device):
        file_names, embeddings, batch_ages = zip(*batch)
        batch_tensor = torch.stack(embeddings, dim=0)
        if device == "cuda":
            batch_tensor = batch_tensor.cpu()
        print("Processed a batch with semantic embeddings tensor shape:", batch_tensor.shape)
        audio_files_list.extend(file_names)
        ages_list.extend(batch_ages)
        emb_list.append(batch_tensor)
    
    if emb_list:
        all_embeddings = torch.cat(emb_list, dim=0)
        print("Total semantic embeddings tensor shape:", all_embeddings.shape)
        
        # Save the semantic results.
        results = {"audio_files": audio_files_list, "ages": ages_list, "embeddings": all_embeddings}
        save_dir = os.path.join("casual_conversations", "matrices")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "age_semantic_matrix.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved semantic results to {save_path}")
        
        # Create file stems for labeling.
        file_stems = [pathlib.Path(p).stem for p in audio_files_list]
        
        # Visualize the combined RDMS (Semantic and Age).
        visualize_combined_rdms(all_embeddings, ages_list, file_stems,
                                save_path=os.path.join("casual_conversations", "matrices", "rdms", "semantic_age_rdm.png"))

if __name__ == "__main__":
    main()
