import os
import json
import torch
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pathlib

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
    audio_stem = pathlib.Path(audio_file).stem.lower()
    for key, value in clean_transcriptions.items():
        key_stem = pathlib.Path(key).stem.lower()
        if audio_stem == key_stem:
            # If the transcription is stored as a dict, extract the "cleaned_transcription" field.
            if isinstance(value, dict):
                transcription = value.get("cleaned_transcription", "")
            else:
                transcription = value
            return key, transcription
    return None, None

def extract_semantic_embeddings_from_audio(audio_root, clean_transcriptions, batch_size=10, device="cpu"):
    audio_files = get_audio_files(audio_root)
    chunk = []
    total = len(audio_files)

    pbar = tqdm(total=total, desc="Processing audio files", unit="file")
    for audio_file in audio_files:
        
        file_stem = audio_file.stem
        age = clean_transcriptions.get(file_stem, None)

        matched_key, transcription = match_transcription(audio_file, clean_transcriptions)

        if transcription is None:
            print(f"No matching transcription found for {audio_file}.")
            pbar.update(1)
            continue
        if age is None or age.upper() == "N/A":
            print(f"N/A age in {audio_file}.")
            pbar.update(1)
            continue
                
        try:
            emb = semantic_seamless_search(transcription)
            emb = emb.to(device, non_blocking=True)

            if emb.dim() == 2 and emb.shape[0] == 1:
                emb = emb.squeeze(0)

            chunk.append((audio_file, emb))

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

def visualize_semantic_rdm(embeddings, labels):
    rdm = torch.cdist(embeddings, embeddings)
    rdm_np = rdm.cpu().numpy()
    np.fill_diagonal(rdm_np, np.nan)

    plt.figure(figsize=(10, 8))
    plt.imshow(rdm_np, interpolation='nearest', cmap='viridis')
    plt.title("RDM: Semantics Based on Age")
    plt.ylabel("Label")
    plt.colorbar(label="Dissimilarity")

    ax = plt.gca()
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')
    plt.xlabel("Label")

    group_ticks = []
    group_labels = []
    prev = None
    for i, label in enumerate(labels):
        if label != prev:
            group_ticks.append(i)
            group_labels.append(label)
            prev = label
        
    plt.xticks(group_ticks, group_labels, rotation=45)
    plt.yticks(group_ticks, group_labels)
    plt.tight_layout()

    save_path = "casual_conversations/matrices/rdms/semantic_rdm_from_audio.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Semantic RDM visualization saved to {save_path}")
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
        file_names, embeddings, ages = zip(*batch)
        batch_tensor = torch.stack(embeddings, dim=0)
        if device == "cuda":
            batch_tensor = batch_tensor.cpu()
        print("Processed a batch with semantic embeddings tensor shape:", batch_tensor.shape)
        audio_files_list.extend(file_names)
        ages_list.extend(ages)
        emb_list.append(batch_tensor)
    
    if emb_list:
        all_embeddings = torch.cat(emb_list, dim=0)
        print("Total semantic embeddings tensor shape:", all_embeddings.shape)
        
        results = {"audio_paths": audio_paths_list, "ages": ages_list, "embeddings": all_embeddings}
        save_dir = os.path.join("casual_conversations", "matrices")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "age_semantic_matrix.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved semantic results to {save_path}")
        
        labels = [pathlib.Path(p).stem for p in audio_paths_list]
        visualize_semantic_rdm(all_embeddings, labels)

if __name__ == "__main__":
    main()
