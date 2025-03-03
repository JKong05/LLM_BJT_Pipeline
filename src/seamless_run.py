import numpy as np
import pathlib
import torch

from utils.prosody_processing import get_prosodic_embeddings
from utils.semantic_processing import get_semantic_embeddings, semantic_seamless_search


def extract_prosodic_embeddings(root_dir):
    # root_dir refers to the root folder for traversal to occur in
    root_path = pathlib.Path(root_dir)
    embs_list = []
    audio_paths = []
    print(root_dir)

    for part_dir in root_path.glob("CC_mini_part_*"):
        if not part_dir.is_dir():
            continue
            

        for audio_file in part_dir.rglob("*.wav"):
            print(f"Processing file: {audio_file}")
            try:
                emb = get_prosodic_embeddings(str(audio_file))

                if emb.dim() == 2 and emb.shape[0] == 1:
                        mb = emb.squeeze(0)
                    
                embs_list.append(emb)
                # in the future, adjust this logic to acdcount for the parpticipent identifier number
                audio_paths.append(audio_file)
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")

    if not embs_list:
        print("error")
        return None, None

    embeddings_matrix = torch.stack(embs_list, dim=0)
    return embeddings_matrix, audio_paths

def main():
    # Change the path below to the correct path for your audios folder

    embeddings_matrix, audio_files = extract_prosodic_embeddings("../casual_conversations/rawdata/audios")
    if embeddings_matrix is not None:
        print("Embeddings matrix shape:", embeddings_matrix.shape)

if __name__ == "__main__":
    main()