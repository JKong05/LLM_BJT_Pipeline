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

# Output results for reperesentational vector to csv
def performance_vector_results(performance_vectors, output_path):
    rows = []
    for key, data in performance_vectors.items():
        row = {
            'participant_id': data['participant_id'],
            'story_id': data['story_id'],
            'vector_similarity': data['vector_similarity'],
            'semantic_similarity': data['semantic_similarity'],
            'audio_similarity': data['audio_similarity'],
            'modality': data['modality'],
            'story_vector_size': data['story_vector'].shape[0], 
            'retelling_vector_size': data['retelling_vector'].shape[0],
            'retelling_semantic_embedding': data['retelling_semantic_embedding'].shape[0],
            'retelling_audio_embedding': data['retelling_audio_embedding'].size(),
            'story_semantic_embedding': data['story_semantic_embedding'].shape[0],
            'story_audio_embedding': data['story_audio_embedding'].size()
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    df['story_num'] = df['story_id'].str.extract('(\d+)').astype(int)
    df['participant_num'] = df['participant_id'].str.extract('(\d+)').astype(int)
    df = df.sort_values(['story_num', 'participant_num'])
    df = df.drop(['story_num', 'participant_num'], axis=1)  # Remove helper columns
    df.to_csv(output_path, sep="\t", index=False)

# Output similarity score results to csv
def vector_comparison_results(performance_vectors):
    df_comp = pd.DataFrame(comparative_data)

    df_comp['story_num'] = df_comp['story_id'].str.extract('(\d+)').astype(int)
    df_comp['p1_num'] = df_comp['participant1_id'].str.extract('(\d+)').astype(int)
    df_comp['p2_num'] = df_comp['participant2_id'].str.extract('(\d+)').astype(int)
    df_comp = df_comp.sort_values(['story_num', 'p1_num', 'p2_num'])
    df_comp = df_comp.drop(['story_num', 'p1_num', 'p2_num'], axis=1)
    df_comp.to_csv(output_csv, sep="\t", index=False)
