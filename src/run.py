import pandas as pd
import numpy as np
import pickle
import os
from input_processing import audio_concat, semantic_concat, vectorize, compare_vectors
    
from model import generate_llm_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
stories_folder = os.path.join(BASE_DIR, "../stories/")
retelling_folder = os.path.join(BASE_DIR, "../retellings/")
samples_folder = os.path.join(BASE_DIR, "../story_samples/")
embeddings_path = os.path.join(BASE_DIR, "../embeddings.pkl")
output_csv = os.path.join(BASE_DIR, "../results/output.csv")
output_path = os.path.join(BASE_DIR, "../results/performance_vectors.csv")
output_llm = os.path.join(BASE_DIR, "../results/llm_performance.csv")

'''
initialization

Function that intializes the pipeline. Reads in localized pkl file
if it exists, which contains embeddings from previous runs, or will generate new 
prosodic or semantic embeddings using internal functions.

'''
def initialization():
    try:
        # Access an existing pkl file to read in pre-existing data
        with open(embeddings_path, 'rb') as f:
            save_data = pickle.load(f)
            audio_embeddings = save_data['audio']
            semantic_embeddings = save_data['semantic']
            existing_participants = set(data['participant_id'] for data in audio_embeddings.values())

        current_participants = get_existing_participants(retelling_folder)
        new_participants = current_participants - existing_participants
        removed_participants = existing_participants - current_participants

        # Check if participants were removed from the retelling folder
        if removed_participants:
            print(f"Participants {removed_participants} cannot be found")
            audio_embeddings = {k: v for k, v in audio_embeddings.items() 
                            if v['participant_id'] not in removed_participants}
            semantic_embeddings = {k: v for k, v in semantic_embeddings.items() 
                                if v['participant_id'] not in removed_participants}
            with open(embeddings_path, 'wb') as f:
                pickle.dump({'audio': audio_embeddings, 'semantic': semantic_embeddings}, f)
        elif new_participants:
            print(f"New participants {new_participants} detected")
            new_audio_embeddings = audio_concat(stories_folder, retelling_folder, samples_folder, participant_filter=new_participants)
            new_semantic_embeddings = semantic_concat(stories_folder, retelling_folder, participant_filter=new_participants)

            audio_embeddings.update(new_audio_embeddings)
            semantic_embeddings.update(new_semantic_embeddings)

            with open(embeddings_path, 'wb') as f:
                pickle.dump({'audio': audio_embeddings, 'semantic': semantic_embeddings}, f)
        if not (new_participants or removed_participants):
            print("No changes detected in participant directories")

        return audio_embeddings, semantic_embeddings
    except FileNotFoundError:
        # Prosodic embeddings generation
        audio_embeddings = audio_concat(stories_folder, retelling_folder, samples_folder)
        # Semantic embeddings generation
        semantic_embeddings = semantic_concat(stories_folder, retelling_folder)

        with open(embeddings_path, 'wb') as f:
            pickle.dump({'audio': audio_embeddings, 'semantic': semantic_embeddings}, f)
        
        return audio_embeddings, semantic_embeddings

# Helper function to retrieve existing participants
def get_existing_participants(retelling_folder):
    participants = set()
    
    # Ensures no duplicate participant directories exist or are accounted for
    for participant_dir in os.listdir(retelling_folder):
        participant_path = os.path.join(retelling_folder, participant_dir)
        if os.path.isdir(participant_path) and participant_dir.startswith('p'):
            participants.add(participant_dir)
    return participants

'''
vectorization

Function that takes in audio and semantic embeddings and creates a combined
representational vector for similarity comparison. Outputs to results folder
in the form of df and csv.

'''
def vectorization(audio_embeddings, semantic_embeddings):
    performance_vectors = vectorize(audio_embeddings, semantic_embeddings)
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

    return performance_vectors

'''
vector_comparison

Function that takes in representational performance vectors and 
performs cosine similarity analysis on the vectors. Outputs similarity
scores to results folder.

'''
def vector_comparison(performance_vectors):
    comparative_data = compare_vectors(performance_vectors)
    df_comp = pd.DataFrame(comparative_data)

    df_comp['story_num'] = df_comp['story_id'].str.extract('(\d+)').astype(int)
    df_comp['p1_num'] = df_comp['participant1_id'].str.extract('(\d+)').astype(int)
    df_comp['p2_num'] = df_comp['participant2_id'].str.extract('(\d+)').astype(int)
    df_comp = df_comp.sort_values(['story_num', 'p1_num', 'p2_num'])
    df_comp = df_comp.drop(['story_num', 'p1_num', 'p2_num'], axis=1)
    df_comp.to_csv(output_csv, sep="\t", index=False)

'''
llm_comparison

Generates and evaluate the LLM responses to the stories using Claude
and creates semantic embeddings as well. Returns similarity between
vectors of the LLM and the participants and saves to results folder.

'''
def llm_comparison(semantic_embeddings):
    # Step 5: Compare semantic embeddings of participants to the semantic embeddings of the LLM (claude)
    print("\nGenerating LLM comparisons")
    df_llm = generate_llm_response(stories_folder, semantic_embeddings, output_llm)
    df_llm.to_csv(output_llm, sep="\t", index=False)
    print(f"\nllm performance results saved to: {output_llm}")
    

def main():
    audio_embeddings, semantic_embeddings = initialization()
    # 0: audio, 1: semantics
    performance_vectors = vectorization(audio_embeddings, semantic_embeddings)
    vector_comparison(performance_vectors)
    llm_comparison(semantic_embeddings)

if __name__ == "__main__":
    main()




