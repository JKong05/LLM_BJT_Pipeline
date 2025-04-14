import pandas as pd
import numpy as np
import pickle
import os
from input_processing import expressive_concat, semantic_concat, vectorize, compare_vectors
from utils.participant_utils import get_existing_participants, save_comparisons_to_csv

from model import generate_llm_response

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
participant_metadata = os.path.join(BASE_DIR, "../retellings/participant_metadata.csv")
stories_folder = os.path.join(BASE_DIR, "../stories/")
retelling_folder = os.path.join(BASE_DIR, "../retellings/")
samples_folder = os.path.join(BASE_DIR, "../story_samples/")
embeddings_path = os.path.join(BASE_DIR, "../embeddings.pkl")

all_comparison_output = os.path.join(BASE_DIR, "../results/all_participants.csv")
group_comparison_output = os.path.join(BASE_DIR, "../results/group_participants.csv")
source_comparison_output = os.path.join(BASE_DIR, "../results/source_participants.csv")


'''
run.py

Script for running the pipeline for the experimental data
'''


'''
initialization

Function that intializes the pipeline. Reads in localized pkl file
if it exists, which contains embeddings from previous runs, or will generate new 
prosodic or semantic embeddings using internal functions.

'''
def initialization():
    try:
        with open(embeddings_path, 'rb') as f:
            save_data = pickle.load(f)
            expressive_embs = save_data['expressive']
            semantic_embs = save_data['semantic']

            existing_participants = set(data['participant_id'] for data in expressive_embs.values())

            current_participants = get_existing_participants(retelling_folder)
            new_participants = current_participants - existing_participants
            removed_participants = existing_participants - current_participants

        # Check if participants were removed from the retelling folder
        if removed_participants:
            print(f"Participants {removed_participants} cannot be found")
            expressive_embs = {k: v for k, v in expressive_embs.items() if v['participant_id'] not in removed_participants}
            semantic_embs = {k: v for k, v in semantic_embs.items() if v['participant_id'] not in removed_participants}

            with open(embeddings_path, 'wb') as f:
                pickle.dump({'expressive': expressive_embs, 'semantic': semantic_embs}, f)

        elif new_participants:
            print(f"New participants added: {new_participants}")
            new_expressive_embs = expressive_concat(stories_folder, retelling_folder, samples_folder, participant_filter=new_participants)
            new_semantic_embs = semantic_concat(stories_folder, retelling_folder, participant_filter=new_participants)

            expressive_embs.update(new_expressive_embs)
            semantic_embs.update(new_semantic_embs)

            with open(embeddings_path, 'wb') as f:
                pickle.dump({'expressive': expressive_embs, 'semantic': semantic_embs}, f)

        if not (new_participants or removed_participants):
            print("No changes detected in participant directories")
        
        return expressive_embs, semantic_embs

    except FileNotFoundError:
        # expressive embedding
        expressive_embs = expressive_concat(stories_folder, retelling_folder, samples_folder)
        # semantic embedding
        semantic_embs = semantic_concat(stories_folder, retelling_folder)

        with open(embeddings_path, 'wb') as f:
            pickle.dump({'expressive': expressive_embs, 'semantic': semantic_embs}, f)
        
        return expressive_embs, semantic_embs

'''
age_integration


'''
def age_integration(expressive_embs, semantic_embs):
    metadata_df = pd.read_csv(participant_metadata)
    metadata_dict = metadata_df.set_index("participant_id").to_dict(orient="index")

    for key, data in expressive_embs.items():
        participant_id, story_id = key
        if participant_id in metadata_dict:
            data["age"] = metadata_dict[participant_id].get("age")
        else:
            data["age"] = None  # or handle missing age as needed

    # Merge age into semantic embeddings.
    for key, data in semantic_embs.items():
        participant_id, story_id = key
        if participant_id in metadata_dict:
            data["age"] = metadata_dict[participant_id].get("age")
        else:
            data["age"] = None 

    return expressive_embs, semantic_embs

# '''
# llm_comparison

# Generates and evaluate the LLM responses to the stories using Claude
# and creates semantic embeddings as well. Returns similarity between
# vectors of the LLM and the participants and saves to results folder.

# '''
# def llm_comparison(semantic_embeddings):
#     # Step 5: Compare semantic embeddings of participants to the semantic embeddings of the LLM (claude)
#     print("\nGenerating LLM comparisons")
#     df_llm = generate_llm_response(stories_folder, semantic_embeddings, output_llm)
#     df_llm.to_csv(output_llm, sep="\t", index=False)
#     print(f"\nllm performance results saved to: {output_llm}")
    

def main():
    expressive_embs, semantic_embs = initialization()
    expressive_embs, semantic_embs = age_integration(expressive_embs, semantic_embs)

    # congruence split
    story_groups = {
        "group1": ["story1", "story2", "story3"],
        "group2": ["story4", "story5", "story6"]
    }

    all_comparisons = pairwise_comparison_all(expressive_embs, semantic_embs)
    group_comparisons = pairwise_comparison_group(expressive_embs, semantic_embs, story_groups)
    source_comparisons = source_comparison(expressive_embs, semantic_embs):

    # csv formatting for all participant comparison
    save_comparisons_to_csv(all_comparisons, all_comparison_output,
                           desired_order=[
                               'story_id',
                               'participant_pair',
                               'expressive_similarity',
                               'semantic_similarity',
                               'participant1_modality',
                               'participant2_modality',
                               'participant1_congruence',
                               'participant2_congruence',
                               'participant1_age',
                               'participant2_age'
                           ])
    # csv formatting for grouped congruence comparison
    save_comparisons_to_csv(group_comparisons, group_comparison_output,
                            desired_order=[
                                'group',
                                'story_id',
                                'participant_pair',
                                'expressive_similarity',
                                'semantic_similarity',
                                'participant1_modality',
                                'participant2_modality',    
                                'participant1_congruence',
                                'participant2_congruence',
                                'participant1_age',
                                'participant2_age',
                                'participant1_group_congruence',
                                'participant2_group_congruence'
                           ])
    # output for participant to source audio
    save_comparisons_to_csv(source_comparisons, source_comparison_output,
                            desired_order=[
                                'participant_id',
                                'story_id',
                                'expressive_similarity',
                                'semantic_similarity',
                                'modality',
                                'congruence',
                                'age'
                           ])


if __name__ == "__main__":
    main()




