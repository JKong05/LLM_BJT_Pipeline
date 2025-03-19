import os
import pandas as pd
from pydub import AudioSegment
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.prosody_processing import get_prosodic_embeddings
from utils.semantic_processing import get_semantic_embeddings, semantic_seamless_search

def prosodic_concat(stories_folder, retelling_folder, samples_folder, participant_filter=None):
    prosodic_embeddings = {}
    sample_duration_ms = 20 * 1000
    audios_folder = os.path.join(stories_folder, "audios")

    filenames = sorted(
        [f for f in os.listdir(audios_folder) if f.endswith(".wav")],
        key=lambda x: int(''.join(filter(str.isdigit, x)))
    )
    
    for filename in filenames:
        story_path = os.path.join(audios_folder, filename)
        story_id = os.path.splitext(filename)[0]
        
        # Create sample audio only if needed
        sample_filename = f"{story_id}_sample.wav"
        sample_path = os.path.join(samples_folder, sample_filename)
        if not os.path.exists(sample_path):
            audio = AudioSegment.from_wav(story_path)
            sample = audio[:sample_duration_ms]
            sample.export(sample_path, format="wav")
        
        retelling_files = []
        for root, dirs, files in os.walk(retelling_folder):
            for retelling_filename in files:
                if f"_{story_id.lower()}" in retelling_filename.lower() and retelling_filename.endswith(".wav"):
                    retelling_files.append((root, retelling_filename))
        
        retelling_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x[1].split('_')[0]))))
        
        for root, retelling_filename in retelling_files:
            participant_id = retelling_filename.split("_")[0]
            if participant_filter and participant_id not in participant_filter:
                continue

            modality = check_modality(retelling_filename)

            key = (participant_id, story_id)
            # Generate new embeddings if they don't exist
            retelling_path = os.path.join(root, retelling_filename)
            story_embedding = get_prosodic_embeddings(sample_path)
            retelling_embedding = get_prosodic_embeddings(retelling_path)

            prosodic_embeddings[key] = {
                'participant_id': participant_id,
                'story_id': story_id,
                'story_prosodic': story_embedding,
                'retelling_prosodic': retelling_embedding,
                'story_path': sample_path,
                'retelling_path': retelling_path,
                'modality': modality
            }
            print(f"Generating prosodic embeddings for {participant_id} - {story_id}: {modality}")

    return prosodic_embeddings

def semantic_concat(stories_folder, retelling_folder, participant_filter=None):
    semantic_embeddings = {}
    story_text_folder = os.path.join(stories_folder, "text")

    for story_file in os.listdir(story_text_folder):
        if story_file.endswith(".txt"):
            story_id = os.path.splitext(story_file)[0]
            story_path = os.path.join(story_text_folder, story_file)
            with open(story_path, 'r') as f:
                story_text = f.read()
        
            for root, dirs, files in os.walk(retelling_folder):
                for retelling_filename in files:
                    if f"_{story_id.lower()}" in retelling_filename.lower() and retelling_filename.endswith(".wav"):
                        participant_id = retelling_filename.split("_")[0]
                        if participant_filter and participant_id not in participant_filter:
                            continue

                        modality = check_modality(retelling_filename)

                        key = (participant_id, story_id)
                        # Generate new embeddings if they don't exist
                        retelling_path = os.path.join(root, retelling_filename)
                        semantic_story_embedding = semantic_seamless_search(story_text)
                        semantic_retelling_embedding = get_semantic_embeddings(retelling_path, story_id, participant_id)

                        semantic_embeddings[key] = {
                            'participant_id': participant_id,
                            'story_id': story_id,
                            'story_semantic': semantic_story_embedding,
                            'retelling_semantic': semantic_retelling_embedding,
                            'modality': modality
                        }
                        print(f"Generated semantic embeddings for {participant_id} - {story_id}: {modality}")

    return semantic_embeddings

def vectorize(prosodic_embeddings, semantic_embeddings, normalize=True):
    performance_vectors = {}

    for key in prosodic_embeddings.keys():
        if key in semantic_embeddings:
            participant_id, story_id = key

            prosodic_embs = prosodic_embeddings[key]
            semantic_embs = semantic_embeddings[key]
            
            # The prosodic and semantic characteristics of the stories.
            story_vector = combine_vectors(
                prosodic_embs['story_prosodic'],
                semantic_embs['story_semantic']
            )

            # The prosodic and semantic characteristics of the retellings.
            retelling_vector = combine_vectors(
                prosodic_embs['retelling_prosodic'],
                semantic_embs['retelling_semantic']
            )

            participant_similarity = F.cosine_similarity(
                ...
            )

            prosodic_similarity = F.cosine_similarity()
            semantic_similarity = F.cosine_similarity()





            # story_vector = combine_vectors(
            #     semantic_embs['story_semantic'],
            #     prosodic_embs['story_audio']
            # )
            # retelling_vector = combine_vectors(
            #     semantic_embs['retelling_semantic'],
            #     prosodic_embs['retelling_audio']
            # )
            # vector_similarity = F.cosine_similarity(
            #     story_vector.unsqueeze(0), 
            #     retelling_vector.unsqueeze(0)
            # ).item()
            # semantic_similarity = F.cosine_similarity(
            #     semantic_embs['story_semantic'].unsqueeze(0),
            #     semantic_embs['retelling_semantic'].unsqueeze(0)
            # ).item()

            # audio_similarity = F.cosine_similarity(
            #     prosodic_embs['story_audio'].squeeze(0).unsqueeze(0),
            #     prosodic_embs['retelling_audio'].squeeze(0).unsqueeze(0)
            # ).item()

            # performance_vectors[key] = {
            #     'modality': prosodic_embs['modality'],
            #     'story_vector': story_vector,
            #     'retelling_vector': retelling_vector,
            #     'vector_similarity': vector_similarity,
            #     'semantic_similarity': semantic_similarity,
            #     'prosodic_similarity': prosodic_similarity,
            #     'participant_id': participant_id,
            #     'story_id': story_id,
            #     'story_semantic_embedding': semantic_embs['story_semantic'],
            #     'retelling_semantic_embedding': semantic_embs['retelling_semantic'],
            #     'story_audio_embedding': prosodic_embs['story_audio'],
            #     'retelling_audio_embedding': prosodic_embs['retelling_audio']
            # }
            # print(f"Processed vectors for {participant_id} - {story_id}")

    return 0

def combine_vectors(prosodic_embs, semantic_embs):
    

    
    print(f"Semantic Embedding Shape: {semantic_emb.shape}, dtype: {semantic_emb.dtype}, device: {semantic_emb.device}")
    print(f"Audio Embedding Shape: {audio_emb.shape}, dtype: {audio_emb.dtype}, device: {audio_emb.device}")



    

def compare_vectors(performance_vectors):
    data = []
    story_groups = {}
    for key, data_point in performance_vectors.items():
        story_id = data_point['story_id']
        if story_id not in story_groups:
            story_groups[story_id] = []
        story_groups[story_id].append(data_point)

    for story_id, participants in story_groups.items():
        # Only compare each pair once: i with all j > i
        for i, data1 in enumerate(participants):
            for j in range(i + 1, len(participants)):
                data2 = participants[j]
                vector_similarity = F.cosine_similarity(
                    data1['retelling_vector'].unsqueeze(0),
                    data2['retelling_vector'].unsqueeze(0)
                ).item()
                semantic_similarity = F.cosine_similarity(
                    data1['retelling_semantic_embedding'].unsqueeze(0),
                    data2['retelling_semantic_embedding'].unsqueeze(0)
                ).item()
                audio_similarity = F.cosine_similarity(
                    data1['retelling_audio_embedding'].squeeze(0).unsqueeze(0),
                    data2['retelling_audio_embedding'].squeeze(0).unsqueeze(0)
                ).item()
            
                comparison_dict = {
                    'participant_pair': f"{data1['participant_id']}-{data2['participant_id']}",
                    'participant1_id': data1['participant_id'],
                    'participant2_id': data2['participant_id'],
                    'story_id': story_id,
                    'vector_similarity': vector_similarity,
                    'semantic_similarity': semantic_similarity,
                    'audio_similarity': audio_similarity,
                    'participant1_modality': data1['modality'],
                    'participant2_modality': data2['modality']
                }
                data.append(comparison_dict)
    
    return data

# helper function to check modality of a retelling file
def check_modality(filename):
    filename_parts = filename.lower().split('_')
    modality = 'audio'  # default
    if len(filename_parts) > 2:
        if 'audiovisual' in filename_parts[-1]:
            modality = 'audiovisual'
        elif 'visual' in filename_parts[-1]:
            modality = 'visual'
        elif 'audio' in filename_parts[-1]:
            modality = 'audio'
    return modality