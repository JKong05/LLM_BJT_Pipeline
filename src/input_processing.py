import os
import pandas as pd
from pydub import AudioSegment
import torch
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from collections import defaultdict
from utils.prosody_processing import get_prosodic_embeddings
from utils.semantic_processing import get_semantic_embeddings, semantic_seamless_search

'''
expressive_concat

'''
def expressive_concat(stories_folder, retelling_folder, samples_folder, participant_filter=None):
    expressive_embs = {}
    sample_duration_ms = 20 * 1000
    audios_folder = os.path.join(stories_folder, "audios")

    # Gather all retelling files (from retelling_folder)
    retelling_files = []
    for root, dirs, files in os.walk(retelling_folder):
        for f in files:
            if f.endswith(".wav"):
                retelling_files.append((root, f))
    
    # Optionally, sort the retelling files – here, using the numeric part at the start of filename.
    retelling_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x[1].split('_')[0]))))
    
    for root, retelling_filename in retelling_files:
        try:
            # Parse the retelling filename
            participant_id, story_id, modality, correctness = parse_audio_filename(retelling_filename)
        except ValueError as e:
            print(e)
            continue
                
        if participant_filter and participant_id not in participant_filter:
            continue

        # Use check_modality to get a final modality value if needed.
        modality = check_modality(retelling_filename)
        key = (participant_id, story_id)
        
        # Get the full path to the retelling file.
        retelling_path = os.path.join(root, retelling_filename)

        sample_filename = f"source_{story_id}_sample.wav"  
        sample_path = os.path.join(samples_folder, sample_filename)
        if not os.path.exists(sample_path):
            story_filename = f"{story_id.capitalize()}.wav" 
            story_source_path = os.path.join(audios_folder, story_filename)
            if not os.path.exists(story_source_path):
                print(f"Story file {story_filename} not found in {audios_folder}.")
                continue
            # Create sample audio from the source story.
            audio = AudioSegment.from_wav(story_source_path)
            sample = audio[:sample_duration_ms]
            sample.export(sample_path, format="wav")
        
        # Generate embeddings using the sample (as the story embedding)
        # and the retelling file (as the retelling embedding).
        source_expressive_embedding = get_prosodic_embeddings(sample_path)
        retelling_expressive_embedding = get_prosodic_embeddings(retelling_path)

        expressive_embs[key] = {
            'participant_id': participant_id,
            'story_id': story_id,
            'retelling_expressive': retelling_expressive_embedding,
            'modality': modality, 
            'congruence': correctness,
            'story_expressive': source_expressive_embedding,
            'story_path': sample_path,
            'retelling_path': retelling_path
        }
        print(f"Generating expressive embeddings for {participant_id} - story {story_id}: {modality}")

    return expressive_embs

'''
semantic_concat


'''
def semantic_concat(stories_folder, retelling_folder, participant_filter=None):
    semantic_embs = {}
    story_text_folder = os.path.join(stories_folder, "text")

    # Iterate over story text files (e.g., "story1.txt")
    for story_file in os.listdir(story_text_folder):
        if story_file.endswith(".txt"):
            story_id = os.path.splitext(story_file)[0]  # e.g., "story1"
            story_path = os.path.join(story_text_folder, story_file)
            with open(story_path, 'r') as f:
                story_text = f.read()
        
            # Traverse the retelling_folder to find matching retelling files
            for root, dirs, files in os.walk(retelling_folder):
                for retelling_filename in files:
                    # Check for retelling files that contain the story_id (case-insensitive)
                    if f"_{story_id.lower()}" in retelling_filename.lower() and retelling_filename.endswith(".wav"):
                        try:
                            # Parse the retelling filename to extract participant, story, modality, correctness
                            participant_id, parsed_story_id, modality, correctness = parse_audio_filename(retelling_filename)
                        except ValueError as e:
                            print(e)
                            continue

                        if participant_filter and participant_id not in participant_filter:
                            continue

                        modality = check_modality(retelling_filename)
                        key = (participant_id, story_id)
                        
                        retelling_path = os.path.join(root, retelling_filename)
                        source_semantic_embedding = semantic_seamless_search(story_text)
                        retelling_semantic_embedding = get_semantic_embeddings(retelling_path, story_id, participant_id)

                        semantic_embs[key] = {
                            'participant_id': participant_id,
                            'story_id': story_id,
                            'retelling_semantic': retelling_semantic_embedding,
                            'modality': modality,    # e.g., "audio", "visual", or "audiovisual"
                            'congruence': correctness,
                            'story_semantic': source_semantic_embedding,
                        }
                        print(f"Generated semantic embeddings for {participant_id} - {story_id}: {modality}")

    return semantic_embs
    

'''
comparison functions

'''

def get_group_congruence(expressive_embeddings, group_stories, participant_id):
    congruences = []
    for story in group_stories:
        key = (participant_id, story)
        if key in expressive_embeddings:
            congruences.append(expressive_embeddings[key].get('congruence'))
    if congruences and len(set(congruences)) == 1:
        return congruences[0]
    else:
        return None

def pairwise_comparison_group(expressive_embeddings, semantic_embeddings, story_groups):
    comparisons = []
    # Process each group separately.
    for group_name, group_stories in story_groups.items():
        # First, determine participants who have entries for every story in this group, and check group-level congruence.
        participants = set()
        for (participant, story) in expressive_embeddings.keys():
            if story in group_stories:
                participants.add(participant)
        # For each participant, check if they have a consistent congruence across the group (using expressive embeddings).
        valid_participants = []
        for participant in participants:
            group_cong = get_group_congruence(expressive_embeddings, group_stories, participant)
            if group_cong is not None:
                valid_participants.append((participant, group_cong))
        valid_participants.sort()  # sort for deterministic order

        # Now, for each individual story in the group, do pairwise comparisons among valid participants 
        # only if their group-level congruence is the same.
        for story in group_stories:
            # Get entries for this story from both dictionaries.
            story_entries = []
            for participant, group_cong in valid_participants:
                key = (participant, story)
                if key in expressive_embeddings and key in semantic_embeddings:
                    story_entries.append((participant,
                                          expressive_embeddings[key],
                                          semantic_embeddings[key]))
            # Now, compare all pairs for this story.
            n = len(story_entries)
            for i in range(n):
                for j in range(i + 1, n):
                    p1, exp_data1, sem_data1 = story_entries[i]
                    p2, exp_data2, sem_data2 = story_entries[j]
                    # Although valid_participants ensured group-level congruence, do a sanity check at the story level:
                    if exp_data1.get('congruence') != exp_data2.get('congruence'):
                        continue
                    # Compute expressive similarity.
                    emb1_exp = exp_data1.get('retelling_expressive')
                    emb2_exp = exp_data2.get('retelling_expressive')
                    if not isinstance(emb1_exp, torch.Tensor):
                        emb1_exp = torch.tensor(emb1_exp)
                    if not isinstance(emb2_exp, torch.Tensor):
                        emb2_exp = torch.tensor(emb2_exp)
                    exp_sim = F.cosine_similarity(emb1_exp.unsqueeze(0), emb2_exp.unsqueeze(0)).item()
                    # Compute semantic similarity.
                    emb1_sem = sem_data1.get('retelling_semantic')
                    emb2_sem = sem_data2.get('retelling_semantic')
                    if not isinstance(emb1_sem, torch.Tensor):
                        emb1_sem = torch.tensor(emb1_sem)
                    if not isinstance(emb2_sem, torch.Tensor):
                        emb2_sem = torch.tensor(emb2_sem)
                    sem_sim = F.cosine_similarity(emb1_sem.unsqueeze(0), emb2_sem.unsqueeze(0)).item()
                    comparisons.append({
                        'group': group_name,
                        'story_id': story,
                        'participant_pair': f"{p1}-{p2}",
                        'expressive_similarity': exp_sim,
                        'semantic_similarity': sem_sim,
                        'participant1_modality': exp_data1.get('modality'),
                        'participant2_modality': exp_data2.get('modality'),
                        'participant1_congruence': exp_data1.get('congruence'),
                        'participant2_congruence': exp_data2.get('congruence'),
                        'participant1_age': exp_data1.get('age'),
                        'participant2_age': exp_data2.get('age'),
                        'participant1_group_congruence': get_group_congruence(expressive_embeddings, group_stories, p1),
                        'participant2_group_congruence': get_group_congruence(expressive_embeddings, group_stories, p2)
                    })
    return comparisons

def pairwise_comparison_all(expressive_embeddings, semantic_embeddings):
    story_entries = defaultdict(list)
    for key, exp_data in expressive_embeddings.items():
        # Retrieve semantic data for the same key.
        sem_data = semantic_embeddings.get(key)
        if sem_data is None:
            continue  # Skip if there is no matching semantic entry.
        participant, story = key
        story_entries[story].append((participant, exp_data, sem_data))
    
    comparisons = []
    for story, entries in story_entries.items():
        n = len(entries)
        for i in range(n):
            for j in range(i+1, n):
                participant1, exp_data1, sem_data1 = entries[i]
                participant2, exp_data2, sem_data2 = entries[j]
                
                # Compute expressive similarity.
                emb1_exp = exp_data1.get('retelling_expressive')
                emb2_exp = exp_data2.get('retelling_expressive')
                if not isinstance(emb1_exp, torch.Tensor):
                    emb1_exp = torch.tensor(emb1_exp)
                if not isinstance(emb2_exp, torch.Tensor):
                    emb2_exp = torch.tensor(emb2_exp)
                expressive_sim = F.cosine_similarity(emb1_exp.unsqueeze(0), emb2_exp.unsqueeze(0)).item()
                
                # Compute semantic similarity.
                emb1_sem = sem_data1.get('retelling_semantic')
                emb2_sem = sem_data2.get('retelling_semantic')
                if not isinstance(emb1_sem, torch.Tensor):
                    emb1_sem = torch.tensor(emb1_sem)
                if not isinstance(emb2_sem, torch.Tensor):
                    emb2_sem = torch.tensor(emb2_sem)
                semantic_sim = F.cosine_similarity(emb1_sem.unsqueeze(0), emb2_sem.unsqueeze(0)).item()
                
                comparisons.append({
                    'story_id': story,
                    'participant_pair': f"{participant1}-{participant2}",
                    'expressive_similarity': expressive_sim,
                    'semantic_similarity': semantic_sim,
                    'participant1_modality': exp_data1.get('modality'),
                    'participant2_modality': exp_data2.get('modality'),
                    'participant1_congruence': exp_data1.get('congruence'),
                    'participant2_congruence': exp_data2.get('congruence'),
                    'participant1_age': exp_data1.get('age'),
                    'participant2_age': exp_data2.get('age')
                })
    return comparisons

def source_comparison(expressive_embeddings, semantic_embeddings):
    comparisons = []
    for key, exp_data in expressive_embeddings.items():
        participant, story = key
        
        # Retrieve corresponding semantic data; skip if missing.
        sem_data = semantic_embeddings.get(key)
        if sem_data is None:
            continue
        
        # Expressive embeddings from source and retelling.
        source_exp = exp_data.get('story_expressive')
        retelling_exp = exp_data.get('retelling_expressive')
        # Ensure they're torch tensors.
        if not isinstance(source_exp, torch.Tensor):
            source_exp = torch.tensor(source_exp)
        if not isinstance(retelling_exp, torch.Tensor):
            retelling_exp = torch.tensor(retelling_exp)
        exp_sim = F.cosine_similarity(source_exp.unsqueeze(0), retelling_exp.unsqueeze(0)).item()
        
        # Semantic embeddings from source and retelling.
        source_sem = sem_data.get('story_semantic')
        retelling_sem = sem_data.get('retelling_semantic')
        if not isinstance(source_sem, torch.Tensor):
            source_sem = torch.tensor(source_sem)
        if not isinstance(retelling_sem, torch.Tensor):
            retelling_sem = torch.tensor(retelling_sem)
        sem_sim = F.cosine_similarity(source_sem.unsqueeze(0), retelling_sem.unsqueeze(0)).item()
        
        comparisons.append({
            'participant_id': participant,
            'story_id': story,
            'expressive_similarity': exp_sim,
            'semantic_similarity': sem_sim,
            'modality': exp_data.get('modality'),
            'congruence': exp_data.get('congruence'),
            'age': exp_data.get('age')
        })
    return comparisons

'''
comparison functions with MEG story

'''
def MEG_baseline_comparison(expressive_embeddings, semantic_embeddings, baseline_stories, story7_id="story7"):
    comparisons = []

    baseline_data = defaultdict(dict)
    for key, exp_data in expressive_embeddings.items():
        participant, story = key
        if story in baseline_stories:
            baseline_data[participant][story] = exp_data
    baseline_sem_data = defaultdict(dict)
    for key, sem_data in semantic_embeddings.items():
        participant, story = key
        if story in baseline_stories:
            baseline_sem_data[participant][story] = sem_data

    MEG_exps = defaultdict(list)
    MEG_sems = defaultdict(list)
    for key, exp_data in expressive_embeddings.items():
        participant, story = key
        if story == story7_id:
            # If more than one retelling exists, assume the value is a list.
            if isinstance(exp_data, list):
                MEG_exps[participant].extend(exp_data)
            else:
                MEG_exps[participant].append(exp_data)
    for key, sem_data in semantic_embeddings.items():
        participant, story = key
        if story == story7_id:
            if isinstance(sem_data, list):
                MEG_sems[participant].extend(sem_data)
            else:
                MEG_sems[participant].append(sem_data)
                
    # Now, for each participant who has both baseline data and MEG retellings, compare.
    for participant in baseline_data.keys():
        if participant not in MEG_exps or participant not in MEG_sems:
            continue
        # For each baseline story, retrieve the baseline entry.
        for base_story in baseline_stories:
            if base_story not in baseline_data[participant] or base_story not in baseline_sem_data[participant]:
                continue
            base_exp_data = baseline_data[participant][base_story]
            base_sem_data = baseline_sem_data[participant][base_story]
            # Convert baseline embeddings to tensors.
            base_exp = base_exp_data.get('retelling_expressive')
            if not isinstance(base_exp, torch.Tensor):
                base_exp = torch.tensor(base_exp)
            base_sem = base_sem_data.get('retelling_semantic')
            if not isinstance(base_sem, torch.Tensor):
                base_sem = torch.tensor(base_sem)
            # For each MEG retelling for this participant, compare.
            for idx, meg_exp_data in enumerate(MEG_exps[participant]):
                # Retrieve the corresponding MEG semantic data.
                # We assume that the lists for expressive and semantic are aligned.
                meg_sem_data = MEG_sems[participant][idx]
                meg_exp = meg_exp_data.get('retelling_expressive')
                if not isinstance(meg_exp, torch.Tensor):
                    meg_exp = torch.tensor(meg_exp)
                meg_sem = meg_sem_data.get('retelling_semantic')
                if not isinstance(meg_sem, torch.Tensor):
                    meg_sem = torch.tensor(meg_sem)
                
                exp_sim = F.cosine_similarity(meg_exp.unsqueeze(0), base_exp.unsqueeze(0)).item()
                sem_sim = F.cosine_similarity(meg_sem.unsqueeze(0), base_sem.unsqueeze(0)).item()
                
                comparisons.append({
                    'participant_id': participant,
                    'baseline_story': base_story,
                    'MEG_index': idx,
                    'MEG_id': story7_id,
                    'retelling_expressive_similarity': exp_sim,
                    'retelling_semantic_similarity': sem_sim,
                    'age': base_exp_data.get('age'),
                    'baseline_modality': base_exp_data.get('modality'),
                    'baseline_congruence': base_exp_data.get('congruence'),
                    'MEG_modality': meg_exp_data.get('modality'),
                    'MEG_congruence': meg_exp_data.get('congruence'),
                })
    return comparisons


'''
helper functions
 - parse_audio_filename -> parses the retelling file for correct structure
 - check_modality -> check the modality of the file in which participant received stimuli
'''

def parse_audio_filename(filename):
    pattern = r"^(p\d+)_((?:[sS]tory\d+))_(\w+)_(\w+)\.wav$"
    match = re.match(pattern, filename, re.IGNORECASE)
    if match:
        participant_id = match.group(1)            
        story_id = match.group(2).lower()         
        modality = match.group(3)                     
        correctness = match.group(4)                 
        return participant_id, story_id, modality, correctness
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")

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