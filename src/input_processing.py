import os
import pandas as pd
from pydub import AudioSegment
import torch
import torch.nn.functional as F
import sys
import re
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
    
    retelling_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x[1].split('_')[0]))))
    
    for root, retelling_filename in retelling_files:
        try:
            participant_id, story_id, modality, correctness = parse_audio_filename(retelling_filename)
        except ValueError as e:
            print(e)
            continue
                
        if participant_filter and participant_id not in participant_filter:
            continue

        modality = check_modality(retelling_filename)
        key = (participant_id, story_id)
        
        retelling_path = os.path.join(root, retelling_filename)

        sample_filename = f"source_{story_id}_sample.wav"  
        sample_path = os.path.join(samples_folder, sample_filename)
        source_expressive_embedding = None  # default if source audio is missing
        story_filename = f"{story_id.capitalize()}.wav"
        story_source_path = os.path.join(audios_folder, story_filename)
        
        if os.path.exists(story_source_path):
            # Create sample audio from the source story.
            audio = AudioSegment.from_wav(story_source_path)
            sample = audio[:sample_duration_ms]
            sample.export(sample_path, format="wav")
            source_expressive_embedding = get_prosodic_embeddings(sample_path)
        else:
            print(f"Story file {story_filename} not found in {audios_folder}.")
            # The entry will still be preserved, but the source expressive embedding is None.

        retelling_expressive_embedding = get_prosodic_embeddings(retelling_path)

        expressive_embs[key] = {
            'participant_id': participant_id,
            'story_id': story_id,
            'retelling_expressive': retelling_expressive_embedding,
            'modality': modality,
            'congruence': correctness,
            'story_expressive': source_expressive_embedding,  # may be None for visual-only stories
            'story_path': sample_path,
            'retelling_path': retelling_path
        }
        print(f"Generating expressive embeddings for {participant_id} - {story_id}: {modality}")

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
                        key = (participant_id, story_id.lower())
                        
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
                        print(f"Generating semantic embeddings for {participant_id} - {story_id}: {modality}")

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
        # If group_stories contains story7, remove it.
        group_stories = [s for s in group_stories if s.lower() != "story7"]
        
        # Determine participants with entries for every story in this group.
        participants = set()
        for (participant, story) in expressive_embeddings.keys():
            if story in group_stories:
                participants.add(participant)
                
        # Check if participants have a consistent group-level congruence.
        valid_participants = []
        for participant in participants:
            group_cong = get_group_congruence(expressive_embeddings, group_stories, participant)
            if group_cong is not None:
                valid_participants.append((participant, group_cong))
        valid_participants.sort()  # sort for deterministic order

        # For each individual story in the group, do pairwise comparisons.
        for story in group_stories:
            # If story is story7, skip (this line can be omitted since we already filtered group_stories)
            if story.lower() == "story7":
                continue
                
            story_entries = []
            for participant, group_cong in valid_participants:
                key = (participant, story)
                if key in expressive_embeddings and key in semantic_embeddings:
                    story_entries.append((participant,
                                          expressive_embeddings[key],
                                          semantic_embeddings[key]))
            n = len(story_entries)
            for i in range(n):
                for j in range(i + 1, n):
                    p1, exp_data1, sem_data1 = story_entries[i]
                    p2, exp_data2, sem_data2 = story_entries[j]
                    if exp_data1.get('congruence') != exp_data2.get('congruence'):
                        continue
                    # Compute expressive similarity.
                    emb1_exp = exp_data1.get('retelling_expressive')
                    emb2_exp = exp_data2.get('retelling_expressive')
                    if not isinstance(emb1_exp, torch.Tensor):
                        emb1_exp = torch.tensor(emb1_exp)
                    if not isinstance(emb2_exp, torch.Tensor):
                        emb2_exp = torch.tensor(emb2_exp)

                    # tensor dimensionality check
                    emb1_exp, emb2_exp = zero_pad(emb1_exp, emb2_exp)
                    exp_sim = F.cosine_similarity(emb1_exp, emb2_exp, dim=-1).item()
                    
                    # Compute semantic similarity.
                    emb1_sem = sem_data1.get('retelling_semantic')
                    emb2_sem = sem_data2.get('retelling_semantic')
                    if not isinstance(emb1_sem, torch.Tensor):
                        emb1_sem = torch.tensor(emb1_sem)
                    if not isinstance(emb2_sem, torch.Tensor):
                        emb2_sem = torch.tensor(emb2_sem)

                    # tensor dimensionality check
                    emb1_sem, emb2_sem = zero_pad(emb1_sem, emb2_sem)
                    sem_sim = F.cosine_similarity(emb1_sem, emb2_sem, dim=0).item()
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
                        'participant2_age': exp_data2.get('age')
                    })
    return comparisons

def pairwise_comparison_all(expressive_embeddings, semantic_embeddings):
    story_entries = defaultdict(list)
    for key, exp_data in expressive_embeddings.items():
        # Extract story id.
        participant, story = key
        # Skip if this story is "story7".
        if story.lower() == "story7":
            continue
        sem_data = semantic_embeddings.get(key)
        if sem_data is None:
            continue  # Skip if there is no matching semantic entry.
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
                
                emb1_exp, emb2_exp = zero_pad(emb1_exp, emb2_exp)
                exp_sim = F.cosine_similarity(emb1_exp, emb2_exp, dim=-1).item()
                
                # Compute semantic similarity.
                emb1_sem = sem_data1.get('retelling_semantic')
                emb2_sem = sem_data2.get('retelling_semantic')
                if not isinstance(emb1_sem, torch.Tensor):
                    emb1_sem = torch.tensor(emb1_sem)
                if not isinstance(emb2_sem, torch.Tensor):
                    emb2_sem = torch.tensor(emb2_sem)
                
                emb1_sem, emb2_sem = zero_pad(emb1_sem, emb2_sem)
                sem_sim = F.cosine_similarity(emb1_sem, emb2_sem, dim=0).item()
                comparisons.append({
                    'story_id': story,
                    'participant_pair': f"{participant1}-{participant2}",
                    'expressive_similarity': exp_sim,
                    'semantic_similarity': sem_sim,
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
        # Skip story7
        if story.lower() == "story7":
            continue
            
        sem_data = semantic_embeddings.get(key)
        if sem_data is None:
            continue
        
        source_exp = exp_data.get('story_expressive')
        retelling_exp = exp_data.get('retelling_expressive')
        if source_exp is not None:
            if not isinstance(source_exp, torch.Tensor):
                source_exp = torch.tensor(source_exp)
            if not isinstance(retelling_exp, torch.Tensor):
                retelling_exp = torch.tensor(retelling_exp)
            source_exp, retelling_exp = zero_pad(source_exp, retelling_exp)
            exp_sim = F.cosine_similarity(source_exp, retelling_exp, dim=-1).item()
        else:
            exp_sim = None

        source_sem = sem_data.get('story_semantic')
        retelling_sem = sem_data.get('retelling_semantic')
        if not isinstance(source_sem, torch.Tensor):
            source_sem = torch.tensor(source_sem)
        if not isinstance(retelling_sem, torch.Tensor):
            retelling_sem = torch.tensor(retelling_sem)
        
        source_sem, retelling_sem = zero_pad(source_sem, retelling_sem)
        sem_sim = F.cosine_similarity(source_sem, retelling_sem, dim=0).item()
        
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
def meg_baseline_comparison(expressive_embeddings, semantic_embeddings, baseline_stories, story7_id="story7"):
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
                
                meg_exp, base_exp = zero_pad(meg_exp, base_exp)
                meg_sem, base_sem = zero_pad(meg_sem, base_sem)

                exp_sim = F.cosine_similarity(meg_exp, base_exp, dim=-1).item()
                sem_sim = F.cosine_similarity(meg_sem, base_sem, dim=0).item()
                
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
    base, _ = os.path.splitext(filename.lower())
    parts = base.split('_')
    # modality should be the third chunk:
    if len(parts) >= 3 and parts[2] in ("audio", "visual", "audiovisual"):
        return parts[2]
    # fallback: scan all parts for one of the keywords
    for m in ("audiovisual", "visual", "audio"):
        if m in parts:
            return m
    # default if we really can't parse it
    return "audio"

def zero_pad(t1: torch.Tensor, t2: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    d1, d2 = t1.size(-1), t2.size(-1)
    print(f"zero_pad called with sizes -> t1: {d1}, t2: {d2}")
    if d1 == d2:
        return t1, t2
    max_d = max(d1, d2)
    if d1 < max_d:
        pad = (0, max_d - d1)
        t1 = F.pad(t1, pad, "constant", 0)
    else:
        pad = (0, max_d - d2)
        t2 = F.pad(t2, pad, "constant", 0)
    return t1, t2
