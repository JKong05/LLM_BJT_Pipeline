import anthropic
import os
import torch.nn.functional as F
import pickle
import pandas as pd
import torch

from dotenv import load_dotenv
from utils.semantic_processing import semantic_search
from sentence_transformers import SentenceTransformer


load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

def collect_participant_ages(semantic_embeddings):
    # get unique participant IDs from semantic_embeddings
    participant_ids = set(data['participant_id'] for data in semantic_embeddings.values())
    participant_ages = []
    
    print(f"\nFound {len(participant_ids)} participants. Please enter their ages:")
    
    for participant_id in sorted(participant_ids, key=lambda x: int(x[1:])):  # Sort by number (p1, p2, etc.)
        valid_age = False
        while not valid_age:
            try:
                age = int(input(f"Enter age for participant {participant_id}: "))
                if 0 <= age <= 120: 
                    participant_ages.append({
                        'participant_id': participant_id,
                        'age': age
                    })
                    valid_age = True
                else:
                    print("Invalid age!")
                    
            except ValueError:
                print("Invalid input!")
                
    results_dir = 'results'

    df_ages = pd.DataFrame(participant_ages)
    df_ages['participant_num'] = df_ages['participant_id'].str.extract('(\d+)').astype(int)
    df_ages = df_ages.sort_values('participant_num')
    df_ages = df_ages.drop('participant_num', axis=1)
    df_ages.to_csv(os.path.join(results_dir, 'participant_data.csv'), sep="\t", index=False)
    print("\nParticipant ages saved to participant_data.csv")
    
    return df_ages

def generate_llm_response(stories_folder, semantic_embeddings, output_llm):
    df_ages = collect_participant_ages(semantic_embeddings)
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    llm_comparisons = []
    llm_embeddings = {}

    # Group semantic embeddings by story
    story_groups = {}
    for key, data in semantic_embeddings.items():
        story_id = data['story_id']
        if story_id not in story_groups:
            story_groups[story_id] = []
        story_groups[story_id].append(data)

    text_folder = os.path.join(stories_folder, "text")
    for story_file in os.listdir(text_folder):
        if not story_file.endswith('.txt'):
            continue
            
        story_id = story_file.replace('.txt', '')
        print(f"\nProcessing story: {story_id}")
        
        # Read story
        with open(os.path.join(text_folder, story_file), 'r') as f:
            story_text = f.read()
        
        # Get participants for this story
        participants = story_groups.get(story_id, [])
        
        # Generate response for each participant's age
        for participant_data in participants:
            participant_id = participant_data['participant_id']
            age = df_ages[df_ages['participant_id'] == participant_id]['age'].iloc[0]
            
            print(f"Generating {age}-year-old retelling for comparison with {participant_id}")
            
            # Create age-specific system prompt
            system_prompt = f"1. You are a {age} year old that is retelling this story naturally and like you're telling a friend. 2. Maintain your age-appropriate vocabulary and speaking style throughout the retelling. 3. Start directly with the story - no introductory phrases like 'Okay' or 'Here's how'"
            
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4000,
                temperature=0.2,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": f"Retell this story as if you're a {age}-year-old: {story_text}"
                }]
            )

            llm_retelling = response.content[0].text
            llm_embedding = torch.tensor(semantic_search(llm_retelling))
            
            embedding_key = f"{participant_id}-{story_id}"
            llm_embeddings[embedding_key] = {
                'participant_id': participant_id,
                'story_id': story_id,
                'retelling_semantic': llm_embedding.tolist(),  # Convert tensor to list for serialization
                'participant_age': age,
                'retelling_text': llm_retelling
            }


            response_filename = f"translations/LLMTT/llm_{participant_id}-{story_id}_response.txt"
            os.makedirs('translations/LLMTT', exist_ok=True)
            with open(response_filename, 'w', encoding='utf-8') as f:
                f.write(llm_retelling)


            # Retrieve participant's semantic embedding
            participant_embedding = torch.tensor(participant_data['retelling_semantic'])
            
            # Compare embeddings
            semantic_similarity = F.cosine_similarity(
                llm_embedding.unsqueeze(0),
                participant_embedding.unsqueeze(0)
            ).item()
            
            llm_comparisons.append({
                'story_id': story_id,
                'participant_id': participant_id,
                'participant_age': age,
                'semantic_similarity': semantic_similarity
            })

    with open('llm_embeddings.pkl', 'wb') as f:
        pickle.dump(llm_embeddings, f)
    
    df_llm = pd.DataFrame(llm_comparisons)
    
    # Sort by story and participant numbers
    df_llm['participant_num'] = df_llm['participant_id'].str.extract('(\d+)').astype(int)
    df_llm['story_num'] = df_llm['story_id'].str.extract('(\d+)').astype(int)
    df_llm = df_llm.sort_values(['participant_num', 'story_num'])
    df_llm = df_llm.drop(['participant_num', 'story_num'], axis=1)
    
    return df_llm