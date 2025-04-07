import numpy as np
import torch
import torch.nn as nn
import os

from utils.prosody_processing import get_prosodic_embeddings
from utils.semantic_processing import semantic_seamless_search

def get_audio_file(transcription_key, root_audio="../casual_conversations/rawdata/audios"):
    path_ = transcription_key.replace(".MP4", ".wav") # adjust file extension
    
    for root, dirs, files in os.walk(root_audio):
        candidate_path = os.path.join(root, path_)
        if os.path.isfile(candidate_path):
            return candidate_path
    
    raise FileNotFoundError(f"Audio file not found for transcription key {transcription_key}")

def extract_latent_vectors(transcription_key, cleaned_transcription):
    semantic_vector = semantic_seamless_search(cleaned_transcription)
    expressive_vector = get_prosodic_embeddings(str(get_audio_file(transcription_key)))
    return semantic_vector, expressive_vector


'''
4.2 - need to discuss with Dr. Tovar whether this is the correct approach or not
'''
class FusionModule(nn.Module):
    def __init__(self, semantic_dim=1024, expressive_dim=512, common_dim=512):
        super(FusionModule, self).__init__()
        # linear projection into common space
        self.semantic_proj = nn.Linear(semantic_dim, common_dim)
        self.expressive_proj = nn.Linear(expressive_dim, common_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5)) # for now, set the weight at 0.5

    def forward(self, semantic_emb, expressive_emb):
        semantic = self.semantic_proj(semantic_emb)
        expressive = self.expressive_proj(expressive_emb)
        fused = self.alpha * semantic + (1 - self.alpha) * expressive
        return fused 

class AgeModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        super(AgeModel, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, 1)) 
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)