from transformers import AutoProcessor, SeamlessM4Tv2Model, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import torchaudio
import os
import whisper

# def get_semantic_embeddings(audio_path, story_id, participant_id, chunk_duration=10):
#     processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
#     model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

#     audio, orig_freq = torchaudio.load(audio_path)
#     audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000)
    
#     duration = audio.shape[1] / 16000

#     if duration > chunk_duration:
#         samples_per_chunk = int(chunk_duration * 16000)
#         num_chunks = int(torch.ceil(torch.tensor(audio.shape[1] / samples_per_chunk)))
        
#         translated_texts = []
        
#         for i in range(num_chunks):
#             start_idx = i * samples_per_chunk
#             end_idx = min((i + 1) * samples_per_chunk, audio.shape[1])
            
#             chunk = audio[:, start_idx:end_idx]
            
#             audio_inputs = processor(audios=chunk, return_tensors="pt", sampling_rate=16000)
#             output_tokens = model.generate(
#                 **audio_inputs,
#                 tgt_lang="eng",
#                 generate_speech=False,
#                 num_beams=6,
#                 do_sample=True
#             )
#             chunk_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
#             translated_texts.append(chunk_text)
            
#         translated_text = " ".join(translated_texts)
#         print(f"Token count for {participant_id}-{story_id} (combined {num_chunks} chunks): {len(output_tokens[0].tolist()[0])}")
#     else:
#         audio_inputs = processor(audios=audio, return_tensors="pt", sampling_rate=16000)
#         output_tokens = model.generate(
#             **audio_inputs, 
#             tgt_lang="eng", 
#             generate_speech=False
#         )
#         translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
#         print(f"Token count for {participant_id}-{story_id}: {len(output_tokens[0].tolist()[0])}")
    
#     output_filename = f"translations/{participant_id}_{story_id}_translation.txt"
#     os.makedirs('translations', exist_ok=True)  # Create translations directory if it doesn't exist
#     with open(output_filename, 'w', encoding='utf-8') as f:
#         f.write(translated_text)

#     return semantic_search(translated_text)

def get_semantic_embeddings(audio_path, story_id, participant_id, chunk_duration=10):
    model = whisper.load_model("turbo")

    audio = whisper.load_audio(audio_path)
    duration = len(audio) / 16000

    if duration > 30:
        # Process in 20-second chunks with 2-second overlap
        chunk_length = 20 * 16000  # 20 seconds
        overlap = 2 * 16000        # 2 seconds overlap
        transcribed_texts = []
        
        for i in range(0, len(audio), chunk_length - overlap):
            chunk = audio[i:i + chunk_length]
            result = model.transcribe(chunk)
            transcribed_texts.append(result["text"])
        
        full_text = " ".join(transcribed_texts)
    else:
        # Process entire audio if under 30 seconds
        result = model.transcribe(audio_path)
        full_text = result["text"]

    output_filename = f"translations/TTST/{participant_id}_{story_id}_translation.txt"
    os.makedirs('translations/TTST', exist_ok=True)
    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(full_text)

    return semantic_search(full_text)


def semantic_search(text_input):
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to('cuda')

    embeddings = model.encode(text_input, task="text-matching", truncate_dim=512)

    return embeddings

# def semantic_search(text_input):
#     # Initialize SentenceTransformer model
#     model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2').to('cuda')
    
#     # Generate embeddings (this will return 768-dimensional vectors)
#     embeddings = model.encode(text_input, 
#                             convert_to_tensor=True,  # Returns torch tensor
#                             normalize_embeddings=True)  # L2 normalize embeddings
    
#     return embeddings.reshape(1, -1)  # Ensure 2D shape (1, 768)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
