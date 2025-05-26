from transformers import AutoProcessor, SeamlessM4Tv2Model, AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sonar.inference_pipelines.text import TextToTextModelPipeline, TextToEmbeddingModelPipeline

import torch
import torch.nn.functional as F
import torchaudio
import os
import whisper

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

    return semantic_seamless_search(full_text)

# 
def semantic_seamless_search(text_input, batch_size=500, overlap=50, source_lang="eng_Latn", max_tokens=512):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    
    # initialize encoders and decoders
    text_tokenization_model = TextToTextModelPipeline(
        encoder="text_sonar_basic_encoder",
        decoder="text_sonar_basic_decoder",
        tokenizer="text_sonar_basic_encoder",
        device=device
    )
    t2vec_model = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder"
    )
    encoder = text_tokenization_model.tokenizer.create_encoder(lang=source_lang)
    decoder = text_tokenization_model.tokenizer.create_decoder()
    tokenized = encoder(text_input)
    token_ids = [t.item() if hasattr(t, "item") else t for t in tokenized]

    stride = max_tokens - overlap
    windows = []
    for start in range(0, len(token_ids), stride):
        end = min(start + max_tokens, len(token_ids))
        window_ids = token_ids[start:end]
        text_chunk = decoder(torch.tensor(window_ids, device=device))
        windows.append(text_chunk)

    embeddings = []
    for chunk in windows:
        emb = t2vec_model.predict([chunk], source_lang=source_lang)
        if emb.dim() == 2 and emb.shape[0] == 1:
            emb = emb.squeeze(0)
        embeddings.append(emb.to(device))

    total_vector = torch.cat(embeddings, dim=-1)  
    return total_vector

