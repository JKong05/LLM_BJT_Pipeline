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

# Seamless Sonar encoder
def semantic_seamless_search(text_input, batch_size=500, overlap=50, source_lang="eng_Latn", max_tokens=512):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        dtype = torch.float16
    
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

    tokenized_text = encoder(text_input)
    token_ids = [token.item() if hasattr(token, "item") else token for token in tokenized_text]

    chunks_text = []
    for i in range(0, len(token_ids), max_tokens - overlap):
        chunk_ids = token_ids[i: i + max_tokens]
        chunk_text = decoder(torch.tensor(chunk_ids))
        chunks_text.append(chunk_text)
        print("Chunk: " + str(i))

    embeddings = []
    for chunk in chunks_text:
        emb = t2vec_model.predict([chunk], source_lang=source_lang)
        embeddings.append(emb)

    # average out the embeddings for the chunks for a long text input
    return torch.mean(torch.stack(embeddings), dim=0)


# jinaAi model
def semantic_search(text_input):
    model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True).to('cuda')

    embeddings = model.encode(text_input, task="text-matching", truncate_dim=512)

    return embeddings

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )
