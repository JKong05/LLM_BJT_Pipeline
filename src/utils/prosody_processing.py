import os
import pathlib
import torch.nn.functional as F
import torch
import torchaudio
from fairseq2.assets import InProcAssetMetadataProvider, asset_store
from fairseq2.data.audio import ( AudioDecoder, WaveformToFbankConverter, WaveformToFbankOutput )
from seamless_communication.models.unity import load_gcmvn_stats
from torch.cuda.amp import autocast
from fairseq2.generation import NGramRepeatBlockProcessor
from fairseq2.memory import MemoryBlock
from huggingface_hub import snapshot_download
from seamless_communication.inference import Translator, SequenceGeneratorOptions
from seamless_communication.models.unity import (
    load_gcmvn_stats,
    load_unity_unit_tokenizer,
)

# Set up device objects:
cpu_device = torch.device("cpu")
if torch.cuda.is_available():
    gpu_device = torch.device("cuda:0")
    dtype = torch.float32
else:
    gpu_device = torch.device("cpu")
    dtype = torch.float32

CHECKPOINTS_PATH = pathlib.Path("/home/wallacelab/teba/multimodal_brain_inspired/LLM_BJT/pipeline/content/SeamlessExpressive")
if not CHECKPOINTS_PATH.exists():
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id="facebook/seamless-expressive", repo_type="model", local_dir=CHECKPOINTS_PATH)

asset_store.env_resolvers.clear()
asset_store.env_resolvers.append(lambda: "demo")

demo_metadata = [
    {
        "name": "seamless_expressivity@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/m2m_expressive_unity.pt"
    },
    {
        "name": "vocoder_pretssel@demo",
        "checkpoint": f"file://{CHECKPOINTS_PATH}/pretssel_melhifigan_wm-16khz.pt",
    }
]

asset_store.metadata_providers.append(InProcAssetMetadataProvider(demo_metadata))

# The Translator (and therefore the model) is loaded on GPU:
translator = Translator(
    model_name_or_card="seamless_expressivity",
    vocoder_name_or_card=None,
    device=gpu_device,  # inference runs here
    dtype=dtype,
)

# Load GCMVN stats on CPU
_gcmvn_mean, _gcmvn_std = load_gcmvn_stats("vocoder_pretssel")
gcmvn_mean_cpu = torch.tensor(_gcmvn_mean, device=cpu_device, dtype=dtype)  
gcmvn_std_cpu = torch.tensor(_gcmvn_std, device=cpu_device, dtype=dtype)

# Create fbank converter on CPU:
convert_to_fbank_cpu = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=False,
    device=cpu_device,
    dtype=dtype,
)

def load_audio(filepath, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
    # Waveform is on CPU by default; convert to mono and add channel dimension
    waveform = waveform.mean(dim=0).unsqueeze(-1)  # Ensuring [time, channels] format
    return waveform, target_sample_rate

def normalize_fbank(data):
    fbank = data["fbank"]
    std, mean = torch.std_mean(fbank, dim=0)
    normalized_fbank = (fbank - mean) / std
    data["fbank"] = normalized_fbank
    data["gcmvn_fbank"] = (fbank - gcmvn_mean_cpu) / gcmvn_std_cpu
    return data

def get_prosodic_embeddings(filepath, chunk_duration=10):
    """
    Performs fbank conversion and normalization on CPU.
    Moves only the minimal tensor to GPU for inference.
    Each chunk's result is moved back to CPU immediately to avoid accumulating GPU memory.
    """
    waveform, sample_rate = load_audio(filepath)
    num_samples = waveform.shape[0]
    duration    = num_samples / sample_rate

    # If longer than chunk_duration, we'll chunk; else do the original single‐pass
    if duration > chunk_duration:
        samples_per_chunk = int(chunk_duration * sample_rate)
        num_chunks = int(torch.ceil(torch.tensor(num_samples / samples_per_chunk)))
        chunk_embeddings = []

        for i in range(num_chunks):
            start = i * samples_per_chunk
            end   = min(start + samples_per_chunk, num_samples)
            chunk = waveform[start:end]
            if chunk.shape[0] < samples_per_chunk:
                pad_amt = samples_per_chunk - chunk.shape[0]
                chunk = F.pad(chunk, (0, 0, 0, pad_amt))

            # CPU fbank + norm
            ex = convert_to_fbank_cpu({"waveform": chunk, "sample_rate": sample_rate})
            ex = normalize_fbank(ex)
            inp = ex["gcmvn_fbank"]
            if inp.ndim == 2:
                inp = inp.unsqueeze(0)  # → [1, T, F]

            inp = inp.to(gpu_device, dtype=dtype)
            with torch.no_grad(), autocast():
                emb = translator.model.prosody_encoder_model(inp)  # [1, E]
            emb = emb.cpu()  # keep the batch dim
            chunk_embeddings.append(emb)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # [num_chunks, E]
        seq = torch.cat(chunk_embeddings, dim=0)

    else:
        # Original single‐pass branch
        ex = convert_to_fbank_cpu({"waveform": waveform, "sample_rate": sample_rate})
        ex = normalize_fbank(ex)
        inp = ex["gcmvn_fbank"]
        if inp.ndim == 2:
            inp = inp.unsqueeze(0)  # → [1, T, F]

        inp = inp.to(gpu_device, dtype=dtype)
        with torch.no_grad(), autocast():
            seq = translator.model.prosody_encoder_model(inp)  # [1, E]
        seq = seq.cpu()

    # Flatten [N, E] into [N * E]
    expressivity_vector = seq.flatten()
    return expressivity_vector
