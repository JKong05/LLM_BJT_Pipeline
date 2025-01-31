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

CHECKPOINTS_PATH = pathlib.Path("/home/wallacelab/LLM_BJT/pipeline/content/SeamlessExpressive")
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

if torch.cuda.is_available():
    device = torch.cuda.set_device(0)
    dtype = torch.float32

translator = Translator(
    model_name_or_card="seamless_expressivity",
    vocoder_name_or_card=None,
    device=device,
    dtype=dtype,
)

_gcmvn_mean, _gcmvn_std = load_gcmvn_stats("vocoder_pretssel")
gcmvn_mean = torch.tensor(_gcmvn_mean, device=device, dtype=dtype)  
gcmvn_std = torch.tensor(_gcmvn_std, device=device, dtype=dtype)

convert_to_fbank = WaveformToFbankConverter(
    num_mel_bins=80,
    waveform_scale=2**15,
    channel_last=True,
    standardize=False,
    device=device,
    dtype=dtype,
)

def load_audio(filepath, target_sample_rate=16000):
    waveform, sample_rate = torchaudio.load(filepath)
    if sample_rate != target_sample_rate:
        waveform = torchaudio.transforms.Resample(sample_rate, target_sample_rate)(waveform)
    waveform = waveform.mean(dim=0).unsqueeze(-1)  # Ensuring [time, channels] format
    return waveform, target_sample_rate


def normalize_fbank(data):
    fbank = data["fbank"]
    std, mean = torch.std_mean(fbank, dim=0)
    data["fbank"] = fbank.subtract(mean).divide(std)
    data["gcmvn_fbank"] = fbank.subtract(gcmvn_mean).divide(gcmvn_std)
    return data

def get_prosodic_embeddings(filepath, chunk_duration=10):
    waveform, sample_rate = load_audio(filepath)
    duration = waveform.shape[0] / sample_rate
    
    if duration > chunk_duration:
        samples_per_chunk = int(chunk_duration * sample_rate)
        num_chunks = int(torch.ceil(torch.tensor(waveform.shape[0] / samples_per_chunk)))
        chunk_embeddings = []
        
        for i in range(num_chunks):
            start_idx = i * samples_per_chunk
            end_idx = min((i + 1) * samples_per_chunk, waveform.shape[0])
            
            chunk = waveform[start_idx:end_idx]
            chunk = chunk.to(device=device, dtype=dtype)

            example = convert_to_fbank({"waveform": chunk, "sample_rate": sample_rate})
            example = normalize_fbank(example)

            prosody_encoder_input = example["gcmvn_fbank"]
            if len(prosody_encoder_input.shape) == 2:
                prosody_encoder_input = prosody_encoder_input.unsqueeze(0)

            with autocast():
                chunk_embedding = translator.model.prosody_encoder_model(prosody_encoder_input)
                chunk_embeddings.append(chunk_embedding)
        
        embeddings = torch.mean(torch.stack(chunk_embeddings), dim=0)
        
    else:
        waveform = waveform.to(device=device, dtype=dtype)
        example = convert_to_fbank({"waveform": waveform, "sample_rate": sample_rate})
        example = normalize_fbank(example)

        prosody_encoder_input = example["gcmvn_fbank"]
        if len(prosody_encoder_input.shape) == 2:
            prosody_encoder_input = prosody_encoder_input.unsqueeze(0)

        with autocast():
            embeddings = translator.model.prosody_encoder_model(prosody_encoder_input)

    return embeddings
