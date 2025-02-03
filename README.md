# Analytical Pipeline
Code for LLM_BJT analytical pipeline (Fall24 ~ Spr24)

```
.
├── content
│   ├── SeamlessExpressive              # Models from SeamlessCommunication
├── results                             # Output results of pipeline on a participant-particpant basis
│   ├── figures
│   ├── outputs
├── retellings                          # The audio .wav files of the participants organized by a participant basis
├── src
│   ├── utils                           # Tools and utilities for inputing_processing and entrypoint run file
│   ├── input_processing.py             # Parent function handling audio and semantic embedding processing
│   ├── model.py                        # Handles LLM model integration
│   └── run.py                          # Entrypoint for application
├── stories
│   ├── audios                          # Raw audio files for the narrative stories
│   └── text                            # Raw text files for the narrative stories
├── story_samples
├── translations                        # Translations of participant responses (TTS) and LLM responses
│   ├── LLMTT                           # Large language model to text
│   └── TTST                            # Text to speech translation
├── README.md
└── visualize.ipynb
```

# Abstract (placeholder)
The human ability to understand and generate an endless variety of novel expressions has long captivated researchers in the fields of neuroimaging and neuroscience. In particular, academia has focused on understanding specific brain activity that prompts developmental linguistic behaviors of an individual. Developmental linguistics is the study of how individuals learn, retain, and process language throughout their lifetime. This concept can be broken down into two key aspects: what individuals say and how they say it. A traditional approach to the experimentation of decomposing language processing was to create highly-controlled and factorial design paradigms. This approach involved participants being exposed to meaningless words, phrases, and sounds in an atypical environment and collecting brain activity data through techniques such as mangeotencepahlography (MEG). However, this procedure often fails to reflect natural, everyday language exposure, raising concerns about its reliability and relevance in measuring high-level cognitive language processing. Recent literature has addressed these gaps in knowledge by substituting these designs with more practical approaches. Researchers now use narrative stories instead of abstract syntactic stimuli and computational language models to analyze features of speech. While these methods do provide refined infrastructure, it is pertinent to evaluate how the experimental environment may also influence linguistic behaviors. The process of developing an analytical pipeline to compare participant retellings of stories involved a two-pronged approach: extraction of prosodic and semantic features and the creation of representational embeddings. Prosodic features are elements of speech that contribute to the accent, rhythm, stress, intonation, tone, and pitch of the spoken language. Semantic features refer to specific characteristics of linguistics that constitute the meaning of the words that are being used. The combination of these two concepts help explain both the way in which an individual speaks and what their words mean. Therefore, a combination of multi-modal models were used to create embeddings.

# Quick Start
> [!NOTE]
> SeamlessExpressive can only be accessed on a request-basis. To download and integrate this model, follow instructions [here](https://github.com/JKong05/LLM_BJT_Pipeline/tree/main/content) or refer to [Seamless Communication](https://github.com/facebookresearch/seamless_communication/tree/main).
1. Clone repository
```
git clone https://github.com/JKong05/LLM_BJT_Pipeline.git
```
2. Install and integrate SeamlessExpressive models by submitting request for access and referring to README in `content`.
3. Create and initialize conda environment
```

```

