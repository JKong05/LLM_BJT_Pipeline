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
