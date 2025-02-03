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
Cool abstract bro

# Quick Start

## Prerequisites
SeamlessExpressive requires the [fairseq2](https://github.com/facebookresearch/fairseq2) library to run. It specifically provides the infrastructure for the [wav2vec 2.0](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/models/wav2vec2) model that is used for speech-to-text translations and embedding generation. Depending on your operating system, installation will vary. Once fairseq2 has been installed on your system, you can integrate it into the pipeline using pip.
```
# Install once fairseq2 has been installed on your system

pip install fairseq2
```

### Linux
For Linux users, fairseq2 depends on [libsndfile](https://github.com/libsndfile/libsndfile), which can be installed via the system package manager on most distributions.
- On **ubuntu systems**, run
  ```
  sudo apt install libsndfile1
  ```
- On **Fedora**, run
  ```
  sudo dnf install libsndfile
  ```
### macOS
For macOS users, fairseq2 depends on [libsndfile](https://github.com/libsndfile/libsndfile) as well, which can be installed via Homebrew.
- In terminal, run
  ```
  brew install libsndfile
  ```
### Windows
There is no currently no native support for Windows installation of fairseq2, but instructions for use on a Windows system is available in [documentation](https://github.com/facebookresearch/fairseq2?tab=readme-ov-file).




## Installation
> [!NOTE]
> SeamlessExpressive can only be accessed on a request-basis. To download and integrate this model, follow instructions [here](https://github.com/JKong05/LLM_BJT_Pipeline/tree/main/content) or refer to [Seamless Communication](https://github.com/facebookresearch/seamless_communication/tree/main).
1. Clone repository
```
git clone https://github.com/JKong05/LLM_BJT_Pipeline.git
```

2. Install and integrate SeamlessExpressive models by submitting request for access and referring to README in `/content`
 
> [!NOTE]
> For someone who may not have conda installed, visit the download page [here](https://docs.anaconda.com/miniconda/) to download the appropriate installer based on your operating system. Refer to [documentation](https://www.anaconda.com/download/success) if you need guidance in the installation process. Additionally, use `conda deactivate` if the environment no longer needs to be used.
3. Create and initialize conda environment.
```
# Step 1: init environment
conda env create -f pipeline_environment.yml

# Step 2: activate the environment
conda activate LLM_pipeline
```
4. Add inputs for pipeline
   - Refer to README in `/retellings` or [here](https://github.com/JKong05/LLM_BJT_Pipeline/tree/main/retellings) for information on subfolder formatting.

5. Run the application
```
# from root
python src/run.py

# from src
cd src
python run.py
```

# Citation
```
@inproceedings{seamless2023,
   title="Seamless: Multilingual Expressive and Streaming Speech Translation",
   author="{Seamless Communication}, Lo{\"i}c Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-juss{\`a}, Maha Elbayad, Hongyu Gong, Francisco Guzm{\'a}n, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, Mary Williamson",
  journal={ArXiv},
  year={2023}
}
```
```
@misc{sturua2024jinaembeddingsv3multilingualembeddingstask,
      title={jina-embeddings-v3: Multilingual Embeddings With Task LoRA}, 
      author={Saba Sturua and Isabelle Mohr and Mohammad Kalim Akram and Michael Günther and Bo Wang and Markus Krimmel and Feng Wang and Georgios Mastrapas and Andreas Koukounas and Andreas Koukounas and Nan Wang and Han Xiao},
      year={2024},
      eprint={2409.10173},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.10173}, 
}

```

