# Analytical Pipeline
Code for LLM_BJT analytical pipeline (Fall24 ~ Spr24)

### Overview
The process of developing an analytical pipeline to compare participant retellings of stories involved a two-pronged approach: extraction of expressive and semantic features and the creation of representational embeddings. Prosodic features are elements of speech that contribute to the accent, rhythm, stress, intonation, tone, and pitch of the spoken language. Semantic features refer to specific characteristics of linguistics that constitute the meaning of the words that are being used. The combination of these two concepts help explain both the way in which an individual speaks and what their words mean. Therefore, a combination of multi-modal models were used to create embeddings.

### Purpose
The purpose of this pipeline is to extract and create representational vectors of a participant's retelling to a narrative story using **Meta's SeamlessExpressive model and Meta's SONAR semantic model**. A typical use case is as follows:
  1. Gather audio retellings of stories in .wav format
  2. Input raw audio into SeamlessExpressive model and generate expressive audio embeddings
  3. Take raw audio and translate to text using ChatGPT Whisper model
     - Input text translations into SONAR semantic encoder and generate semantic text embeddings 
  4. Concatenate embeddings from longer recordings into representational vectors.
  5. Perform cosine similarity between participant vectors and create representational dissimilarity matrix

### Goals
1. Determine trends and commonalities in and between age group linguistics by evaluating similarities in prosodic and semantic features of speech.
2. Evaluate and establish potential associations and relationships between age and retelling fidelity.
3. Develop foundational code infrastructure for refinement and expansion.

# Folder Structure
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

# Abstract
The ability for humans to perceive and comprehend stimuli of varying sensory modalities and levels of complexities has long been a central question in both neuroscience and developmental linguistics. Traditionally, research in these fields have focused primarily on how individuals of varying ages acquire and integrate information to support language and cognitive development. Recent advances in computational modeling—in particular, neural alignment models—have opened new avenues for probing the mechanisms underlying multisensory integration, attention, and narrative comprehension. However, the stimuli used to evaluate such models are often constrained by unnatural, two-dimensional designs, limiting their ecological validity. We present a novel paradigm in which participants experience narrative stories in auditory, visual, or audiovisual modalities within a three-dimensional virtual reality (VR) environment. Immersion is manipulated by varying the congruence between the narrative and its surrounding virtual context, allowing us to examine the effects of environmental mismatch on multisensory encoding. By analyzing participant language output using the Seamless Communication model infrastructure from Meta, we can extract semantic and expressive features of speech to assess patterns in linguistic behavior. Our work offers new insights into the dynamic intersection of sensory context and language processing, highlighting the critical role of immersive congruency in shaping language and informing future models of language development in immersive settings.


# Quick Start
> [!NOTE]
> SeamlessExpressive requires the [fairseq2](https://github.com/facebookresearch/fairseq2) library to run. It specifically provides the infrastructure for the [wav2vec 2.0](https://github.com/facebookresearch/fairseq2/tree/main/src/fairseq2/models/wav2vec2) model that is used for speech-to-text translations and embedding generation. Depending on your operating system, installation will vary.


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

