# Participant Retellings
This directory houses the input audio files for each participant detailing the retelling of each story. Participant data
should be formatted as subfolders (ex. p1, p2, etc.) with the `participant_id` as a unique identifier for each participant. 
Within the subfolders should be the audio retellings (collected from the virtual environment) for that specific participant formatted
as such: `particpant_id`, `story_id`, `sensory_modality`.wav. Depending on how many stories each participant is exposed to, the
number of audio files will vary. Ensure that these naming conventions are kept consistent throughout to ensure consistency of the
pipeline. 

```
# example folder structure
p1/                                            # example participant_1
│   ├── p1_story1_audio.wav         
│   ├── p1_story4_visual.wav
│   ├── p1_story5_audiovisual.wav
│   ├── p1_story9_audio.wav
│   ├── p1_story9_audiovisual.wav
│   └── p1_story9_visual.wav
```
