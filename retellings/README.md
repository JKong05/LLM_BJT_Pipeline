# Participant Retellings
This directory houses the input audio files for each participant detailing the retelling of each story. Participant data
should be formatted as subfolders (ex. p1, p2, etc.) with the `participant_id` as a unique identifier for each participant. 
Within the subfolders should be the audio retellings (collected from the virtual environment) for that specific participant formatted
as such: `particpant_id`, `story_id`, `sensory_modality`,`congruence`.wav. Depending on how many stories each participant is exposed to, the
number of audio files will vary. Ensure that these naming conventions are kept consistent throughout to ensure consistency of the
pipeline.
<br />
participant_metadata.csv should be filled out with corresponding `participant_id` and `age` of the participant.


> [!NOTE]
> Order of modality for each story does not matter (i.e. story1 can be audio, visual, or audiovisual)

```
# example folder structure

p1/                                            # example participant_1
│   ├── p1_story1_audio_correct.wav                    # example audio file for p1 retelling of story1 (audio-only) with a correct environment exposure
│   ├── p1_story2_visual_correct.wav
│   ├── p1_story3_audiovisual_correct.wav
│   ├── p1_story4_audio_incorrect.wav
│   ├── p1_story5_audiovisual_incorrect.wav
│   └── p1_story6_visual_incorrect.wav
```
