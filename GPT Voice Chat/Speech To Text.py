import torch
from transformers import Speech2TextProcessor, Speech2TextForConditionalGeneration
from datasets import load_dataset
import soundfile as sf
import librosa

model = Speech2TextForConditionalGeneration.from_pretrained("facebook/s2t-medium-librispeech-asr")
processor = Speech2TextProcessor.from_pretrained("facebook/s2t-medium-librispeech-asr")

##def map_to_array(batch):
##    audio_path = batch["file"]
##    speech, _ = sf.read(audio_path)
##    input_features = processor(
##        speech,
##        sampling_rate=16_000,
##        return_tensors="pt"
##    ).input_features
##    return input_features

audio_path = "./Test Audio 1.wav"
audio, original_sampling_rate = librosa.load(audio_path, sr=None)
##print(original_sampling_rate)
audio_resampled = librosa.resample(audio, orig_sr=original_sampling_rate, target_sr=16000)

##output_file_path = "./resampled_audio.wav"
##sf.write(output_file_path, audio_resampled, 16000)  # 16000 is the target sampling rate
##print(f"Resampled audio saved to: {output_file_path}")

##audio_path = "./resampled_audio.wav"
##dummy_batch = {"file": audio_path}
##input_features = map_to_array(dummy_batch)

input_features = processor(
    audio_resampled,
    sampling_rate=16_000,
    return_tensors="pt"
    ).input_features

generated_ids = model.generate(input_features=input_features)
##generated_ids = model.generate(input_features=input_features , num_beams=4, length_penalty=0.6, temperature=0.8)

transcription = processor.batch_decode(generated_ids)
print(transcription)
