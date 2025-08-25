from TTS.api import TTS

tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)

tts.tts_to_file(
    text="This is a test of cloned voice generation.",
    speaker_wav="data/trump.wav",
    language="en",
    file_path="out/test.wav",
)
