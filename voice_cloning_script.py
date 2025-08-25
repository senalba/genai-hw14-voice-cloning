import argparse
import time
import pandas as pd
import json
from pathlib import Path
from TTS.api import TTS

def load_texts(input_file):
    path = Path(input_file)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.iloc[:,0].tolist()
    elif path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [item["text"] for item in data]
    else:
        raise ValueError("Unsupported file type. Use CSV or JSON.")

def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    # ініціалізація моделі (multi-speaker + voice cloning)
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False)

    texts = load_texts(args.input)

    # простий варіант: одне речення
    tts.tts_to_file(
        text="Це тестове речення для клонування голосу.",
        speaker_wav=args.voice,
        file_path=outdir / "simple.wav"
    )

    # деталізований варіант: багато речень
    for i, sentence in enumerate(texts, 1):
        print(f"Generating {i}/{len(texts)}..")
        t0 = time.time()
        tts.tts_to_file(
            text=sentence,
            speaker_wav=args.voice,
            file_path=outdir / f"out_{i:02d}.wav"
        )
        print(f"Done in {time.time() - t0:.2f} sec.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV or JSON with text")
    parser.add_argument("--voice", required=True, help="Path to sample voice .wav")
    parser.add_argument("--outdir", default="out", help="Output directory")
    args = parser.parse_args()
    main(args)
