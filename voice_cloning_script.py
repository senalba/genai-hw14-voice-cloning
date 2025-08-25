# voice_cloning_script.py
import argparse
import json
from pathlib import Path

import pandas as pd
from TTS.api import TTS
import time


def load_texts(input_file: str) -> list[str]:
    path = Path(input_file)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
        return df.iloc[:, 0].astype(str).tolist()
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [str(item["text"]) for item in data]
    raise ValueError("Unsupported file type. Use CSV or JSON.")


def main(args):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # XTTS-v2: zero-shot cloning, multilingual
    tts = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False
    )

    # Simple case: one sentence from --text
    if args.text:
        tts.tts_to_file(
            text=args.text,
            speaker_wav=args.voice,
            language=args.lang,
            file_path=outdir / "simple.wav",
        )

    # Detailed case: batch from CSV/JSON (if provided)
    if args.input:
        texts = load_texts(args.input)
        for i, sentence in enumerate(texts, 1):
            print(f"Generating {i}/{len(texts)}..")
            t0 = time.time()
            tts.tts_to_file(
                text=sentence,
                speaker_wav=args.voice,
                language=args.lang,
                file_path=outdir / f"{args.prefix}out_{i:02d}.wav",
            )
            print(f"Done in {time.time() - t0:.2f} sec.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="CSV or JSON with a 'text' column/field")
    parser.add_argument("--text", help="Single sentence for the simple case")
    parser.add_argument("--voice", required=True, help="Path to sample voice .wav")
    parser.add_argument("--lang", default="en", help="Language code (e.g., en, uk, de)")
    parser.add_argument("--outdir", default="out", help="Output directory")
    parser.add_argument("--prefix", default="", help="Prefix for output files")
    args = parser.parse_args()
    main(args)
