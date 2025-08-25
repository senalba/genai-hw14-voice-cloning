# Homework 16 — Voice Cloning Metrics Evaluation

## 📄 Мета
Дослідити можливості відкритих AI-моделей для клонування голосу, зосереджуючись на їхньому застосуванні в задачах перетворення тексту в мовлення (TTS).

---

## 📂 Project Structure
```
genai-hw14-voice-cloning/
├── environment.yml
├── voice_cloning_script.py
├── data/
│   ├── input.csv
│   ├── input.json
│   └── sample_voice.wav
├── out/
│   └── ... generated audio files ...
```

---

## ⚙️ Setup
```bash
cd genai-hw14-voice-cloning
mamba env create -f environment.yml
```

---

## ▶️ Usage
Run with CSV or JSON input:

```bash
# Example with CSV
python voice_cloning_script.py --input data/input.csv --voice data/sample_voice.wav --outdir out

# Example with JSON
python voice_cloning_script.py --input data/input.json --voice data/sample_voice.wav --outdir out

# Example with Text
python voice_cloning_script.py --text "This is a sample text to be spoken" --voice data/sample_voice.wav --outdir out --prefix "text_"
```

- **--input** : Path to CSV or JSON with text  
- **--voice** : Path to sample voice `.wav` for cloning  
- **--outdir** : Output folder (default: `out`)  
- **--prefix** : Prefix for output files (default: '') 

---

## 🧪 Evaluation Criteria
1. **Similarity** – Does the generated voice resemble the source speaker?  
2. **Naturalness** – Is the speech fluent and free of robotic artifacts?  
3. **Intelligibility** – Are words clear and easy to understand?  

---

## ✅ Example
- Input:  
  `"Artificial intelligence can now replicate human voices with surprising accuracy."`

- Output:  
  `out/out_02.wav` – cloned voice reading the sentence.

## Preprocessing
```bash
python -m demucs --two-stems=vocals -d cpu -n htdemucs_ft data/trump_short.wav

ffmpeg -y -i separated/htdemucs_ft/trump_short/vocals.wav -ar 22050 -ac 1 data/sample_voice.wav
```
---

## 📊 Висновки

### Оцінка:
- Схожість: низька — тембр і манера лише частково збігаються; zero-shot XTTS не «влучає» в голос.
- Природність: добра — інтонації й паузи ок, інколи чути дрібні артефакти на складних словах.
- Розбірливість: висока — текст зрозумілий, слів не «з’їдає».


Деталізований підхід (batch із різними фразами) лишається найінформативнішим: він виразно показує слабкі місця та стабілізує оцінку.  
Експерименти підтвердили залежність якості від семпла: короткий шумний — погано; короткий очищений — помітно краще; довший чистий найкраща природність і розбірливість, але **схожість далека від ідеалу**.  
Zero-shot XTTS-v2 добре тримає інтонації та дикцію, однак тембр копіює лише частково; очищення (на кшталт Demucs) допомагає, але не замінює довший чистий матеріал.  
Працювати з моделлю зручно (один виклик), проте залежності вимогливі: сумісні PyTorch/torchaudio та `numpy<2` — обов’язково.  
Сильні сторони open-source: локальний запуск, контроль над даними, без платних API, можливість тюнінгу та відтворюваність.  
Критичні фактори: якість та тривалість семпла, збіг мови/акценту й стилю з таргет-текстом, стабільна гучність, мінімум шуму.  
Для «1:1» схожості варто планувати донавчання (fine-tuning) на власних даних або збільшення/варіативність чистих семплів; для озвучки матеріалів поточний підхід уже придатний.
