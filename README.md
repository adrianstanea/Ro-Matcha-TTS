<div align="center">

# 🍵 Matcha-TTS Romanian: Fast Neural TTS for Romanian Language

### A Romanian language adaptation of [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![huggingface](https://img.shields.io/badge/🤗-Models_on_Hub-yellow)](https://huggingface.co/adrianstanea/Ro-Matcha-TTS)

<p style="text-align: center;">
  <img src="https://shivammehta25.github.io/Matcha-TTS/images/logo.png" height="128"/>
</p>

</div>

> **Romanian Language Adaptation** of the fast conditional flow matching TTS architecture from [ICASSP 2024](https://arxiv.org/abs/2309.03199). This fork extends Matcha-TTS to support high-quality Romanian text-to-speech synthesis using the SWARA 1.0 dataset.

**Official Repository:** [shivammehta25/Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)

## 🚀 Quick Start with Pre-trained Models

**Get started with Romanian TTS in seconds!** Visit our HuggingFace repository for the easiest inference experience:

### 🤗 [adrianstanea/Ro-Matcha-TTS](https://huggingface.co/adrianstanea/Ro-Matcha-TTS)

The HuggingFace repository provides:

- **Pre-trained model checkpoints** ready for download
- **Clean, ready-to-run examples** for Romanian speech generation
- **Step-by-step inference guide** optimized for quick setup
- **Sample texts and outputs** to get you started immediately

The repository leverages this source code for model architecture while providing a streamlined inference experience with all necessary checkpoints and examples.

#### 📊 Available Models

| Speaker | Gender | Model Size | HuggingFace Link                                                                                     |
| ------- | ------ | ---------- | ---------------------------------------------------------------------------------------------------- |
| BAS     | Male   | ~100M      | [adrianstanea/Ro-Matcha-TTS](https://huggingface.co/adrianstanea/Ro-Matcha-TTS/tree/main/models/bas) |
| SGS     | Male   | ~100M      | [adrianstanea/Ro-Matcha-TTS](https://huggingface.co/adrianstanea/Ro-Matcha-TTS/tree/main/models/sgs) |

---

**For Training & Development:** Continue reading below for installation, training, and adaptation instructions.

## 🔬 Romanian Language Adaptation

This adaptation extends Matcha-TTS to Romanian through several key modifications:

### Core Changes

1. **Phonemizer Configuration**: Switched from English to Romanian eSpeak backend
2. **Text Processing**: Custom `romanian_cleaners()` function that preserves Romanian diacritics
3. **Dataset Integration**: SWARA 1.0 Romanian speech corpus with computed statistics
4. **Training Optimizations**: Batch handling improvements for training stability

**Phonetic Processing:**

- Uses eSpeak Romanian phonemizer (`language="ro"`)
- Preserves stress markers and punctuation for natural prosody
- Removes ASCII conversion to maintain Romanian character integrity
- Handles diacritics: ă, â, î, ș, ț

## 🛠️ Installation & Setup

### For Inference Only

```bash
git clone https://github.com/adrianstanea/Matcha-TTS.git
cd Matcha-TTS
conda create -n matcha-tts-ro python=3.10 -y
conda activate matcha-tts-ro
pip install -e .
```

### For Training

```bash
# Additional dependencies for training
pip install pytorch-lightning wandb
# Install phonemizer with eSpeak support
sudo apt-get install espeak espeak-data libespeak1 libespeak-dev
pip install phonemizer
```

## 📖 Usage Guide

### Inference with Pre-trained Models

#### Command Line Interface

```bash
python inference_RO.py \
    --file texts.txt \
    --checkpoint matcha-tts-BAS.ckpt \
    --n_timesteps 50 \
    --length_scale 0.95 \
    --temperature 0.667
```

#### Parameters

- `--n_timesteps`: Number of ODE solver steps (default: 50)
- `--length_scale`: Speaking rate control (default: 0.95)
- `--temperature`: Sampling temperature (default: 0.667)

### Training from Scratch

#### Prepare Dataset

1. Organize your Romanian speech data following SWARA format:

2. Update dataset configuration:

```bash
# Edit configs/data/swara.yaml with your paths
train_filelist_path: path/to/your/metadata_train.csv
valid_filelist_path: path/to/your/metadata_val.csv
```

#### Start Training

```bash
# Train base model
python matcha/train.py experiment=swara trainer.max_epochs=1000

# Fine-tune for specific speaker
python matcha/train.py experiment=swara_bas trainer.max_epochs=1000 ckpt_path=base_model.ckpt
```

#### Using Makefile

```bash
# Available training targets
make train-swara          # Base SWARA training
make finetune-swara_bas   # Fine-tune BAS speaker
make finetune-swara_sgs   # Fine-tune SGS speaker
```

### Fine-tuning for New Speakers

1. **Prepare speaker data** (minimum 10 high-quality samples recommended)
2. **Create dataset configuration**:
```yaml
# configs/data/your_speaker.yaml
defaults:
  - swara
  - _self_

train_filelist_path: resources/filelists/your_speaker_train.csv
valid_filelist_path: resources/filelists/your_speaker_val.csv
```

3. **Start fine-tuning**:
```bash
python matcha/train.py \
    experiment=your_speaker \
    trainer.max_epochs=500 \
    ckpt_path=matcha-base-1000.ckpt
```

## 🔍 For Researchers

### Adaptation Methodology

Our Romanian adaptation follows a systematic approach suitable for other languages:

1. **Phonemizer Selection**: Evaluate available phonemizers (eSpeak, Phonemizer, etc.)
2. **Text Cleaning Pipeline**: Design language-specific text processing
3. **Dataset Integration**: Compute mel-spectrogram statistics for target language
4. **Training Strategy**: Base model training followed by speaker-specific fine-tuning

### Key Technical Decisions

**Text Processing Trade-offs:**
- **Removed ASCII conversion**: Preserves Romanian diacritics essential for pronunciation
- **Disabled abbreviation expansion**: Romanian abbreviations differ significantly from English
- **Maintained stress markers**: Critical for Romanian prosody and naturalness

**Training Optimizations:**
- **Batch size handling**: Drop incomplete batches to ensure training stability
- **Mel statistics**: Language-specific normalization improves convergence
- **Multi-speaker approach**: Base model + speaker fine-tuning for efficiency

## 🌍 Extending to Other Languages

This repository provides a template for adapting Matcha-TTS to new languages. Follow these steps:

### Step 1: Text Processing Setup

```python
# matcha/text/cleaners.py
def your_language_cleaners(text):
    """Pipeline for [YourLanguage] text processing"""
    # Step 1: Decide on ASCII conversion based on script
    if uses_latin_script_with_diacritics:
        # Skip convert_to_ascii(text) like Romanian
        pass
    else:
        text = convert_to_ascii(text)

    # Step 2: Language-specific preprocessing
    text = lowercase(text)

    # Step 3: Handle abbreviations if applicable
    if has_similar_abbreviations_to_english:
        text = expand_abbreviations(text)

    # Step 4: Phonemization
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes

# Configure phonemizer
global_phonemizer = phonemizer.backend.EspeakBackend(
    language="your-lang-code",  # e.g., "de", "fr", "es"
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
)
```

### Step 2: Dataset Configuration

```yaml
# configs/data/your_language.yaml
defaults:
  - ljspeech
  - _self_

name: your_language
train_filelist_path: resources/filelists/your_lang_train.csv
valid_filelist_path: resources/filelists/your_lang_val.csv
cleaners: [your_language_cleaners]

# Compute these statistics from your dataset
data_statistics:
  mel_mean: -X.XXX  # Compute using scripts/compute_data_statistics.py
  mel_std: X.XXX
```

### Step 3: Training Configuration

```yaml
# configs/experiment/your_language.yaml
defaults:
  - override /data: your_language.yaml

tags: ["your_language"]
run_name: your_language_base
```


## 📚 Model Details

### Architecture

Based on Matcha-TTS conditional flow matching architecture:
- **Text Encoder**: Transformer-based with Romanian phoneme embeddings
- **Duration Predictor**: Variance-preserving flow matching
- **Acoustic Model**: Conditional flow matching for mel-spectrogram generation
- **Vocoder**: HiFi-GAN universal vocoder

### Training Details

**Speaker Fine-tuning:**

- **Base checkpoint**: Pre-trained SWARA model (1K epochs)
- **Strategy**: Continued training with speaker-specific data

## 🤝 Contributing & Citation

### Contributing

We welcome contributions for:
- Additional Romanian speakers/datasets
- Other language adaptations
- Training improvements and optimizations
- Bug fixes and documentation

Please open an issue or pull request on GitHub.

### Citation

If you use this Romanian adaptation in your research, please cite:

```bibtex
@ARTICLE{11269795,
  author={Răgman, Teodora and Bogdan Stânea, Adrian and Cucu, Horia and Stan, Adriana},
  journal={IEEE Access},
  title={How Open Is Open TTS? A Practical Evaluation of Open Source TTS Tools},
  year={2025},
  volume={13},
  number={},
  pages={203415-203428},
  keywords={Computer architecture;Training;Text to speech;Spectrogram;Decoding;Computational modeling;Codecs;Predictive models;Acoustics;Low latency communication;Speech synthesis;open tools;evaluation;computational requirements;TTS adaptation;text-to-speech;objective measures;listening test;Romanian},
  doi={10.1109/ACCESS.2025.3637322}
}
```

**Original Matcha-TTS Citation:**
```bibtex
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

### Acknowledgments

- **Original Matcha-TTS**: [Shivam Mehta et al.](https://github.com/shivammehta25/Matcha-TTS)
- **SWARA 1.0 Dataset**: Romanian speech corpus used for training
- **eSpeak**: Phonemization support for Romanian language

### License

This project maintains the same license as the original Matcha-TTS repository.

---

**🎯 Quick Links:**

- [SWARA Dataset](https://speech.utcluj.ro/swarasc)
