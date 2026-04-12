# Hindi Speech-to-Text Fine-Tuning — Research Document

**Name:** Aneerban  
**Position:** AI Engineer Intern  
**Company:** MySivi  

---

📓 **Notebook Link:** [Add your public Colab link here]  
🎥 **Video Demo Link:** [Add your public video link here]  

---

## 1. Introduction & Objective

The objective of this project is to fine-tune an open-source Speech-to-Text (STT) model to improve its transcription accuracy for **Hindi language audio**. This document details the end-to-end research process — from model and dataset selection to fine-tuning methodology and evaluation results.

Hindi presents unique challenges for automatic speech recognition (ASR):
- **Devanagari script** with complex conjunct consonants (*संयुक्ताक्षर*) and vowel matras
- High **dialectal variation** across regions (Khariboli, Braj, Awadhi, etc.)
- **Code-switching** with English in everyday speech
- Limited availability of high-quality, large-scale labeled speech data compared to English
- Phonetic complexity — Hindi has aspirated and unaspirated consonant pairs that sound similar but carry different meaning

---

## 2. Model Selection

### Models Considered

| Model | Architecture | Parameters | Hindi Support | Free GPU Feasible | Pros | Cons |
|-------|-------------|-----------|---------------|-------------------|------|------|
| Whisper Tiny | Encoder-Decoder | 39M | Basic | ✅ Easily | Very fast training | Low baseline accuracy |
| **Whisper Small** | **Encoder-Decoder** | **244M** | **Good** | **✅ Yes** | **Best balance of quality & feasibility** | **Moderate training time** |
| Whisper Medium | Encoder-Decoder | 769M | Better | ⚠️ Tight | Higher baseline | OOM risk on free T4 |
| Whisper Large-v3 | Encoder-Decoder | 1.55B | Best | ❌ Too large | State-of-the-art | Impossible on free GPU |
| wav2vec2-large-xlsr | CTC | 300M | Needs fine-tune | ✅ Yes | Fast inference | No language model; struggles with Hindi morphology |
| MMS-1B (Meta) | CTC | 1B | Good | ⚠️ Tight | 1000+ languages | Too large for free GPU; CTC limitations |

### Why Whisper Small?

I selected **OpenAI Whisper Small** (`openai/whisper-small`, 244M parameters) for the following reasons:

1. **Pre-trained multilingual knowledge**: Whisper was trained on **680,000 hours** of labeled audio spanning 99 languages, including Hindi. Fine-tuning leverages this existing knowledge, adapting an already capable model rather than training from scratch. This means even with limited Hindi-specific data, the model can generalize well.

2. **Encoder-Decoder architecture superiority**: Unlike CTC-based models (wav2vec2, MMS), Whisper uses a sequence-to-sequence architecture with attention. This is critical for Hindi because:
   - It can model **complex output dependencies** (e.g., correct placement of matras based on context)
   - It handles **word boundaries** more naturally without needing an external language model
   - It's more robust to **varying speaking speeds** and accents

3. **Memory-efficient for free GPUs**: At 244M parameters, Whisper Small fits comfortably on a T4 GPU (16GB VRAM) available on both Google Colab and Kaggle. With FP16 mixed-precision training and gradient checkpointing, the model uses approximately 8-10GB VRAM during training, leaving headroom for batch processing.

4. **Balanced performance-cost tradeoff**: While Whisper Medium (769M) and Large-v3 (1.55B) offer better baseline performance, the Small variant provides a strong starting point that can be **meaningfully improved** through fine-tuning within the constraints of free compute resources. As my results show, the Small model achieved a **60.8% relative WER improvement** after fine-tuning — demonstrating that model size is not the only factor; targeted fine-tuning on domain-specific data matters greatly.

5. **Robust preprocessing pipeline**: Whisper's built-in feature extractor handles audio normalization, resampling (to 16kHz), and 80-channel log-Mel spectrogram extraction consistently, reducing preprocessing complexity and potential sources of error.

---

## 3. Dataset

### Dataset: Google FLEURS (Hindi)

**Source:** [HuggingFace — google/fleurs](https://huggingface.co/datasets/google/fleurs)  
**License:** CC-BY-4.0 (fully open source, no access restrictions)  
**Configuration:** `hi_in` (Hindi — India)

| Split | Samples | Purpose |
|-------|---------|---------|
| Train | ~2,000 | Model fine-tuning |
| Validation | ~400 | Hyperparameter tuning / checkpoint selection |
| **Test** | **418** | **Final evaluation only (never seen during training)** |

### Why FLEURS?

1. **Freely available**: Unlike Mozilla Common Voice (which requires gated access and terms acceptance), FLEURS is fully open under the CC-BY-4.0 license with no access restrictions — critical for reproducibility.

2. **High quality**: FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech) is professionally recorded speech data, part of Google's FLoRes-101 benchmark covering 102 languages. The recordings are clean, well-segmented, and accurately transcribed.

3. **Sufficient for fine-tuning**: With approximately **12 hours of Hindi audio supervision**, FLEURS provides enough data for meaningful fine-tuning of Whisper Small. My results validate this — achieving a 41.73 percentage point WER reduction with just this dataset.

4. **Pre-defined splits**: Standard train/validation/test splits prevent data leakage and ensure reproducible, fair evaluation. The test set of 418 samples was **strictly held out** and never used during training.

5. **Easy integration**: Available directly via HuggingFace `datasets` library with a single line of code: `load_dataset("google/fleurs", "hi_in")`.

### Data Preprocessing Pipeline

```
Raw Audio (48kHz, various formats)
  → Resample to 16kHz (Whisper's requirement)
  → Whisper Feature Extractor
  → 80-channel Log-Mel Spectrogram (30-second windows)

Hindi Text (Devanagari)
  → Whisper Tokenizer (with language="hi", task="transcribe")
  → Token IDs with special language and task tokens
```

**Key preprocessing decisions:**
- **Resampling to 16kHz**: Standard for Whisper's architecture; reduces memory footprint without meaningful quality loss for speech recognition. Human speech information is primarily below 8kHz, well within the Nyquist limit at 16kHz sampling.
- **No text normalization**: Hindi Devanagari text is used as-is to preserve original diacritical marks, conjuncts, and chandrabindu (ँ) — critical for accurate evaluation.
- **Column pruning**: Removed all metadata columns (age, gender, speaker ID, etc.) to keep only `audio` and `transcription`, reducing memory usage during preprocessing.

---

## 4. Fine-Tuning Approach

### Training Configuration

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Base Model | `openai/whisper-small` | Best accuracy within free GPU constraints |
| Learning Rate | 1e-5 | Conservative rate to prevent catastrophic forgetting of pre-trained multilingual knowledge |
| Warmup Steps | 250 | Gradual LR increase over ~12% of training for stability |
| Batch Size | 16 | Maximizes T4 GPU utilization; larger batches provide more stable gradients |
| Max Steps | 2,000 | ~1.5 hours on T4; sufficient for convergence on FLEURS Hindi |
| Precision | FP16 (mixed) | Halves memory usage with negligible quality impact; enables larger batch sizes |
| Gradient Checkpointing | Enabled | Trades 20-30% compute overhead for ~40% memory savings |
| Optimizer | AdamW (default) | Standard for transformer fine-tuning; handles sparse gradients well |
| Evaluation Frequency | Every 400 steps | Frequent monitoring for early stopping and best model selection |
| Best Model Selection | Lowest WER on validation set | Optimizes directly for the primary evaluation metric |
| Checkpoints Saved | 3 (best) | Prevents disk overflow on Colab while preserving recovery options |

### Architecture

The fine-tuning updates **all parameters** of the Whisper Small model (full fine-tuning, not adapter-based):
- **Encoder** (12 transformer layers): Adapts acoustic feature extraction to Hindi phonetic characteristics — aspirated consonants, nasalized vowels, and retroflex sounds unique to Indian languages
- **Decoder** (12 transformer layers): Adapts text generation to Hindi Devanagari output patterns, including proper matra placement, conjunct formation, and sentence structure

### Custom Data Collator

I implemented a custom `DataCollatorSpeechSeq2SeqWithPadding` class that:
- Pads input spectrograms to the maximum length in each batch
- Pads label sequences with `-100` tokens (ignored by the cross-entropy loss function)
- Removes the BOS token when Whisper handles it internally
- This ensures efficient batching without losing information from variable-length audio clips

### Platform & Resources
- **Google Colab** (Free tier)
- **GPU:** NVIDIA T4 (16GB VRAM)
- **Training Time:** ~1.5 hours
- **Total experiment time:** ~3 hours (including data download, preprocessing, and evaluation)

---

## 5. Evaluation

### Metrics

- **WER (Word Error Rate)**: 
  - Formula: `(Substitutions + Insertions + Deletions) / Total Reference Words × 100`
  - Measures word-level transcription accuracy
  - Primary metric for ASR model comparison
  - A word is counted as wrong even if only one character differs

- **CER (Character Error Rate)**:
  - Formula: `(Substitutions + Insertions + Deletions) / Total Reference Characters × 100`
  - Character-level accuracy, more granular than WER
  - Especially important for Hindi where Devanagari characters carry individual phonetic meaning
  - A model with high WER but low CER indicates it understands the language phonetics but makes minor errors

### Results

| Model | WER (%) | CER (%) |
|-------|---------|---------|
| Whisper Small (Base) | 68.61 | 34.43 |
| Whisper Small (Fine-tuned) | **26.87** | **10.41** |
| **Absolute Improvement** | **↓ 41.73** | **↓ 24.02** |
| **Relative Improvement** | **↓ 60.8%** | **↓ 69.8%** |

### Key Observations

1. **Dramatic WER reduction**: The base Whisper Small model had a WER of 68.61% on the Hindi test set — essentially getting more than 2 out of every 3 words wrong. After fine-tuning, WER dropped to 26.87%, meaning the model now correctly transcribes approximately **3 out of 4 words**.

2. **Even stronger CER improvement**: CER dropped from 34.43% to 10.41% — a 69.8% relative improvement. This indicates that even when the fine-tuned model makes word-level errors, the character-level predictions are very close to the ground truth. The model has learned the phonetic structure of Hindi well.

3. **Small dataset, large impact**: These results were achieved with only the FLEURS Hindi dataset (~12 hours of audio), demonstrating that targeted fine-tuning on clean, domain-specific data can dramatically improve performance even without massive datasets.

4. **Cost-effective training**: The entire fine-tuning process took approximately 1.5 hours on a free T4 GPU — showing that significant ASR improvements are achievable without expensive compute infrastructure.

### Error Analysis

After examining sample predictions from the fine-tuned model, I observed the following patterns:

**Types of errors that improved significantly:**
- **Script accuracy**: The base model frequently produced garbled or incorrect Devanagari characters; the fine-tuned model produces well-formed Hindi text
- **Word boundaries**: The base model often merged or split words incorrectly; fine-tuning corrected most word segmentation errors
- **Common vocabulary**: High-frequency Hindi words are now transcribed almost perfectly

**Remaining error patterns:**
- **Rare/specialized vocabulary**: Less common Hindi words occasionally have character substitutions
- **Homophone confusion**: Words that sound similar but are spelled differently remain challenging
- **Sentence endings**: Minor errors at the end of some utterances where audio quality tapers off

---

## 6. Challenges & Solutions

| Challenge | Solution | Impact |
|-----------|----------|--------|
| GPU memory constraints on free Colab (16GB T4) | Used FP16 mixed precision + gradient checkpointing + Whisper Small (244M) | Enabled training with batch size 16 without OOM errors |
| Colab session timeout risk | Saved checkpoints every 400 steps; kept training to ~1.5 hours | Never lost progress during training |
| Common Voice dataset requires gated access | Switched to Google FLEURS (CC-BY-4.0, fully open) | No access barriers; fully reproducible |
| `datasets` library version incompatibility | Pinned `datasets<3.0.0` to support FLEURS loading script | Resolved `RuntimeError: Dataset scripts are no longer supported` |
| Input tensor type mismatch (float64 vs float32) | Explicit `dtype=torch.float32` casting + `model.float()` | Fixed `RuntimeError` during inference |
| Newer `transformers` API deprecations | `evaluation_strategy` → `eval_strategy`; `tokenizer` → `processing_class` | Ensured compatibility with latest transformers |

---

## 7. Future Improvements

Given more time and resources, the following improvements could be explored:

1. **Larger model**: Fine-tune Whisper Medium (769M) or Large-v3 (1.55B) on an A100 GPU for a higher baseline accuracy and potentially even greater improvements after fine-tuning.

2. **LoRA / Parameter-Efficient Fine-Tuning (PEFT)**: Use Low-Rank Adaptation to fine-tune only a small fraction (~1-2%) of parameters. This would enable fine-tuning Whisper Large-v3 on a T4 GPU while maintaining comparable accuracy.

3. **Data augmentation**: Apply SpecAugment (time/frequency masking), speed perturbation (0.9x-1.1x), and background noise injection to improve robustness to real-world audio conditions.

4. **Larger & diverse datasets**: Combine FLEURS with IndicVoices, Kathbath, and crowd-sourced Hindi datasets for broader dialect and accent coverage.

5. **Language model fusion**: Integrate a Hindi language model (e.g., IndicBERT) for post-processing to correct common transcription errors using contextual understanding.

6. **Code-switching handling**: Fine-tune on Hindi-English mixed speech data (Hinglish) for real-world scenarios where speakers frequently switch between languages.

7. **Quantization for deployment**: Apply INT8 or INT4 quantization for faster inference without significant accuracy loss, enabling real-time transcription on edge devices.

8. **Streaming support**: Implement chunked audio processing for real-time transcription of live Hindi audio streams.

---

## 8. Assumptions Made

As mentioned in the assignment, the following assumptions were made:

1. **Dataset choice**: Chose FLEURS over Common Voice due to access restrictions on Common Voice. FLEURS provides sufficient Hindi data for meaningful fine-tuning.
2. **Model size**: Chose Whisper Small over larger variants to ensure feasibility on free GPU resources while still achieving strong results.
3. **Training duration**: Limited to 2,000 steps (~1.5 hours) as a practical constraint of free Colab sessions. Longer training may yield further improvements.
4. **Evaluation scope**: Evaluated on the FLEURS test set only. Real-world performance may vary with different accents, recording conditions, and speaking styles.

---

## 9. References

1. Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." *OpenAI*. [Paper](https://arxiv.org/abs/2212.04356)
2. Google FLEURS Dataset: [https://huggingface.co/datasets/google/fleurs](https://huggingface.co/datasets/google/fleurs)
3. Conneau, A., et al. (2022). "FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech." *Google*. [Paper](https://arxiv.org/abs/2205.12446)
4. HuggingFace Whisper Fine-Tuning Guide: [https://huggingface.co/blog/fine-tune-whisper](https://huggingface.co/blog/fine-tune-whisper)
5. Whisper Model Card: [https://huggingface.co/openai/whisper-small](https://huggingface.co/openai/whisper-small)

---

*Submitted by: Aneerban*  
*Date: [Fill in submission date]*  
*Email: hiring@mysivi.ai*
