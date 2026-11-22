# exercise-01 — Whisper Pipeline With Language Detection

> Reference solution. The exercise brief lives in the paired `nlp-engineer-learning` repo at
> `lessons/mod-109-speech-text-interface/exercises/exercise-01-whisper-pipeline-with-language-detection.md`.
> The target learning outcomes implied by that filename: build an **end-to-end Whisper ASR pipeline** that **auto-detects the spoken language** from the first audio window, routes to the right decoding config for that language, handles **long-form audio** past Whisper's 30-second window, and reports **word error rate** against a reference transcript the way the Whisper paper does — same reference normaliser, same case/punctuation folding.

## 1. Solution overview

Whisper (Radford et al. 2022, OpenAI) is a 30 s window encoder-decoder trained on ~680 k h of weakly-supervised multilingual speech. Out of the box it works; done badly it hallucinates, loops, and silently mislabels language. Four subtleties catch every first-time pipeline:

| Failure mode | What goes wrong | How to fix |
|---|---|---|
| **Wrong language, confidently decoded** | Whisper's `language=None` default calls `detect_language` on the first 30 s. On music, silence, or a short preamble ("Hi — welcome back —") the detector picks a wrong language with >0.9 confidence; the decoder then fluently *translates* the rest into that language | Call `detect_language` on a VAD-filtered window with speech, keep the top-k and the confidence, fall back to a configured default when the margin is small, and log every decision |
| **30-second window misuse** | Whisper's audio encoder is fixed at 30 s log-mel frames. Feeding a 10-minute file to the Python `whisper.transcribe()` works because it slides a window internally, but feeding chunks to `.generate()` directly without the chunker drops everything past 30 s | Use `transcribe()` for file-level; for streaming/batched, use `transformers`' `AutomaticSpeechRecognitionPipeline(chunk_length_s=30, stride_length_s=(5, 5))` with 5 s overlap on each side |
| **Silence hallucination** | On silent or music-only segments, Whisper generates plausible text ("Thanks for watching!", "♪ music ♪", or paper-training-set phrases). The decoder never stops because no EOS was seen; the segment then poisons later chunks via the `prompt` mechanism | Run a VAD pass (Silero or `webrtcvad`) first, skip non-speech segments, and set `condition_on_previous_text=False` so a bad chunk cannot cascade |
| **WER on mismatched normalisation** | Reporting WER of raw hypothesis vs raw reference inflates error by 10–30 % on real data (`"25"` vs `"twenty-five"`, `"i.e."` vs `"ie"`, casing, trailing punctuation) and is not comparable to the Whisper paper's numbers | Apply Whisper's `EnglishTextNormalizer` / `BasicTextNormalizer` to both hyp and ref before `jiwer.wer`; this is the exact protocol OpenAI uses in the paper and in the released eval scripts |

The deliverable is a transcription pipeline with six components:

1. **Audio loader + VAD** — resample to 16 kHz mono, run Silero VAD, strip leading/trailing silence, emit speech segments.
2. **Language detector** — mel-spec the first VAD-confirmed speech window; call `model.detect_language()`; keep top-3 + confidence; apply a configurable fallback rule.
3. **Chunked decoder** — Hugging Face `pipeline("automatic-speech-recognition", chunk_length_s=30, stride_length_s=(5, 5))` with `return_timestamps="word"`.
4. **Generation config** — pinned: `temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0)` fallback ladder, `compression_ratio_threshold=2.4`, `logprob_threshold=-1.0`, `no_speech_threshold=0.6`, `condition_on_previous_text=False`.
5. **WER evaluator** — Whisper's own `EnglishTextNormalizer` / `BasicTextNormalizer` + `jiwer`; signature (normaliser version, language) recorded in output JSON.
6. **Language confidence diagnostics** — a logged-but-not-blocking classifier that flags transcripts whose detected language disagrees with a second-opinion fastText `lid.176` run on the final text.

Reference datasets (used to validate, not retrain):
- **LibriSpeech test-clean / test-other** (Panayotov et al. 2015) — English read speech; the WER anchor used in the Whisper paper.
- **Common Voice** (Ardila et al. 2020) — crowd-sourced multilingual; the paper's main multilingual benchmark.
- **FLEURS** (Conneau et al. 2022) — 102-language ASR + speech translation, 10 s clips, aligned with FLORES-101 text; used for low-resource language coverage checks.

Reference checkpoints — pick by latency / accuracy / cost:
- **`openai/whisper-large-v3`** — 1.55 B params; the current flagship; best WER on most languages, needs GPU for real-time.
- **`openai/whisper-large-v2`** — the paper's reported checkpoint; still competitive; slightly worse than v3 on noisy audio.
- **`openai/whisper-medium`** — 769 M; good GPU-latency trade; the worked example uses this.
- **`openai/whisper-small`** / **`openai/whisper-base`** / **`openai/whisper-tiny`** — CPU-viable; English-only variants (`.en`) drop ~30 % WER vs the multilingual ones on English in the tiny/base sizes.

## 2. Worked implementation

### 2.1 Project layout

```
asr_bake/
  audio.py           # load, resample, VAD
  langid.py          # Whisper detect_language + fastText second opinion
  pipeline.py        # chunked ASR via transformers pipeline
  generation.py      # pinned GenerationConfig + temperature fallback
  metrics.py         # Whisper normaliser + jiwer WER
  run.py             # CLI
  tests/
    test_vad.py
    test_langid.py
    test_wer.py
    test_chunking.py
  requirements.txt
```

### 2.2 Audio loading + VAD — do not transcribe silence

```python
# audio.py
import numpy as np
import torch
import torchaudio

TARGET_SR = 16_000


def load_mono_16k(path: str) -> np.ndarray:
    wav, sr = torchaudio.load(path)             # (C, T)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)     # mono mixdown
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav.squeeze(0).numpy().astype(np.float32)


def vad_segments(wav: np.ndarray, sr: int = TARGET_SR) -> list[tuple[float, float]]:
    """Return list of (start_s, end_s) speech segments using Silero VAD."""
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", trust_repo=True
    )
    get_speech_timestamps, _, _, _, _ = utils
    ts = get_speech_timestamps(
        torch.from_numpy(wav), model,
        sampling_rate=sr,
        threshold=0.5,
        min_speech_duration_ms=250,
        min_silence_duration_ms=100,
    )
    return [(t["start"] / sr, t["end"] / sr) for t in ts]
```

Points worth defending in review:

- **16 kHz mono is mandatory.** Whisper's log-mel front-end is trained on 16 kHz; feeding 44.1 kHz gives *audibly correct* mel features that are shifted in frequency and drops WER by several points. Resample explicitly.
- **VAD is upstream of language detection.** `detect_language` on a silent / music-only window is a coin flip; feed it the first *speech* window.
- **`min_speech_duration_ms=250`.** Below ~0.25 s most "speech" detected by VAD is noise bursts, mic bumps, lip smacks. Keeps the first-window LID honest.

### 2.3 Language detection — top-k, confidence, fallback

```python
# langid.py
from __future__ import annotations
import whisper
import numpy as np

AUDIO_WINDOW_S = 30.0  # Whisper's native input length


def detect_language(
    model: whisper.Whisper,
    wav: np.ndarray,
    sr: int = 16_000,
    top_k: int = 3,
) -> tuple[str, float, list[tuple[str, float]]]:
    """
    Returns (lang_code, confidence, top_k_probs). Uses the first 30 s of audio,
    which the Whisper paper and the openai-whisper CLI call `detect_language` on.
    """
    audio = whisper.pad_or_trim(wav, length=int(AUDIO_WINDOW_S * sr))
    mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
    _, probs = model.detect_language(mel)
    # probs is a dict[lang_code -> prob]
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top = items[:top_k]
    return top[0][0], top[0][1], top


def apply_fallback(
    detected: str,
    confidence: float,
    default_lang: str = "en",
    min_confidence: float = 0.6,
) -> tuple[str, str]:
    """If detection is low-confidence, fall back to the configured default and
    log the override. Returns (used_lang, reason)."""
    if confidence < min_confidence:
        return default_lang, f"fallback:{detected}@{confidence:.2f}<{min_confidence}"
    return detected, "detected"
```

- **Run `detect_language` on one VAD-confirmed 30 s window, not on the whole file.** Running it on the full waveform then averaging is both slower and biases toward the majority language in multilingual clips.
- **Log the top-3 probs.** When the user opens a bug "it transcribed French as English", `(en 0.42, fr 0.39, de 0.08)` tells the triager "this clip is near the LID decision boundary"; `(en 0.98)` tells them "the audio is almost certainly not French".
- **`min_confidence=0.6`.** Below this, override to a configured default and log the override. Whisper's LID is well calibrated on clean speech but overconfident on noise; the hard threshold is cheap insurance.

### 2.4 Chunked long-form decoding

```python
# pipeline.py
from transformers import pipeline

def build_asr_pipeline(model_id: str = "openai/whisper-medium", device: int = 0):
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,
        chunk_length_s=30,
        stride_length_s=(5, 5),   # 5 s overlap on each side
        return_timestamps="word",
        device=device,
        torch_dtype="auto",
    )


def transcribe_long(asr, wav, language: str, task: str = "transcribe") -> dict:
    return asr(
        {"array": wav, "sampling_rate": 16_000},
        generate_kwargs={
            "language": language,
            "task": task,                      # "transcribe" or "translate"
            "condition_on_previous_text": False,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "num_beams": 5,
        },
    )
```

- **`chunk_length_s=30, stride_length_s=(5, 5)`.** The Hugging Face chunker windows at 30 s with 5 s overlap on each side; it then performs longest-common-substring stitching on the overlap to remove duplicates. 10 s stride is cleaner but slower; 5 s is the paper's default.
- **`task="transcribe"` vs `"translate"`.** Whisper's second task is English-only translation ("decode any language *into* English"). Setting this to `"translate"` silently changes the output language to English regardless of `language=`; confusing if you don't expect it.
- **`condition_on_previous_text=False`.** The default (True) prepends the previous chunk's text as a prompt, which empirically improves consistency on clean long-form audio — but when one chunk hallucinates, the hallucination propagates for the rest of the file. Set to False for robustness unless you have measured the trade-off on your specific data.
- **Temperature fallback ladder.** `(0.0, 0.2, ..., 1.0)` is the Whisper paper's exact fallback schedule: if `compression_ratio > 2.4` (compression ratio of the generated text: a proxy for repetition loops) or `avg_logprob < -1.0` (the model is unconfident), re-decode at the next temperature. This is Whisper's built-in hallucination escape hatch; do not disable it.
- **`no_speech_threshold=0.6`.** The Whisper decoder emits a `<|nospeech|>` probability; if above 0.6 and logprob is low, the chunk is marked non-speech and skipped. This is what prevents the "♪ music ♪" hallucinations.

### 2.5 Word-level timestamps

```python
# pipeline.py (fragment)
result = transcribe_long(asr, wav, language=lang)
for chunk in result["chunks"]:
    start, end = chunk["timestamp"]            # (float, float) seconds
    text = chunk["text"]
    # chunks are word-level when return_timestamps="word"
```

Whisper derives word timestamps from the cross-attention peaks between decoder tokens and encoder frames (the DTW-alignment trick introduced in the paper's §C). They are *approximate* — noisy to ~100–200 ms depending on content — but usable for captioning. For hard-alignment work (e.g. karaoke, forced align) prefer WhisperX or a dedicated forced aligner like MFA.

### 2.6 WER — normalise with Whisper's own normaliser

```python
# metrics.py
import jiwer
from whisper.normalizers import EnglishTextNormalizer, BasicTextNormalizer


def wer(hyps: list[str], refs: list[str], language: str = "en") -> dict:
    """
    Word Error Rate using the exact normaliser the Whisper paper uses.
    Returns both the raw and normalised WER; report the normalised one.
    """
    norm = EnglishTextNormalizer() if language == "en" else BasicTextNormalizer()
    hyps_n = [norm(h) for h in hyps]
    refs_n = [norm(r) for r in refs]

    # drop empty references after normalisation (would 0/0)
    pairs = [(h, r) for h, r in zip(hyps_n, refs_n) if r.strip()]
    hyps_n, refs_n = zip(*pairs)

    raw = jiwer.wer(refs, hyps)
    norm_wer = jiwer.wer(list(refs_n), list(hyps_n))
    return {
        "wer_raw": raw,
        "wer_normalised": norm_wer,
        "normaliser": type(norm).__name__,
        "n_pairs": len(refs_n),
    }
```

Four things reviewers must check:

- **Normaliser matches the language.** `EnglishTextNormalizer` includes English-specific rules (contractions, number spell-out via `inflect`, filler-word stripping); `BasicTextNormalizer` is unicode case-fold + symbol-strip + whitespace-collapse and is what the Whisper paper uses for all non-English languages.
- **Empty-reference guard.** After normalisation some references ("♪") become empty strings; `jiwer.wer` on empty refs divides by zero. Drop those pairs and record the count.
- **`jiwer` is `wer(refs, hyps)`, not `wer(hyps, refs)`.** Flip the argument order and you get a different number because WER is asymmetric (deletions in one direction are insertions in the other). Pin an argument-order unit test.
- **Record the normaliser class name and `jiwer` version.** Whisper's `EnglishTextNormalizer` has changed between `openai-whisper==20231117` and later; the signature protects you from pretending two differently-normalised runs are comparable.

## 3. Validation steps

1. **Resampling round trip.** Load a 48 kHz file, `load_mono_16k`, assert `shape == (duration_s * 16_000,)` and `dtype == float32`. If this fails, every downstream mel-spec is wrong.
2. **VAD smoke test.** 10 s clip = 5 s silence + 5 s speech. `vad_segments` must return exactly one speech segment starting after 5 s. If it returns a segment covering 0 s onward, your VAD threshold is too low.
3. **Language-detection determinism.** On the same clip at the same seed, `detect_language` must return bit-identical probs across runs. If probs drift, you have a non-deterministic resample or a non-determinised `cudnn`.
4. **Confidence thresholding.** Synthesise a short noise clip (white noise, 10 s). `detect_language` top-1 confidence should be <0.3; `apply_fallback(default="en", min_confidence=0.6)` must return `("en", "fallback:...")`. If confidence is >0.9 on noise, your window includes silence padding that the model is confidently classifying.
5. **Chunking without stride drops the tail.** Transcribe a 2-minute clip with `chunk_length_s=30, stride_length_s=0`. Count hyp words vs ref words. Expect gaps near chunk boundaries. Enable `stride_length_s=(5, 5)`; the gaps disappear. This verifies stride is live.
6. **`task="translate"` is a different output language.** On a French clip, `task="transcribe"` → French transcript; `task="translate"` → English transcript. Confirm both; a test accidentally setting `task="translate"` once bit us for weeks.
7. **Temperature-fallback live-fires.** Force the first-pass to hit a repetition loop (short clip with background music often does). Assert the fallback escalates the temperature and the final `compression_ratio` is <2.4. If the pipeline returns on the first temperature despite a 4.0 compression ratio, the fallback kwargs are not being plumbed.
8. **WER sanity.** `hyp="the cat sat on the mat"`, `ref="the cat sat on the mat"` → WER 0. `hyp="the cat sat"`, `ref="the cat sat on the mat"` → WER 0.5 (3 deletions / 6 words). Pin this.
9. **Normaliser parity check.** `hyp="he said twenty five dollars"`, `ref="he said $25"` → raw WER > 0, normalised WER ≈ 0. If normalised WER is also high, the normaliser isn't being applied or isn't the English one.
10. **LibriSpeech test-clean smoke run.** 100 utterances at `openai/whisper-medium` should yield normalised WER in the single digits. If it's >15 %, something upstream is broken (wrong SR, VAD eating speech, wrong model).

## 4. Rubric / review checklist

| Dimension | Pass | Partial | Fail |
|---|---|---|---|
| **Audio ingest** | Explicit 16 kHz mono resample; dtype pinned; sample-rate round-trip test | Resample present, no test | Feeds model native-rate audio |
| **VAD before LID** | Silero/webrtcvad, config documented, LID runs on speech window | VAD present, LID still runs on raw audio | No VAD; LID on silence hallucinates |
| **LID with confidence + fallback** | Top-k returned, confidence threshold, default-language fallback, every override logged | Top-1 only, no threshold | `language=None` always; no logs |
| **Chunked long-form** | `chunk_length_s=30, stride_length_s=(5,5)`; boundary stitching test | Chunking present, no overlap | Transcribe > 30 s files without chunking |
| **Temperature fallback** | Fallback ladder enabled; `compression_ratio_threshold`, `logprob_threshold`, `no_speech_threshold` all set | Some set | All disabled; hallucinations silently accepted |
| **`condition_on_previous_text` discipline** | Set False (or True with a documented data-specific measurement) | Default without justification | True with no awareness; one bad chunk cascades |
| **WER protocol** | Whisper `EnglishTextNormalizer`/`BasicTextNormalizer` + `jiwer`, normaliser class + version recorded, empty-ref drop | WER without normaliser | `.split()` + Levenshtein on raw text |
| **Word timestamps** | `return_timestamps="word"`, caveat about ±100–200 ms accuracy documented | Segment-level only | None |
| **Dataset hygiene** | Whisper checkpoint version, VAD version, sample-rate, audio licence recorded | One of those | None |
| **Reproducibility** | Deterministic seed; eval JSON has all config + normaliser + jiwer version | Seed set, no echo | No seed |
| **Threat model** | OWASP ML Top 10 + MITRE ATLAS risks noted (adversarial audio, prompt injection via transcribed voice, PII in transcripts) with mitigations | Mentioned without mitigations | Ignored |
| **Documentation / card** | NIST AI RMF Map/Measure applied: language coverage, LID confidence histogram, WER per language, normaliser version, known biases (accented/dialectal English) | Some of those | None |

## 5. Common mistakes

- **Running `detect_language` on the full file.** Whisper's LID consumes a fixed 30 s window; feeding more does not improve it and in practice *hurts* because `pad_or_trim` grabs the first 30 s silently. Pick a VAD-confirmed window explicitly.
- **Trusting LID when confidence is near uniform.** 99-language Whisper has 99 possible outputs; a "winner" with p=0.08 is noise. Always threshold-and-fallback; never silently accept a low-confidence winner.
- **Forgetting that `task="translate"` changes the output language.** The API treats transcribe and translate as the same function; a stray `task="translate"` in your decoding kwargs turns French speech into English text and makes the WER go to the moon against a French reference. Guard with a unit test.
- **Passing raw audio to `pipeline(...)` without a dict.** `pipeline(wav_array)` sometimes works and sometimes guesses the sample rate; `pipeline({"array": wav, "sampling_rate": 16_000})` is the contract.
- **Disabling the temperature-fallback ladder.** Setting `temperature=0.0` alone drops a `tuple` to a scalar and disables the fallback; on noisy audio you then see the classic Whisper repetition loops ("you you you you you ..."). Keep the tuple.
- **Setting `condition_on_previous_text=True` by default on noisy data.** On clean podcasts it helps 1–2 WER; on noisy call-centre audio it compounds errors file-wide. The paper uses True; the paper evaluates on cleaner data than most pipelines see.
- **Reporting raw WER.** Raw WER on English ASR is 2–3× the normalised WER on the same hypotheses. It is not comparable to any Whisper paper number and is not the metric reviewers expect. Always report normalised, with the normaliser class name.
- **`jiwer.wer(hyps, refs)` instead of `jiwer.wer(refs, hyps)`.** Argument order is asymmetric; swap them and your delete-vs-insert ratio inverts. The function signature is `wer(reference, hypothesis)`; pin a sign-of-change test.
- **Mel channel count mismatch.** Whisper `large-v3` uses 128-channel mels (`n_mels=128`); earlier versions use 80. Setting the wrong one against the current model silently produces log-mel features that the model has never seen — the transcript is grammatical but unrelated. Always pull `n_mels` from `model.dims.n_mels`.
- **Forgetting that `large-v3` has new language tokens.** v3 added Cantonese (`yue`) and some regional codes v2 didn't have. Code that hard-codes v2's language list will silently drop the new codes to `None`.
- **Assuming word timestamps are frame-accurate.** They are DTW-aligned attention peaks, ±100–200 ms typical, worse on overlapping speech or music. For forced alignment at <50 ms, use a real forced-aligner (MFA) or WhisperX.

## 6. References

Primary sources used in this solution:

- Radford, A. et al. (2022). *Robust Speech Recognition via Large-Scale Weak Supervision* (Whisper). OpenAI. <https://arxiv.org/abs/2212.04356> — the Whisper paper; §2.3 defines the temperature-fallback schedule, §4.4 defines the text normaliser, §C the DTW-based word-timestamp trick.
- OpenAI Whisper repo — <https://github.com/openai/whisper> — the reference CLI implementation (`detect_language`, `transcribe`, `EnglishTextNormalizer`, `BasicTextNormalizer`).
- Hugging Face Transformers ASR pipeline — <https://huggingface.co/docs/transformers/main/en/main_classes/pipelines#transformers.AutomaticSpeechRecognitionPipeline> — the `chunk_length_s`/`stride_length_s` long-form chunker used in §2.4.
- Panayotov, V. et al. (2015). *LibriSpeech: an ASR corpus based on public domain audio books*. ICASSP. <https://www.openslr.org/12/> — LibriSpeech test-clean / test-other reference splits.
- Ardila, R. et al. (2020). *Common Voice: A Massively-Multilingual Speech Corpus*. LREC. <https://arxiv.org/abs/1912.06670> — multilingual benchmark used throughout the Whisper paper.
- Conneau, A. et al. (2022). *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*. <https://arxiv.org/abs/2205.12446> — multilingual low-resource benchmark aligned with FLORES-101.
- Silero VAD — <https://github.com/snakers4/silero-vad> — the voice-activity-detection module used in §2.2.
- `jiwer` — <https://github.com/jitsi/jiwer> — WER / CER reference implementation.
- fastText language identification `lid.176` — <https://fasttext.cc/docs/en/language-identification.html> — the second-opinion text-side LID used for post-transcription cross-check.

Context references listed on the work packet:

- OWASP Machine Learning Security Top 10 — <https://owasp.org/www-project-machine-learning-security-top-10/> — *ML01 Input Manipulation* applies: a voice input can be an adversarial audio clip (ultrasonic, perturbation) that transcribes to arbitrary attacker-chosen text; treat transcripts as untrusted input before any downstream LLM, moderation, or action. *ML02 Data Poisoning* applies when fine-tuning Whisper on scraped audio (transcripts may be mis-aligned or deliberately misleading).
- MITRE ATLAS — <https://atlas.mitre.org/> — ASR is a well-studied adversarial-example surface; the catalogue indexes concrete techniques relevant whenever a transcript is consumed by an agent / LLM / command pipeline.
- NIST AI Risk Management Framework — <https://www.nist.gov/itl/ai-risk-management-framework> — *Map*: record language coverage (per-language WER), checkpoint version, VAD version, sample-rate, dataset licence, known biases (accented/dialectal English, code-switching); *Measure*: report normalised WER per language with the normaliser class name, jiwer version, and the LID-confidence distribution — bias and fairness checks per dialect belong in mod-111 and mod-113.
