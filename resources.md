# Resources

The automated resource-finding stage was **skipped** for this run (`.neurico/pipeline_state.json`), so no
literature review, papers, datasets or code were pre-gathered. Everything below was assembled during the
experiment session.

## Models (open weights, run locally; no API key was available in the environment)
| Role | Model | Notes |
|---|---|---|
| Teacher / generator | `Qwen/Qwen2.5-3B-Instruct` | vLLM 0.30, bf16, T=1.0, top-p 0.95, max 512 new tokens |
| Student | `Qwen/Qwen2.5-0.5B-Instruct` | LoRA fine-tuning (PEFT 0.21), evaluated with transformers `generate` |
| Embeddings (diversity, Fréchet distance) | `sentence-transformers/all-MiniLM-L6-v2` | CPU |

## Datasets
See `datasets/README.md`: MBPP (code), GSM8K (math), ROCStories 2018 (human-written five-sentence stories).

## Prior work drawn on (from memory; no PDFs downloaded)
- Shumailov et al., "The Curse of Recursion" (2023) / "AI models collapse when trained on recursively
  generated data" (Nature 2024) — recursive collapse, loss of distribution tails.
- Alemohammad et al., "Self-Consuming Generative Models Go MAD" (2023).
- Gerstgrasser et al., "Is Model Collapse Inevitable?" (2024) — accumulating real + synthetic avoids collapse.
- Dohmatob et al., "A Tale of Tails: Model Collapse as a Change of Scaling Laws" (2024).
- Bertrand et al., "On the Stability of Iterative Retraining of Generative Models on their own Data" (2024).
- Guo et al., "The Curious Decline of Linguistic Diversity" (2023).
- He et al., "Is Synthetic Data from Generative Models Ready for Image Recognition?" (2022); Azizi et al.,
  "Synthetic Data from Diffusion Models Improves ImageNet Classification" (2023); Fan et al., "Scaling Laws
  of Synthetic Images for Model Training … for Now" (2023).
- Ye et al., "ZeroGen" (2022); Yu et al., "Large Language Model as Attributed Training Data Generator"
  (2023); Li et al., "Synthetic Data Generation with Large Language Models for Text Classification:
  Potential and Limitations" (2023).
- Friedman & Dieng, "The Vendi Score" (2022) — diversity metric used in E3.
- Austin et al., MBPP (2021); Cobbe et al., GSM8K (2021); Mostafazadeh et al., ROCStories (2016).

## Code
All code in `src/` was written for this study (no external baselines were cloned).

## Compute actually available
4× RTX A6000 (48 GB) **shared** with four other concurrent jobs at ~100% utilisation; typically 5–20 GB
free per card. This shaped several design decisions (see "Deviations" in `planning.md`).
