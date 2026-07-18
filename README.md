# Adapting STORM for Named Entity Recognition (NER)

## Introduction

This repository contains the codebase for the Bachelor's thesis "Adapting Sequence-Level Meta-Learning for Robust Named Entity Recognition: Training on Real-World Noisy Data". It extends the sequence-level STORM (Self-Taught On-the-fly Rescaling via Meta loss) framework to handle token-level classification tasks, specifically NER. 

While the original STORM framework dynamically mitigates label noise without requiring clean validation data, its sequence-level design presents architectural challenges when applied to granular, multi-token labeling. This project bridges that gap by introducing:
*   **Token-Aggregated Sentence Loss:** Flattens 2D token-level loss grids into 1D per-sentence loss vectors using custom dimension-flattening, padding-masking, and loss-averaging mechanisms.
*   **Class-Balanced Dictionary Weights:** Modulates token-level loss contributions according to class frequency to prevent the dominant 'O' background class from masking critical errors on sparse entity tokens.
*   **Entity Density Amplifier:** Scales the flattened 1D sequence loss in proportion to the concentration of entity tokens, ensuring that sparse but informative entity signals are preserved and amplified for the meta-rescaler.

## Recent updates

*   2026.07.18: Initial commit (Codebase for Bachelor thesis submission)

## Datasets

This adaptation is specifically evaluated using the NOISEBENCH benchmark, a derivation of the CoNLL-2003 English NER dataset that provides several noisy training variants. The raw text data is organized in the `data/noisebench/` directory:
*   **Clean Reference:** `clean/clean.train`, `clean.dev`, `clean.test`
*   **Authentic Human Noise:** `noise_crowd.train`
*   **Algorithmic Weak Supervision:** `noise_weak.train`

### Data Preprocessing

Instead of providing hardcoded PyTorch tensors, we provide a dynamic preprocessing script (`preprocess_ner.py`). This script reads the raw CoNLL files, tokenizes them using `roberta-base`, aligns the BIO tags with subword tokens, and automatically computes the noise mask by comparing the noisy labels against the clean reference labels.

To process the Crowd noise dataset as an example, run:

```bash
python preprocess_ner.py \
    --clean_train data/noisebench/clean/clean.train \
    --noisy_train data/noisebench/noise_crowd.train \
    --dev_file data/noisebench/clean/clean.dev \
    --test_file data/noisebench/clean/clean.test \
    --out_prefix crowd 
```    




### Preparations and Data Generation

To reproduce the diagnostic and boundary experiments detailed in the thesis, we provide scripts to generate specific noise distributions from the base NOISEBENCH files.

**1. Generate Controlled 30% Mixed Crowd Noise:**
Generates a mixed corpus composed of 70% clean sentences and 30% crowd-noisy sentences.
```bash
python3 scripts/mix_real_noise.py \
    --clean_input data/noisebench/clean/clean.train \
    --noisy_input data/noisebench/noise_crowd.train \
    --output data/noisebench/mixed_crowd30.train \
    --noise_rate 0.30
```    
**2. Generate Synthetic 30% Uniform Noise:**
Injects uniform random noise into the clean training set by corrupting 30% of the tokens.
```bash
python3 scripts/inject_noise.py \
    --input data/noisebench/clean/clean.train \
    --output data/noisebench/noise_synthetic_30.train \
    --noise_rate 0.30
```

## How to run

### Preparations

1. Clone this repository to your local machine or HPC environment.
2. **Download Local Base Model:** The HPC execution scripts use the `--local_files_only` flag to prevent connection timeouts on compute nodes. You must download the [roberta-base](https://huggingface.co/FacebookAI/roberta-base) files from Hugging Face into a root folder named `roberta_local` before running any scripts:
```bash
   git lfs install
   git clone https://huggingface.co/FacebookAI/roberta-base roberta_local
```
   
3. Run preprocess_ner.py (see Data Preprocessing above) to generate the processed tensors for your chosen variant; the DO.* scripts expect these to be present.
4. **Calculate Weights:** Compute the dataset-specific Class-Balanced Dictionary Weights. The input tensor path is set at the top of the script (default: `train_ner_crowd.pt`) — edit it to point at the processed `.pt` file for the split you're training on, then run:
   ```bash
   python3 scripts/calc_weights.py
   ```
   The script prints a `weights_array` you copy into your training config. These weights are pre-configured in the provided `DO.*` scripts for reproducibility.

### Run

Scripts are provided to execute the full 5-seed benchmark suites (Pure Baseline, Enhanced Baseline, Pure STORM, and STORM Full) across the different noise conditions evaluated in the thesis.
- `DO.benchmark_clean` Establishes the oracle upper bound using clean data.
- `DO.benchmark_synthetic30` boundary experiment on 30% uniform random noise.
- `DO.ablation` Performs component ablation studies on the weak supervision split.
- `DO.benchmark_crowd`  Trains on authentic human annotations (Crowd Noise).
- `DO.benchmark_crowd30` diagnostic investigation (controlled noise saturation) on the 70/30 mixed real-world corpus.

To run a benchmark on an HPC cluster : 
   ```bash
   qsub DO.benchmark_crowd
   ```

Using the command line parameter `--simulate_only` to storm.py (already configured inside the DO.* scripts for the baseline runs) will recreate the baselines without applying STORM.
Note that this will deactivate the loss rescaling, but not the training of the loss rescaler. The loss rescaler merely remains unused.

## Requirements
- torch (tested: 2.6.0+cu124)
- transformers (tested: 4.57.6)
- tensorboardX (tested: 2.6.4)
- scikit-learn
- git-lfs (required to download the local model)

## Citation

If you use this NER adaptation or the custom token-aware loss preparation methods in your own work, please cite the Bachelor's thesis:

```bibtex
@mastersthesis{bellaaj2026stormner,
    title = "Adapting Sequence-Level Meta-Learning for Robust Named Entity Recognition: Training on Real-World Noisy Data",
    author = "Bellaaj, Oussema",
    school = "Heinrich Heine University D{\"u}sseldorf",
    year = "2026",
    type = "Bachelor's Thesis"
}
```
This repository builds upon the foundational sequence-level STORM framework, published as Learning from Noisy Labels via Self-Taught On-the-Fly Meta Loss Rescaling. If you use the core STORM meta-learning algorithm, please also cite the original authors:

```
@inproceedings{heck2025storm,
    title = "Learning from Noisy Labels via Self-Taught On-the-Fly Meta Loss Rescaling",
    author = "Heck, Michael and Geishauser, Christian and Lubis, Nurul and van Niekerk, Carel and 
    	      Feng, Shutong and Lin, Hsien-Chin and Ruppik, Benjamin Matthias and Vukovic, Renato and
              Ga{\v{s}}i{\'c}, Milica",
    booktitle = "Proceedings of the AAAI Conference on Artificial Intelligence",
    month = "Mar.",
    year = "2025",
    volume = "39",
    address = "Philadelphia, Pennsylvania, USA",
    publisher = "AAAI Press, Washington, DC, USA",
    organization = "Association for the Advancement of Artificial Intelligence"
}
```
