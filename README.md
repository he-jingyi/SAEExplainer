# SAEExplainer

SAEExplainer is a feature-explanation training pipeline built around:

- SFT for feature-conditioned explanation generation
- preference construction from generated explanations and SAE activation scoring
- iterative DPO training on the resulting preference pairs
- evaluation of produced explanations with local activation-based metrics

## Directory Layout

```text
SAEExplainer/
├── README.md
├── requirements.txt
├── configs/
│   └── explamma_15/                             # main concrete config set for this repo
│       ├── config_data_explamma_15.yaml         # optional raw feature collection
│       ├── config_sft_explamma_15.yaml          # SFT training
│       ├── config_preference_explamma_15_dpo_train.yaml
│       ├── config_preference_explamma_15_val_from_sft.yaml
│       ├── config_dpo_explamma_15.yaml          # DPO round 1
│       ├── config_preference_explamma_15_dpo_round2_train.yaml
│       ├── config_preference_explamma_15_val_from_dpo1.yaml
│       └── config_dpo_explamma_15_round2.yaml   # DPO round 2
├── data/
│   └── explamma_15/                             # current committed dataset entrypoint
│       ├── sft.jsonl                            # SFT train split
│       ├── val.jsonl                            # SFT val split / DPO val source
│       ├── dpo.jsonl                            # preference-construction source
│       └── test.jsonl                           # final evaluation split
├── scripts/
│   ├── install_server_deps.sh                   # environment bootstrap
│   ├── prepare_sft_data.py                      # optional Neuronpedia feature collection
│   ├── train_sft.py                             # SFT training
│   ├── run_preference_pipeline.py               # preference construction
│   ├── filter_preference_pairs.py               # optional explicit post-filtering
│   ├── precompute_dpo_reference_logps.py        # optional reference cache build
│   ├── train_dpo.py                             # DPO training
│   ├── evaluate_dpo.py                          # generation DPO evaluation
│   ├── evaluate_feature_descriptions.py         # input/output metric evaluation
│   └── ...
├── src/
│   ├── sft/                                     # SFT model, dataset, trainer, eval
│   ├── preference/                              # preference construction and SAE scoring
│   ├── dpo/                                     # DPO dataset, trainer, evaluation
│   ├── data_collection/                         # Neuronpedia data utilities
│   └── feature_description_eval.py              # feature-description evaluation pipeline
└── outputs/                                     # generated, gitignored
```


## Environment

```bash
cd /path/to/SAEExplainer
python3 -m venv .venv
source .venv/bin/activate
bash scripts/install_server_deps.sh --with-flash-attn
```

Recommended server baseline:

- `torch==2.8.0`
- `torchvision==0.23.0`
- `torchaudio==2.8.0`
- PyTorch wheel index: `https://download.pytorch.org/whl/cu126`

For preference construction and evaluation:

```bash
export OPENAI_API_KEY="your_openai_api_key"
```

For optional Neuronpedia-backed collection:

```bash
export NEURONPEDIA_API_KEY="your_neuronpedia_api_key"
```

Optional runtime settings:

```bash
export TOKENIZERS_PARALLELISM=false
export SAE_EXPLAINER_TORCH_NUM_THREADS=1
export SAE_EXPLAINER_TORCH_INTEROP_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Optional Step 0: Collect Raw Features

```bash
python scripts/prepare_sft_data.py \
  --config configs/explamma_15/config_data_explamma_15.yaml
```


## Step 1: Train SFT

```bash
python -u scripts/train_sft.py \
  --config configs/explamma_15/config_sft_explamma_15.yaml
```


## Step 2: First-Round Preference Generation

Training preference file for DPO1:

```bash
python -u scripts/run_preference_pipeline.py \
  --config configs/explamma_15/config_preference_explamma_15_dpo_train.yaml
```

Validation preference file for DPO1:

```bash
python -u scripts/run_preference_pipeline.py \
  --config configs/explamma_15/config_preference_explamma_15_val_from_sft.yaml
```


## Step 3: Train DPO Round 1


Training:

```bash
python -u scripts/train_dpo.py \
  --config configs/explamma_15/config_dpo_explamma_15.yaml
```


## Step 4: Second-Round Preference Generation

Training preference file for DPO2:

```bash
python -u scripts/run_preference_pipeline.py \
  --config configs/explamma_15/config_preference_explamma_15_dpo_round2_train.yaml
```

Validation preference file for DPO2:

```bash
python -u scripts/run_preference_pipeline.py \
  --config configs/explamma_15/config_preference_explamma_15_val_from_dpo1.yaml
```


## Step 5: Train DPO Round 2


Training:

```bash
python -u scripts/train_dpo.py \
  --config configs/explamma_15/config_dpo_explamma_15_round2.yaml
```


## Evaluation Commands

### Generative Test


```bash
python scripts/evaluate_dpo.py \
  --checkpoint-dir outputs/dpo_explamma_15_round2/final \
  --sft-config configs/explamma_15/config_sft_explamma_15.yaml \
  --input-jsonl data/explamma_15/test.jsonl \
  --output-jsonl outputs/dpo_explamma_15_round2/dpo_eval.jsonl \
  --output-summary outputs/dpo_explamma_15_round2/dpo_eval.summary.json \
  --generated-explanations-jsonl outputs/dpo_explamma_15_round2/dpo_eval.generated_explanations.jsonl \
  --progress-json outputs/dpo_explamma_15_round2/dpo_eval.progress.json
```

### Input Test


```bash
python scripts/evaluate_feature_descriptions.py \
  --sft-config configs/explamma_15/config_sft_explamma_15.yaml \
  --input-jsonl data/explamma_15/test.jsonl \
  --explanation-source dpo_checkpoint \
  --checkpoint-dir outputs/dpo_explamma_15_round2/final \
  --skip-output \
  --output-jsonl outputs/feature_desc_explamma_15/input_only_dpo2.jsonl \
  --output-summary outputs/feature_desc_explamma_15/input_only_dpo2.summary.json \
  --progress-json outputs/feature_desc_explamma_15/input_only_dpo2.progress.json
```

### Output Test

Output-only evaluation for DPO round 2:

```bash
python scripts/evaluate_feature_descriptions.py \
  --sft-config configs/explamma_15/config_sft_explamma_15.yaml \
  --input-jsonl data/explamma_15/test.jsonl \
  --explanation-source dpo_checkpoint \
  --checkpoint-dir outputs/dpo_explamma_15_round2/final \
  --skip-input \
  --output-jsonl outputs/feature_desc_explamma_15/output_only_dpo2.jsonl \
  --output-summary outputs/feature_desc_explamma_15/output_only_dpo2.summary.json \
  --progress-json outputs/feature_desc_explamma_15/output_only_dpo2.progress.json
```
