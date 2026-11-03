# RULER: Instance-aware Rubric Rewards for SVG Generation

[![Project Page](https://img.shields.io/badge/🌐_Project_Page-2F80ED?style=flat-square)](https://hangyuran.github.io/RULER/)
[![Code](https://img.shields.io/badge/Code-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/HangyuRan/RULER)
[![arXiv](https://img.shields.io/badge/arXiv-2609.25270-B31B1B?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2609.25270)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-Paper-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2609.25270)
[![Dataset](https://img.shields.io/badge/🤗_Dataset-RULER-FFD21E?style=flat-square)](https://huggingface.co/datasets/hangyuran/RULER)

## Setup

Create a Python 3.12.0 environment on Linux with NVIDIA GPUs and install the dependencies:

```bash
git clone https://github.com/HangyuRan/RULER.git
cd RULER

conda create -n ruler python=3.12.0 -y
conda activate ruler

pip install -r requirements.txt
pip install "verl[vllm] @ git+https://github.com/volcengine/verl.git@v0.7.1"
bash setup_environment.sh
```

## 1. Rubric Generation

Configure an OpenAI-compatible model endpoint and generate instance-aware rubrics:

```bash
export OPENAI_API_KEY="YOUR_API_KEY"
export OPENAI_BASE_URL="YOUR_API_BASE_URL"
export RUBRIC_MODEL="YOUR_MODEL_NAME"

python rubric_generation/generate_rubrics.py \
  --input_parquet data/icon.parquet \
  --output_dir outputs/icon_rubrics
```

The script generates an ideal SVG and a six-item rubric for each prompt.

## 2. Training

Set the vision-language model (VLM) judge endpoint, model path, and training data path:

```bash
export VLM_ENDPOINTS="http://127.0.0.1:8000/v1"
export VLM_MODEL="YOUR_SERVED_JUDGE_MODEL"
export HF_MODEL_PATH="/path/to/model"
export TRAIN_DATA_PATH="/path/to/icon.parquet"

bash training/train_icon_grpo.sh
```

For illustration training, use `training/train_illustration_grpo.sh`.

The rubric reward implementation is in `training/reward/`.

To convert a VERL FSDP checkpoint to Hugging Face format, run:

```bash
bash training/merge_checkpoint.sh \
  /path/to/checkpoint/actor \
  /path/to/output_model
```

## 3. Inference

Generate icons with a trained model:

```bash
python inference/generate_icons.py \
  --model_path /path/to/checkpoint \
  --data_path /path/to/text2svg.parquet \
  --output_dir outputs/icons
```

For illustration generation, use `inference/generate_illustrations.py`.

## 4. Evaluation

Compute offline evaluation metrics:

```bash
python evaluation/evaluate_offline_metrics.py \
  --results-json outputs/icons/results.json \
  --output-json outputs/icons/metrics.json
```

For VLM-based evaluation, use `evaluation/evaluate_with_vlm_judge.py`. Individual metric implementations are in `evaluation/metrics/`.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{ran2026rulerinstanceawarerubricrewards,
  title={{RULER}: Instance-aware Rubric Rewards for {SVG} Generation},
  author={Hangyu Ran and Yuhao Zheng and Yingying Zhang and Kevin Qinghong Lin and Han Peng},
  year={2026},
  eprint={2609.25270},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2609.25270}
}
```
