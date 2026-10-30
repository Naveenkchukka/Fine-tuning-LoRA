# TinyLlama Instruction Fine-Tuning with LoRA

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Naveenkchukka/Fine-tuning-LoRA/blob/main/notebooks/tinyllama_lora_dolly.ipynb)

Parameter-efficient adaptation of **TinyLlama-1.1B-Chat-v1.0** using **Databricks Dolly-15K**, Hugging Face PEFT, and TRL's SFTTrainer.

**Status:** Executed on a Tesla T4. The notebook preserves the supplied run outputs and includes explanatory Markdown notes. Validation loss improved; qualitative response quality is mixed. These results come from the uploaded execution record, not an independently repeated run.

## Recorded results

| Metric | Result |
| --- | --- |
| Trainable parameters | 2,252,800 (0.2044%) |
| Validation loss, before → after | 1.8756 → 1.3242 (29.4% reduction) |
| Training | 376 optimizer steps, 2 epochs |
| Training runtime | 31.35 minutes |
| Best saved checkpoint | 350 |
| Adapter weights | 8.62 MiB |
| Peak PyTorch allocated GPU memory | 2.3306 GiB |

Short extraction/classification answers improve in several examples, while other responses become repetitive or misclassify items. The notebook retains all 20 comparisons, the loss plot, training logs and adapter reload output. No numerical human-scored win rate is claimed.

## Objective

Explore a low-cost approach to instruction tuning by training small attention adapters while keeping the base model frozen. The notebook covers data preparation, baseline generation, adapter injection, supervised fine-tuning, persistence, reload, and evaluation.

## Project structure

```text
notebooks/
  tinyllama_lora_dolly.ipynb   # Complete experiment
requirements.txt             # Training stack dependencies
.gitignore                   # Excludes weights, checkpoints and secrets
README.md
```

## Run on Google Colab

1. Open Google Colab and upload `notebooks/tinyllama_lora_dolly.ipynb`.
2. Choose **Runtime → Change runtime type → T4 GPU**.
3. The executed notebook starts with environment verification, not installation. For a fresh dedicated Colab runtime, add and run the setup cell below before the notebook cells.
4. If any libraries were already imported, restart the session after installation and continue from the next code cell.
5. Run the remaining cells in order. A CUDA-enabled PyTorch installation and internet access are required.
6. Download the generated `lora_results` directory before ending the runtime; Colab storage is temporary.

Fresh Colab setup (changes only the current runtime):

```python
%pip uninstall -y gradio gradio-client diffusers gcsfs
%pip install -q pandas==2.2.3 huggingface-hub==0.36.2 fsspec==2025.10.0 transformers==4.57.3 trl==0.26.0 peft==0.18.0 accelerate==1.12.0 datasets==4.4.1 jedi
```

Restart the session if libraries were already imported, then begin with the notebook's environment verification cell. The recorded run used pandas 2.3.3 and reported dependency conflicts; those original warnings are preserved. The suggested cleaned setup has not been independently rerun here. For a local CUDA environment, use a clean virtual environment, a CUDA-compatible PyTorch build and `requirements.txt`; skip the Colab-only uninstall commands.

## Experiment configuration

| Setting | Value |
| --- | --- |
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 |
| Dataset | databricks/databricks-dolly-15k |
| Training / validation / test examples | 3,000 / 200 / 200 |
| Qualitative comparison panel | 20 fixed held-out examples |
| Maximum conversation length | 512 tokens |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Microbatch / gradient accumulation | 1 / 16 |
| Epochs / learning rate | 2 / 1e-4 |
| Precision | FP16 backbone and mixed precision; FP32 adapters |
| Memory control | Non-reentrant gradient checkpointing |
| Training objective | Completion-only cross-entropy |
| Checkpoint selection | Lowest saved validation loss |

Rank 8 provides a modest starting capacity; alpha 16 gives a scaling factor of 2. Adapting all four attention projections allows queries, keys, values and attention output mixing to change. Embeddings, feed-forward layers and the language-model head remain frozen. The code measures and asserts that fewer than 1% of parameters are trainable.

## Data and evaluation

Optional context is appended to the instruction. Exact duplicate user prompts are removed before splitting. Conversations exceeding 512 tokens are excluded, introducing a short-example bias. The dataset is converted to chat messages and then conversational prompt/completion pairs for response-only loss.

Baseline responses are captured before adapter injection. After training, the saved adapter is reloaded onto a fresh backbone and evaluated on the same prompts with identical greedy decoding settings. Model and dataset revision hashes, metrics and training logs are saved for reproducibility. The code resolves the latest source revisions on rerun; use the recorded hashes for the same source versions.

The chat checkpoint already follows instructions. Fine-tuning may improve, leave unchanged, or degrade its responses. Compare instruction compliance, contextual correctness and relevance across the full panel; report wins, ties and regressions. Lower training loss alone does not establish improvement.

## Generated outputs

Running the notebook creates `lora_results/` containing:

- `metrics.json`: validation losses, training metrics, GPU memory and adapter size.
- `training_log.csv` and `loss_curve.png`: measured training history.
- `baseline.json`, `comparison.json` and `comparison.csv`: evaluation outputs.
- `adapter/`: adapter weights, configuration and tokenizer.
- `checkpoints/`: intermediate training checkpoints.

These outputs are ignored by Git. The executed notebook already embeds the loss plot, metrics and comparisons. Saved model weight files are not embedded in the notebook; retain the original Colab downloads separately. Do not commit large model weights, credentials or fabricated results.

## Current validation

- Python code cells pass syntax parsing.
- Instruction/context formatting assertions have been checked locally.
- Supplied outputs record completed CUDA training, adapter saving, successful reload and 20 paired comparisons.
- All original code cells, execution counts and outputs are preserved; annotations are Markdown only.
- The run has no saved exception outputs, but its environment check reports dependency conflicts.
- The run was not independently repeated. Qualitative observations show both improvements and regressions.

## Data and model terms

The dataset card lists Dolly-15K under CC BY-SA 3.0. Refer to the upstream dataset and model cards for their respective terms; this repository does not relicense those materials. No project code license is currently specified.

## References

- [TinyLlama model card](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Dolly-15K dataset card](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
- [TRL 0.26.0 SFTTrainer documentation](https://huggingface.co/docs/trl/v0.26.0/en/sft_trainer)
- [LoRA paper](https://arxiv.org/abs/2106.09685)
