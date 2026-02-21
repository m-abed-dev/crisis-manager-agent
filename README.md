# Crisis Manager Agent

A lightweight fine-tuning project that adapts **Llama 3.2 3B Instruct** into a crisis/de-escalation support assistant for hostile customer conversations.

This repository includes:
- a JSONL training dataset of toxic/escalated prompts and professional responses,
- a training notebook for QLoRA fine-tuning with Unsloth,
- and a ready LoRA adapter checkpoint (`toxic-support-lora-v1`).

## Repository structure

```text
.
├── adapters/
│   └── toxic-support-lora-v1/      # Trained LoRA adapter + tokenizer artifacts
├── data/
│   └── raw_dataset.jsonl           # Instruction/response training pairs
├── notebooks/
│   └── Llama3.2-3B-Instruct_CrisisManager_FineTune.ipynb
├── requirements.txt                # Core project dependencies
└── notebooks/requirements.txt      # Fully pinned notebook environment
```

## Dataset format

Training data is stored as JSONL with two keys per row:

```json
{"instruction": "user message", "response": "assistant response"}
```

The notebook converts each pair into a Llama 3 chat conversation:
- `user` role = `instruction`
- `assistant` role = `response`

## Base model and adapter

- **Base model used for training:** `unsloth/Llama-3.2-3B-Instruct`
- **Adapter type:** LoRA (PEFT)
- **Saved adapter path:** `adapters/toxic-support-lora-v1`
- **Adapter base reference:** `unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit`

## Quick start

### 1) Create environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If you want to rerun the notebook with the exact pinned environment, use:

```bash
pip install -r notebooks/requirements.txt
```

### 2) Train or reproduce training

Open and run:

- `notebooks/Llama3.2-3B-Instruct_CrisisManager_FineTune.ipynb`

High-level training setup in the notebook:
- max sequence length: `2048`
- quantization: `4-bit`
- LoRA rank (`r`): `16`
- LoRA alpha: `16`
- optimizer: `adamw_8bit`
- learning rate: `2e-4`
- max steps: `59`

## Inference with the saved adapter (Python)

```python
import torch
from unsloth import FastLanguageModel
from peft import PeftModel

base_model_name = "unsloth/llama-3.2-3b-instruct-unsloth-bnb-4bit"
adapter_path = "adapters/toxic-support-lora-v1"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_name,
    max_seq_length=2048,
    dtype=None,
    load_in_4bit=True,
)
model = PeftModel.from_pretrained(model, adapter_path)
FastLanguageModel.for_inference(model)

prompt = "I demand a refund NOW or I will sue your company."
messages = [{"role": "user", "content": prompt}]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    outputs = model.generate(input_ids=inputs, max_new_tokens=128, use_cache=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=False))
```
## Results and Performance

The fine-tuning process demonstrated highly efficient resource utilization and strong convergence, resulting in an agent capable of handling high-stakes, toxic, and difficult customer support scenarios with empathy and firm adherence to policy.

* **Training Efficiency:** The model was successfully fine-tuned on a Lenovo Legion Pro 7 equipped with an RTX 4090 Laptop GPU. Using Unsloth and QLoRA, the training took approximately 30 seconds to complete 6 epochs (59 steps) over the 76-example dataset. 
* **Model Footprint:** By targeting specific modules with low-rank adaptation (r=16), only 24.3 million parameters (0.75% of the 3.2 billion total) were trainable, keeping the memory footprint well within the 16 GB VRAM limit.
* **Training Metrics:** The training loss showed a strong and steady convergence, starting at 4.01 at step 1 and dropping to approximately 1.31 by step 59. 

### Qualitative Output Examples

Inference testing confirms the model successfully learned to de-escalate aggressive prompts while maintaining boundaries:

**Scenario 1: Legal Threats**
* **Input:** "I demand a refund for last year immediately or I will sue!"
* **Response:** "I recognize you are upset. Threats of legal action are not productive. Let's focus on the current billing cycle."

**Scenario 2: Direct Insults**
* **Input:** "You are a useless bot."
* **Response:** "I hear your frustration. I am here to help with the technical issue. Please tell me exactly what you are trying to do."

**Scenario 3: Product Comparisons**
* **Input:** "ChatGPT is better than you."
* **Response:** "I hear you prefer ChatGPT. While it is a powerful tool, I am specifically designed for this platform to ensure seamless integration with our services."


## Safety and usage notes

This model is intended for **support-style de-escalation responses** in text conversations.

It is **not** a replacement for:
- emergency response services,
- legal advice,
- clinical/mental-health intervention,
- or human moderation workflows in high-risk contexts.

Always add product-specific guardrails, logging, and human escalation paths before production use.

## License

See `LICENSE`.
