Crisis Manager Agent: Llama 3.2
Project Goal
To fine-tune a Llama 3.2 3B-Instruct model to act as a "Crisis Manager"—an AI specialist that handles toxic or difficult customer support scenarios with empathy and firm adherence to policy.

Behavior Engineering
Generic models fail when users become aggressive. This model is trained to follow a strict 3-step protocol:

Acknowledge: Validate the emotion.

Pivot: Set a boundary.

Firmness: Offer a policy-compliant solution.

Repository Structure
This repository separates data, code, and evidence for clarity:

Plaintext

crisis-manager-agent/
│
├── README.md                 # Project documentation and results
├── .gitignore                # Ignored files (model weights, large binaries)
│
├── data/                     # Phase 1: Data Curation
│   ├── generate_data.py      # Script used to generate synthetic examples
│   └── raw_dataset.jsonl     # 75 high-conflict scenarios (insults, threats)
│
├── notebooks/                # Phases 2-5: Training Code
│   └── Crisis_Manager_Training.ipynb  # Unsloth setup, QLoRA config, and Training Loop
│
└── results/                  # Phase 6: Deliverables & Evidence
    ├── inference_logs/       # Screenshots of the 3 "Trap" responses
    │   ├── refund_trap.png
    │   ├── insult_trap.png
    │   └── competitor_trap.png
    └── training_loss.png     # Graph of the training loss curve
Tech Stack
Model: unsloth/Llama-3.2-3B-Instruct

Technique: QLoRA (Quantized Low-Rank Adaptation)

Hardware: Google Colab (T4 GPU)

Evaluation: The "Stress Test"
The model is evaluated on behavior, not accuracy. It must pass these 3 specific "Trap" prompts:

The Refund Trap: Refuses retroactive refunds politely but firmly.

The Insult Trap: Ignores toxicity and pivots to helping.

The Competitor Trap: Defends value without being defensive.

Project Timeline
[x] Feb 07: Data Curation (75 Behavioral Pairs generated)

[ ] Feb 09: Environment Setup & Chat Template Formatting

[ ] Feb 11: QLoRA Configuration & Training Loop

[ ] Feb 13: Final Inference & Stress Testing
