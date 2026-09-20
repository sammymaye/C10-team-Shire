# C10-team-Shire
# A Domain-Specific Small Language Model for Type 2 Diabetes Self-Management Support

An instruction-tuned language model based on `google/gemma-2b-it`, fine-tuned using **4-bit NF4 Quantization (QLoRA)** and **PEFT** to provide medical instruction responses on diabetes care, guidance, and clinical guidelines.

---

## 📌 Dataset

The training corpus is a consolidated collection of **6,192 medical Q&A and instruction pairs** derived from three distinct medical datasets:
1. `diabetes_QA_dataset.csv`: Domain-specific clinical questions and answers on diabetes management.
2. `diabetes_instruct_temp_v44.csv`: Contextual instruction-following pairs.
3. `ada_diabetes_5000_instruction.csv`: Standardized guidelines aligned with American Diabetes Association standards.
   

## Preprocessing & Formatting
All entries are unified into a single chat-formatted instruction pipeline following Gemma's turn-based template:

```text
<start_of_turn>user
You are a domain-specific assistant specialized in diabetes care, guidance, and medical context. Provide accurate, helpful, and concise responses.

User Query: {instruction / context}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>

```

# 🏗️ Training Pipeline
## Data Collection & Preprocessing
Raw dataset files are loaded, cleaned, and combined into a tabular structure. Each row's user query is wrapped with a standardized system prompt prepended before tokenization.

## Model Architecture & Key Design Choices

### Base Model
google/gemma-2b-it (Gemma 2B Instruction-Tuned).

### Quantization
4-bit NormalFloat (NF4) quantization via BitsAndBytesConfig with bfloat16 compute data type and double quantization enabled for optimal VRAM efficiency.   

### PEFT / LoRA Configuration
Parameter-Efficient Fine-Tuning using Low-Rank Adaptation (r=16, lora_alpha=32, dropout=0.05).Adapters were attached across all linear attention and MLP projections (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj).

## Hyperparameters & Search

### Optimizer
8-bit AdamW (paged_adamw_8bit).   

### Learning Rate & Schedule
$2 \times 10^{-4}$ with a linear decay schedule and a $0.03$ warmup ratio.

### Batch Size & Accumulation
per_device_train_batch_size=1 with gradient_accumulation_steps=8 (effective batch size of 8).

### Training Epochs & Steps
Trained for $0.1$ epochs ($78$ total steps) to establish rapid convergence without catastrophic forgetting on the base instruction capabilities.

# 🧪 Evaluation

Method verification was conducted via training loss logging, checkpoint analysis, and validation set inference comparisons against clinical reference standards.

## Training Loss Trajectory

| Step | Training Loss |
| :---: | :---: |
| 10 | 1.516167 |
| 20 | 0.215249 |
| 30 | 0.149305 |
| 40 | 0.181473 |
| 50 | 0.149449 |
| 60 | 0.100576 |
| 70 | 0.121223 |

The model achieved its minimal training loss of 0.100576 at step 60.

## Qualitative Output Verification

Generated outputs were sampled against standard ADA clinical guidelines to confirm accurate advice on blood glucose target ranges, emergency escalation signs (e.g., ketoacidosis), and dietary management.

## 🔁 Reproduction Steps

Follow these exact steps in order to reproduce data processing, fine-tuning, and inference from scratch:

### 1. Environment Setup

git clone [https://github.com/YOUR_USERNAME/gemma-2b-diabetes-assistant.git](https://github.com/YOUR_USERNAME/gemma-2b-diabetes-assistant.git)
cd gemma-2b-diabetes-assistant
pip install -r requirements.txt

### 2. Download and Preprocess Data

Run the data script to verify or assemble the consolidated CSV datasets into formatted JSONL pairs:

python data/download_data.py
python scripts/preprocess.py

### 3. Run Fine-Tuning Pipeline

Train the LoRA adapter weights on the preprocessed corpus:

python scripts/train.py

### 4. Run Evaluation & Inference
Generate responses and run verification benchmarks against test queries:

python scripts/evaluate.py

Alternatively, you can run the step-by-step interactive Google Colab notebook located at notebooks/gemma_2b_diabetes_finetuning.ipynb.

## 👥 Appendix: Contributors & Mentors
### Team Members

Adejumo O. Emmanuel — Lead Machine Learning Engineer (Dataset curation, QLoRA pipeline development, model training, and documentation)

Cohort & Mentors
AI Saturdays Lagos (Cohort 10)
