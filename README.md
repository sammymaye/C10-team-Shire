# C10-team-Shire
# Fine-Tuning Gemma-2B for Medical Diabetes Instruction & Q&A 🩺🤖

An instruction-tuned language model based on `google/gemma-2b-it`, fine-tuned using **4-bit NF4 Quantization (QLoRA)** and **PEFT** to provide medical instruction responses on diabetes care, guidance, and clinical guidelines.

---

## 📌 Dataset

The training corpus is a consolidated collection of **6,192 medical Q&A and instruction pairs** derived from three distinct medical datasets:
1. `diabetes_QA_dataset.csv`: Domain-specific clinical questions and answers on diabetes management.
2. `diabetes_instruct_temp_v44.csv`: Contextual instruction-following pairs.
3. `ada_diabetes_5000_instruction.csv`: Standardized guidelines aligned with American Diabetes Association standards.
   

### Preprocessing & Formatting
All entries are unified into a single chat-formatted instruction pipeline following Gemma's turn-based template:

```text
<start_of_turn>user
You are a domain-specific assistant specialized in diabetes care, guidance, and medical context. Provide accurate, helpful, and concise responses.

User Query: {instruction / context}<end_of_turn>
<start_of_turn>model
{output}<end_of_turn>

