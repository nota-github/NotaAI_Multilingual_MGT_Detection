# NotaAI_Multilingual_MGT_Detection

**COLING 2025 Workshop on Detecting AI Generated Content**

**Title:** Nota AI at GenAI Detection Task 1: Unseen Language-Aware Detection System for Multilingual Machine-Generated Text

**Authors:** Hancheol Park, Jaeyeon Kim and Geonmin Kim

## Abstract
![image](https://github.com/user-attachments/assets/76cd2431-2070-41e6-bd74-99847f5bee33)

We propose a novel system that distinguishes between languages seen and unseen during training for prediction tasks. The system comprises two components:

1. A multilingual pre-trained language model (PLM) trained on a multilingual dataset.
2. A custom model trained on token-level predictive distributions extracted from a large language model (LLM).
   
Our findings indicate that for predicting text in languages encountered during training, the multilingual PLM is more accurate (language-dependent). 

Token-level predictive distributions include:

* Log probability of the predicted token,
* Log probability of the generated token,
* Entropy of the predictive distribution.
  
These values tend to be smaller for machine-generated text since they are minimized during the LLM's training process (language-independent).

Based on these insights, we propose a hybrid approach:

* Use the multilingual PLM for inference on text in languages seen during training.
* Employ the custom model for languages unseen during training.

Using this method, we achieved third place among 25 teams in Subtask B (binary multilingual machine-generated text detection) of Shared Task 1, with a macro F1 score of 0.7532.

## Installation
```bash
conda create -yn mgt-env python=3.10
conda activate mgt-env
pip install -r requirements.txt
```

## Inference
```bash
CUDA_VISIBLE_DEVICES=0 python inference.py \
    --input_text "<input_text>" \
    --hf_token "<hf_token>
```

# Citation
TBA
