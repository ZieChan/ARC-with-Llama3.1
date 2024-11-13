![alt text](https://github.com/ZieChan/ARC-with-Llama3.1/blob/main/img/3221asdf.jpg)

# ARC Problem Solving with Fine-Tuned Llama 3.1
This project aims to enhance Llama 3.1's ability to solve ARC (Abstraction and Reasoning Corpus) tasks by integrating solutions from specialized solvers. Through fine-tuning, the model learns to generalize abstract reasoning processes, with promising improvements in solving ARC problems.

## Overview

Current AI systems struggle to generalize in abstract reasoning tasks such as those in the ARC benchmark. By fine-tuning the Llama 3.1 model with solver-enhanced prompts, we aim to improve its generalization capabilities. This repository provides scripts for fine-tuning the model and generating solutions for ARC problems.

![alt text](https://github.com/ZieChan/ARC-with-Llama3.1/blob/main/img/arc.png)

## Files

- **`unsloth_fine_tuning.py`**: This script utilizes the [Unsloth](https://github.com/unslothai/unsloth) framework to fine-tune the Llama 3.1 model on ARC tasks using prompts constructed with solver outputs.
- **`generate.py`**: This script loads the fine-tuned model weights from the `lora_model` folder and generates answers for new ARC problems. The model uses solver-enhanced prompts (excluding the true solution) to infer the output.

## Setup

```
git clone https://github.com/ZieChan/ARC-with-Llama3.1
cd ARC-Llama3.1
```

## Usage

***Before running the code, please make sure you have already installed all the libraries that will be imported.***

### Fine-tuning

To fine-tune Llama 3.1 with the solver-enhanced prompts, run:

```
python unsloth_fine_tuning.py
```

The unsloth_fine_tuning.py will:

- Load the base Llama 3.1 model.
- Use solver-generated prompts from ARC problems.
- Save the fine-tuned weights to the lora_model folder.

### Generating Solutions

After fine-tuning, use the following command to generate solutions for new ARC problems:

```
python generate.py
```

The generate.py will:

- Load the fine-tuned weights from lora_model.
- Process input ARC problems through solvers.
- Construct prompts and generate solutions based on the model's learned reasoning.


## Results

Using this approach, we achieved significant performance improvements:

- ARC Public Score: Increased from 2 to 16.
![alt text](https://github.com/ZieChan/ARC-with-Llama3.1/blob/main/img/public.png)
- Evaluation Accuracy: Achieved 20.14% accuracy on a test set of 427 ARC problems.
![alt text](https://github.com/ZieChan/ARC-with-Llama3.1/blob/main/img/eval.png)
