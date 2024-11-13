# %%capture
# !pip install pip3-autoremove
# !pip-autoremove torch torchvision torchaudio -y
# !pip install unsloth

import os
os.environ["WANDB_DISABLED"] = "true"

# THIS IS WRONG DO NOT DO THIS
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
print("Pytorch version：")
print(torch.__version__)
print("CUDA Version: ")
print(torch.version.cuda)
print("cuDNN version is :")
print(torch.backends.cudnn.version())

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset

max_seq_length = 10240 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

def generate_and_tokenize_prompt(training_data, input_test_data, sklearn_tree = -1, symmetry_repairing = -1, colors_counter = -1, ice_cube = -1):
    full_prompt =f"""Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input: 
    ---------------------------------------- 
    {training_data} 
    ---------------------------------------- 
    Here are some solutions from Sklearn Tree, Symmetry Repairing, Colors Counter and ICE Cube. You can refer to their solutions to think out your own solution. If their solution is -1, it means they didn’t give a solution.
    ---------------------------------------- 
    Sklearn Tree: {sklearn_tree} 
    Symmetry Repairing: {symmetry_repairing} 
    Colors Counter: {colors_counter} 
    ICE Cube: {ice_cube} 
    ---------------------------------------- 
    Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.: 
    ---------------------------------------- 
    [{{'input': {input_test_data}, 'output': [[]]}}] 
    ---------------------------------------- 
    What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:
    """
    return tokenize(full_prompt)

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 15 trillion tokens model 2x faster!
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # We also uploaded 4bit for 405b!
    "unsloth/Mistral-Nemo-Base-2407-bnb-4bit", # New Mistral 12b 2x faster!
    "unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit",
    "unsloth/mistral-7b-v0.3-bnb-4bit",        # Mistral v3 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",          # Phi-3 2x faster!d
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", # "unsloth/Meta-Llama-3.1-8B"
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=max_seq_length,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

