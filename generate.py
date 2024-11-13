import pandas as pd
import json
import os
import ast
import re
import numpy as np
from datasets import Dataset

from time import time

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    set_seed,
    pipeline
)
import torch

# Set test_run variable: False: create submission file for private test set, True: Evaluate on public tasks
test_run = True

# Function to split the tasks that have multiple test input/output pairs.
# This makes the handling easier, we will combine it again at the end for the submission
def split_dictionary(data):
    result = {}
    split_files = []
    for key, value in data.items():
        test_list = value.get("test", [])
        train_list = value.get("train", [])
        if len(test_list) > 1:
            for idx, test_item in enumerate(test_list):
                new_key = f"{key}_{idx}"
                result[new_key] = {
                    "test": [test_item],
                    "train": train_list
                }
                split_files.append(new_key)
        else:
            result[key] = value
    return result, split_files

# Prepare data for DataFrame

# Load JSON data from the files
if test_run:
    with open('/kaggle/input/arc-prize-2024/arc-agi_evaluation_challenges.json') as f:
        challenges = json.load(f)
        # Split tasks with multiple test inputs
        challenges, split_files = split_dictionary(challenges) 

    with open('/kaggle/input/arc-prize-2024/arc-agi_evaluation_solutions.json') as f:
        solutions = json.load(f)
else:
    with open('/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json') as f:
        challenges = json.load(f)
    # Split tasks with multiple test inputs
    challenges, split_files = split_dictionary(challenges) 

# Print how many files have been split and their names
split_file_count = len(split_files)//2

print(f"Number of files split: {split_file_count}")
print("File names:")
for name in split_files:
    print(name)

# Prepare data
data = []
        
for file_name, grids in challenges.items():
    train_grids = grids.get('train', [])
    test_inputs = grids.get('test', [])
    if test_run:
        # Handle files with multiple test inputs
        parts = file_name.split('_')
        if len(parts) > 1:
            test_nr = int(parts[1])
        else:
            test_nr = 0
        test_outputs = solutions.get(parts[0], [])
        # Transform test grids to lists of dicts with 'output' key
        test_outputs_transformed = [{'output': test_outputs[test_nr]}]
        # Combine test inputs and outputs in alternating manner
        combined_tests = [{'input': test_inputs[0]['input'], 'output': test_outputs_transformed[0]['output']}]
    data.append({
            'file_name': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs_transformed if test_run else [[0, 0]],
            'test': combined_tests if test_run else test_inputs
    })

# Create DataFrame
df = pd.DataFrame(data)

LLAMA_3_CHAT_TEMPLATE = """{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"""
max_seq_length = 2048
dtype = None
load_in_4bit = True

model_id = "/lora_model"

# Set the data type for computations to float16, bfloat16 not supported on T4/P100
compute_dtype = getattr(torch, "float16")

# Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Enable 4-bit quantization
    bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
    bnb_4bit_quant_type="nf4",  # Specify the quantization type
    bnb_4bit_compute_dtype=compute_dtype,  # Set the computation data type
)

print("Loading model")
# Load the pre-trained model with specified configurations
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True, # Allow the model to use custom code from the repository
    quantization_config=bnb_config, # Apply the 4-bit quantization configuration
    attn_implementation='sdpa', # Use scaled-dot product attention for better performance
    torch_dtype=compute_dtype, # Set the data type for the model
    use_cache=False, # Disable caching to save memory
    device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
)

# Load the tokenizer associated with the model
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.chat_template = LLAMA_3_CHAT_TEMPLATE # Apply the chat message template

# The system_prompt defines the initial instructions for the model, setting the context for solving ARC tasks.
system_prompt = '''You are a puzzle solving wizard. You are given a puzzle from the abstraction and reasoning corpus developed by Francois Chollet.'''

# User message template is a template for creating user prompts. It includes placeholders for training data and test input data, guiding the model to learn the rule and apply it to solve the given puzzle.
user_message_template = '''Here are the example input and output pairs from which you should learn the underlying rule to later predict the output for the given test input:
----------------------------------------
{training_data}
----------------------------------------
Now, solve the following puzzle based on its input grid by applying the rules you have learned from the training data.:
----------------------------------------
[{{'input': {input_test_data}, 'output': [[]]}}]
----------------------------------------
What is the output grid? Only provide the output grid in the form as in the example input and output pairs. Do not provide any additional information:'''

def preprocess(task, test_run, train_mode=False):
    """
    Preprocess a single ARC task to create the prompt and solution for the model.

    This function formats the system and user messages using a predefined template and the task's training and test data.
    If in training mode, it also includes the assistant's message with the expected output.

    Parameters:
    task (dict): The ARC task data containing training and test examples.
    train_mode (bool): If True, includes the assistant's message with the expected output for training purposes.

    Returns:
    dict: A dictionary containing the formatted text prompt, the solution, and the file name.
    """
    # System message
    system_message = {"role": "system", "content": system_prompt}

    # Extract training data and input grid from the task
    training_data = task['train']
    input_test_data = task['test'][0]['input']
    if test_run:
        output_test_data = task['test'][0]['output']
    else:
        output_test_data = [[0 ,0]]

    # Format the user message with training data and input test data
    user_message_content = user_message_template.format(training_data=training_data, input_test_data=input_test_data)
    user_message = {
        "role": "user",
        "content": user_message_content
    }

    # Include the assistant message with the expected output if in training mode
    if train_mode:
        assistant_message = {
            "role": "assistant",
            "content": str(output_test_data)
        }

        # Combine system, user, and assistant messages
        messages = [system_message, user_message, assistant_message]
    else:
        messages = [system_message, user_message]
    # Convert messages using the chat template for use with the instruction finetuned version of Llama
    messages = tokenizer.apply_chat_template(messages, tokenize=False)
    if test_run:
        return {"text": messages, "solution": output_test_data, "file_name": task['file_name']}
    else:
        return {"text": messages, "file_name": task['file_name']}

# Convert the loaded data to a Huggingface Dataset object
dataset = Dataset.from_pandas(df)

# Apply the preprocess function to each task in the dataset
dataset = dataset.map(lambda x: preprocess(x, test_run), batched=False, remove_columns=dataset.column_names)

# Define the maximum number of tokens allowed
max_tokens = 8000  # Adjust this value as needed


# Function to calculate the number of tokens
def count_tokens(text):
    return len(tokenizer.encode(text))

# Filter the dataset to include only tasks with a number of tokens within the allowed limit
filtered_dataset = dataset.filter(lambda x: count_tokens(x['text']) <= max_tokens)

# Define your LLM pipeline
text_gen_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Define terminators for the pipeline
terminators = [
    text_gen_pipeline.tokenizer.eos_token_id,
    text_gen_pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Function to generate outputs
def generate_solution(task, max_new_tokens=1024, do_sample=True, temperature=0.1, top_p=0.1):
    # Extract the prompt from the task
    prompt = task['text']
    
    # Generate the model's output based on the prompt
    outputs = text_gen_pipeline(
        prompt, 
        max_new_tokens=max_new_tokens, 
        eos_token_id=terminators, 
        do_sample=do_sample, 
        temperature=temperature, 
        top_p=top_p
    )
    
    # Extract the generated solution from the model's output
    generated_solutions = outputs[0]["generated_text"][len(prompt):]
    return {'generated_solution': generated_solutions}

def extract_solution(text):
    try:
        # Find the part of the text that looks like a nested list
        start = text.index('[[')
        end = text.index(']]', start) + 2
        array_str = text[start:end]
        
        # Use ast.literal_eval to safely evaluate the string as a Python expression
        array = ast.literal_eval(array_str)
        
        # Check if the result is a list of lists
        if all(isinstance(i, list) for i in array):
            return array
        else:
            return [[0]]
    except (ValueError, SyntaxError):
        return [[0]]
    
def pad_array_with_value(array, target_shape, pad_value):
    padded_array = np.full(target_shape, pad_value, dtype=int)
    original_shape = np.array(array).shape
    padded_array[:original_shape[0], :original_shape[1]] = array
    return padded_array

def compare_solutions_with_padding(generated_output, correct_output, pad_value=-1):
    max_rows = max(len(generated_output), len(correct_output))
    max_cols = max(len(generated_output[0]), len(correct_output[0]))
    target_shape = (max_rows, max_cols)
    
    padded_generated = pad_array_with_value(generated_output, target_shape, pad_value)
    padded_correct = pad_array_with_value(correct_output, target_shape, pad_value)
    
    total_pixels = max_rows * max_cols
    correct_pixels = np.sum((padded_generated == padded_correct) & (padded_generated != pad_value) & (padded_correct != pad_value))
    correct_percentage = (correct_pixels / total_pixels) * 100
    
    is_correct = (correct_pixels == total_pixels)
    
    return is_correct, correct_percentage

solution_dict = {}

for i, task in enumerate(filtered_dataset):
    file_name = task['file_name']
    generated_text = task["generated_solution"]
    # Extract the solution generated by the model
    gen_solution = extract_solution(generated_text)
    # For now we only do one attempt
    solution_dict[file_name] = [
        {
            "attempt_1": gen_solution,
            "attempt_2": gen_solution
        }
    ]

# Recombining the solutions for split files
combined_solution_dict = {}
combined_files = {}

for file_name, attempts in solution_dict.items():
    base_name = file_name.split('_')[0]
    if base_name not in combined_solution_dict:
        combined_solution_dict[base_name] = []
        combined_files[base_name] = []
    combined_solution_dict[base_name].extend(attempts)
    if '_' in file_name:
        combined_files[base_name].append(file_name)

# Printing which file names have been combined
print("Files that have been combined:")
for base_name, files in combined_files.items():
    if files:  # Print only if there are files that were combined
        print(f"{base_name}: {', '.join(files)}")

# We still need to fill in dummy solutions for the tasks we did not consider to make a valid submission:
# Load the sample submission file
with open('/kaggle/input/arc-prize-2024/sample_submission.json') as f:
    sample_submission = json.load(f)
# Fill in all entries that are still missing from the sample_submission file
for key, value in sample_submission.items():
    if key not in combined_solution_dict:
        combined_solution_dict[key] = value

# Create submission
with open("submission.json", "w") as json_file:
    json.dump(combined_solution_dict, json_file)