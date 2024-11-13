![alt text]([https://pub-e93d5c9fdf134c89830082377f6df465.r2.dev/2024/07/Meta-Llama-31-1024x576.webp](https://github.com/ZieChan/ARC-with-Llama3.1/blob/main/img/3221asdf.jpg))

# Fine-Tuning Llama 3.1 on medical questionnaires
Llama 3.1 is a great open-source Large Language Model. It is 3 different models. For this fine-tuning, the 8 billion parameter lightweight model was used due to the limitation of compute and GPU. Google Colab was used with T4 GPU and High RAM, and the unsloth accelerator was used for faster and more efficient fine-tuning. 

## Data Preparation
[notebook](notebooks/Llama_3_1_fine_tuning_with_LoRA.ipynb)

The first step was to pick the dataset and perform some data preparation. The dataset that was picked was Multiple Choice Questions, which would have to be prepared for the Question/Answer prompt. There were 4 columns with the choices and once column mentioning the correct choices column. There was also a column that explained the answer choice

For this training purpose, the questions that had single-choice instead of multiple-choice answers were selected. The answer was added to the new `answer` column. 

It was observed that there were repetitions of the answer choice number mentioned in the answer explanation column, so a data cleanup was required. Some patterns were identified for the repetition and using regex, the matched patterns were replaced. 
Since the number of rows was in the hundreds of thousands, for this exercise only 10,000 rows were chosen based on the explanation length

## Fine Tuning
[notebook](notebooks/data_prep_for_medical_question_training.ipynb)

Due to the ease of use with Google Colab, the unsloth accelerator was used. It is very beginner friendly and has great support with the Hugging Face transformer library.

For the fine-tuning adapter, Low-Rank Adaptation (LoRA) was chosen. LoRA is a technique used to fine-tune large language models like LLaMA because it allows for efficient and effective adaptation to specific tasks or domains. LoRA fine-tuning updates only a small subset of the model's parameters, making it more computationally efficient and preserving the original model's knowledge while adapting to new information. This approach enables the development of specialized models like me, tailored to assist with a wide range of tasks and questions. Only selected target modules were targeted for updates.

Using the Supervised Fine Tuning Trainer, the training was done. Auto find batching size was used to map the training batch.

Once the model was trained, the model was saved locally and also pushed to the hugging face hub. Here is a sample of the text generated from the inference - 
```
<|begin_of_text|>Give an answer for the following medical question.

### Question:
What is X-linked muscular dystrophy?

### Answer:

Duchenne's muscular dystrophy. Duchenne's muscular dystrophy is X-linked muscular dystrophy.
<|end_of_text|>
```

As it can be seened the model was able to correctly extrapolate the question and provide an acceptable answer


# Run the model
To run the model for inference, the unsloth accelerator is needed. Pull the model and tokenizer from the hf hub and set the model for inference and generate text by passing the promt - 

```python
model_from_hub, tokenizer_from_hub = FastLanguageModel.from_pretrained("quazirab/Llama-3.1-fine-tuning-with-LoRA-medical-qa-datasets")

FastLanguageModel.for_inference(model_pretrained)

inputs = tokenizer(
    [   """
        Give an answer for the following medical question.

        ### Question:
        "What is the cause of cherry red spot?"

        ### Answer:

        """
    ], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model_pretrained.generate(**inputs, streamer = text_streamer, max_new_tokens = 4096)
```

This will give the following output - 

```
<|begin_of_text|>Give an answer for the following medical question.

### Question:
What is the cause of cherry red spot

### Answer:

Retinal haemorrhage. Retinal haemorrhage is seen in cases of head injury.
<|end_of_text|>
```
