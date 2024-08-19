
# Text Generation with GPT-Neo Models

This project demonstrates how to use different variants of the GPT-Neo model for text generation. The implementation is done in Python using the Hugging Face `transformers` library and is designed to work within a Google Colab environment.

## Requirements

To run this project, you need to install the `transformers` library. If you're using Google Colab, the installation can be done using the following command:

```bash
!pip install transformers
```

### Dependencies

The following dependencies are required:

- `transformers`: The main library used for loading and interacting with GPT-Neo models.
- `numpy`: Used for numerical operations.
- `tqdm`: Provides progress bars for loops.
- `requests`: Handles HTTP requests.
- `safetensors`: Manages safe serialization of tensors.

All of these dependencies are automatically installed with the `transformers` library.

## Models Used

This project uses the following GPT-Neo models for text generation:

1. `EleutherAI/gpt-neo-125m`: A smaller version of GPT-Neo.
2. `EleutherAI/gpt-neo-1.3B`: A medium-sized version of GPT-Neo.
3. `EleutherAI/gpt-neo-2.7B`: The largest version of GPT-Neo used in this project.

## Implementation

### 1. Importing the Necessary Libraries

```python
from transformers import pipeline
```

### 2. Defining the Prompt

```python
prompt = """All the world's a stage and we are"""
```

### 3. Function to Generate Text

The `text_generator` function initializes the text generation pipeline with the selected model and generates text based on the given prompt.

```python
def text_generator(model, prompt):
    generator = pipeline('text-generation', model=model)
    results = generator(prompt, max_length=125, pad_token_id=50256)
    return results[0]['generated_text']
```

### 4. Text Generation with Different Models

The text generation is performed sequentially using the three GPT-Neo models:

- **Small Model** (`EleutherAI/gpt-neo-125m`)
- **Medium Model** (`EleutherAI/gpt-neo-1.3B`)
- **Large Model** (`EleutherAI/gpt-neo-2.7B`)

```python
small_generator = text_generator('EleutherAI/gpt-neo-125m', prompt)
medium_generator = text_generator('EleutherAI/gpt-neo-1.3B', prompt)
large_generator = text_generator('EleutherAI/gpt-neo-2.7B', prompt)
```

### 5. Results and Observations

After generating text with each model, the results are printed out to compare the quality of the outputs. The generated texts were as follows:

- **Small Model**: Repetitive and less coherent.
- **Medium Model**: Improved coherence but still somewhat fragmented.
- **Large Model**: The most coherent and logical, providing an output that made the most sense.

```python
print(small_generator)
print(medium_generator)
print(large_generator)
```

### 6. Conclusion

Among the three models, the **EleutherAI/gpt-neo-2.7B** model provided the most sensible output, indicating that larger models tend to generate more coherent and contextually accurate text.

## Thoughts

The project highlights the difference in performance between smaller and larger models within the GPT-Neo family. It emphasizes the importance of model size when aiming for higher quality text generation.

## Running the Project

To run this project, simply open it in Google Colab, install the required dependencies, and execute the cells sequentially. The results will showcase the differences in text generation capabilities between the models.

## License

This project is licensed under the MIT License.

