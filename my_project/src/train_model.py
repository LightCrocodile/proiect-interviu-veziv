import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))


from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
import os

# Import the custom download function
from download_data import download_data

# Ensure `GemmaTokenizer` can be imported
from keras_nlp.my_modules.models.gemma.gemma_tokenizer import GemmaTokenizer

def load_training_data(data_path):
    examples = []
    for file_name in os.listdir(data_path):
        with open(os.path.join(data_path, file_name), 'r', encoding='utf-8') as file:
            examples.append(file.read())
    return examples

def preprocess_text(texts):
    return [text.replace('\n', ' ') for text in texts]

if __name__ == "__main__":
    # Download the data
    download_data('https://drive.google.com/uc?export=download&id=1CEx_TnzAlTMEGqqq5O2QXKoz8fPOUjW8', './data')

    data_path = './data/Oferte test'  # Ensure your training data is placed here
    training_data = load_training_data(data_path)
    training_data = preprocess_text(training_data)

    # Path to the model
    model_path = r"D:\ardu\my_project\data\gemma2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # Tokenize training data
    inputs = tokenizer(training_data, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=inputs['input_ids'],
        data_collator=lambda data: {'input_ids': torch.stack(data)},
    )

    # Train the model
    trainer.train()
