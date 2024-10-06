from datasets import Dataset, load_dataset
from transformers import MarianMTModel, MarianTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from typing import Tuple, Callable, Dict, Any

def load_model_and_tokenizer(translation_model_name: str) -> Tuple[MarianTokenizer, MarianMTModel]:
    '''
    Loads the translation model and associated tokenizer from the translation model's name.

    Args:
        translation_model_name (str): The name of the pre-trained translation model

    Returns:
        tuple:
            - tokenizer (MarianTokenizer): The loaded tokenizer
            - model (MarianMTModel): The loaded translation model
    '''
    tokenizer = MarianTokenizer.from_pretrained(translation_model_name)
    model = MarianMTModel.from_pretrained(translation_model_name)

    return tokenizer, model

def load_and_train_test_split_dataset(dataset_name: str) -> Tuple[Dataset, Dataset]:
    '''
    Loads a dataset using its name and performs a train-test split
    with 80% train and 20% test.

    Args:
        dataset_name (str): The name of the dataset to load

    Returns:
        tuple: The train and test datasets:
            - train_dataset (Dataset): The training dataset
            - test_dataset (Dataset): The test dataset
    '''
    dataset = load_dataset(dataset_name)

    train_test_split = dataset["train"].train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return train_dataset, test_dataset

def translation_tokenize_function(
    tokenizer: MarianTokenizer, examples: Dict[str, Any], source_language: str = "en", target_language: str = "es"
) -> Dict[str, Any]:
    '''
    Tokenizes a translation example using the provided tokenizer.
    Tokenizes the example from the specified source language to the 
    specified target language.

    Args:
        tokenizer (MarianTokenizer): The tokenizer used for encoding the text
        examples (dict): A dictionary containing input texts for tokenization
                         Should contain keys corresponding to the specified
                         source and target languages
        source_language (str): The language code for the source language 
                               (default is 'en' for English)
        target_language (str): The language code for the target language 
                               (default is 'es' for Spanish)

    Returns:
        dict: A dictionary containing the tokenized example.
    '''
    return tokenizer(
        examples[source_language],
        text_target=examples[target_language],
        padding="max_length",
        truncation=True
    )

def tokenize_dataset(
    dataset: Dataset, tokenizer: MarianTokenizer, tokenize_function: Callable[[MarianTokenizer, Dict[str, Any]], Dict[str, Any]]
) -> Dataset:
    '''
    Tokenizes an entire dataset using the specified tokenizer and tokenization function.
    Maps the provided `tokenize_function` across all examples in the dataset.

    Args:
        dataset (Dataset): The dataset containing the raw text examples to be tokenized
        tokenizer (MarianTokenizer): The tokenizer used for encoding the text
        tokenize_function (function): A function that tokenizes an example using
                                        the given tokenizer

    Returns:
        Dataset: The tokenized dataset.
    '''
    tokenized_dataset = dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    return tokenized_dataset

def fine_tune_model_lora(
    translation_model_name: str, dataset_name: str, lora_config: LoraConfig, training_arguments: TrainingArguments
) -> None:
    '''
    Fine-tunes a translation model using the Low-Rank Adaptation method (LoRA).

    Args:
        translation_model_name (str): The name the pre-trained translation model
        dataset_name (str): The name of the dataset to use for fine-tuning
        lora_config (LoraConfig): The configuration for LoRA adaptation
        training_arguments (TrainingArguments): The arguments for the training process
    '''
    tokenizer, model = load_model_and_tokenizer(translation_model_name)
    train_dataset, test_dataset = load_and_train_test_split_dataset(dataset_name)

    tokenized_train = tokenize_dataset(train_dataset, tokenizer, translation_tokenize_function)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, translation_tokenize_function)

    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test
    )

    trainer.train()
    trainer.save_model("./lora_fine_tuned_model")

    print(f"Evaluation results: {trainer.evaluate()}")
