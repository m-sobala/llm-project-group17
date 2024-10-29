from pathlib import Path
import matplotlib.pyplot as plt
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import List, Tuple, Dict, Callable, Optional, Any

from src.config import MAX_LENGTH, ORDERED_SCORES

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Loads the model and associated tokenizer from the model's name.

    Args:
        model_name (str): The name of the pre-trained model.

    Returns:
        (Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]):
            - (AutoTokenizer): The loaded tokenizer.

            - (AutoModelForSeq2SeqLM): The loaded model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return model, tokenizer

def add_task_prefix(
    example: Dict[str, Any],
    source_language: str = "en",
    task_prefix: str = "Translate the following text from English to Spanish: "
) -> Dict[str, Any]:
    """
    Add a task prefix to the source language text in a dataset example.

    Args:
        example (Dict[str, Any]): A dictionary representing a single example from the dataset.

        source_language (str, default="en"): The key in the example that corresponds
            to the source language text.

        task_prefix (str, default="Translate the following text from English to Spanish: "):
            The prefix to be added to the source language text.

    Returns:
        (Dict[str, Any]): The modified example with the task prefix added to the source language text.
    """
    example[source_language] = f"{task_prefix} {example[source_language]}"
    return example

def load_and_train_test_split_dataset(dataset_name: str, test_size: float = .2) -> Tuple[Dataset, Dataset]:
    """
    Loads a dataset using its name and performs a train-test split with a specified split.

    Args:
        dataset_name (str): The name of the dataset to load.

        test_size (float, default=.2): Size of the test set (in percent).

    Returns:
        (Tuple[Dataset, Dataset]):
            - (Dataset): The training dataset.

            - (Dataset): The test dataset.
    """
    dataset = load_dataset(dataset_name)["train"]

    train_test_split = dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]

    return train_dataset, test_dataset

def translation_tokenize_function(
    tokenizer: AutoTokenizer,
    examples: Dict[str, Any],
    source_language: str = "en",
    target_language: str = "es"
) -> Dict[str, Any]:
    """
    Tokenizes a translation example using the provided tokenizer.
    Tokenizes the example from the specified source language to the 
    specified target language.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used for encoding the text.

        examples (Dict[str, Any]): A dictionary containing input
            texts for tokenization. Should contain keys corresponding
            to the specified source and target languages.

        max_length (int, default=128): The maximum token length of the
            tokenized example.

        source_language (str, default="en"): The language code
            for the source language.

        target_language (str, default="es"): The language code
            for the target language.

    Returns:
        (Dict[str, Any]): A dictionary containing the tokenized example.
    """
    tokenized = tokenizer(
        examples[source_language],
        text_target=examples[target_language],
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    return tokenized

def tokenize_dataset(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    tokenize_function: Callable[[AutoTokenizer, Dict[str, Any]], Dict[str, Any]]
) -> Dataset:
    """
    Tokenizes an entire dataset using the specified tokenizer and tokenization function.
    Maps the provided `tokenize_function` across all examples in the dataset.

    Args:
        dataset (Dataset): The dataset containing the raw text examples to be tokenized.

        tokenizer (AutoTokenizer): The tokenizer used for encoding the text.

        tokenize_function (Callable[[AutoTokenizer, Dict[str, Any]], Dict[str, Any]]):
            A function that tokenizes an example using the given tokenizer.

    Returns:
        (Dataset): The tokenized dataset.
    """
    tokenized_dataset = dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    return tokenized_dataset

def get_model_scores(
    performance: Dict[str, Any], ordered_scores: Optional[List[str]] = ORDERED_SCORES
) -> Tuple[List[str], List[float]]:
    """
    Extract model names and their corresponding BLEU scores from the performance dictionary.

    Args:
        performance (Dict[str, Any]): A performance dictionary for different models. 
        
        ordered_scores (Optional[List[str]], default=ORDERED_SCORES): A list of model
            names specifying the order in which to retrieve the BLEU scores.

    Returns:
        (Tuple[List[str], List[float]])
            - A list of model names in the order specified by ordered_scores.
            - A list of corresponding BLEU scores for each model.
    """
    models = []
    bleu_scores = []

    for key in ordered_scores:
        models.append(key)
        bleu_scores.append(performance[key]['score'])

    return (models, bleu_scores)

def plot_performance_models(
    performance: Dict[str, Any], ordered_scores: Optional[List[str]] = ORDERED_SCORES, save_path: Path = "bleu_scores_plot.png"
) -> None:
    """
    Plots the BLEU scores of different models in a bar chart and saves the plot as a PNG file.

    Args:
        performance (Dict[str, Any]): A performance dictionary for different models. 
        
        ordered_scores (Optional[List[str]], default=ORDERED_SCORES): A list of model
            names specifying the order in which to retrieve the BLEU scores.
    
        save_path (Path, default="bleu_scores_plot.png"): Path to save the plot image.
    """
    model_names, bleu_scores = get_model_scores(performance, ordered_scores)

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, bleu_scores, color="#4169E1")
    plt.xlabel("Models", color="black")
    plt.ylabel("BLEU Score", color="black")
    plt.title("BLEU Scores for Different Models", color="black")
    plt.xticks(rotation=45, color="black")
    plt.yticks(color="black")
    plt.tight_layout()
    
    plt.savefig(save_path)
