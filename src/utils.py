from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from typing import Tuple, Callable, Dict, Any

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
    train_dataset = train_test_split["train"].select(range(5))
    test_dataset = train_test_split["test"].select(range(5))

    return train_dataset, test_dataset

def translation_tokenize_function(
    tokenizer: AutoTokenizer,
    examples: Dict[str, Any],
    max_length: int = 128,
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
        padding="max_length",
        truncation=True,
        max_length=max_length
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
