import numpy as np
import evaluate
from datasets import Dataset, load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments, EvalPrediction
from peft import LoraConfig, get_peft_model
from typing import Tuple, Callable, Dict, Any, List

def load_model_and_tokenizer(model_name: str) -> Tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
    """
    Loads the model and associated tokenizer from the model's name.

    Args:
        model_name (str): The name of the pre-trained model

    Returns:
        tuple:
            - tokenizer (AutoTokenizer): The loaded tokenizer
            - model (AutoModelForSeq2SeqLM): The loaded model
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

def add_task_prefix(
    example: Dict[str, Any], source_language: str = "en",
    task_prefix: str = "Translate the following text from English to Spanish: "
) -> Dict[str, Any]:
    """
    Add a task prefix to the source language text in a dataset example.

    Args:
        example (Dict[str, Any]): A dictionary representing a single example from the dataset
        source_language (str): The key in the example that corresponds to the source language text. Default is "en"
        task_prefix (str): The prefix to be added to the source language text.
                           Default is "Translate the following text from English to Spanish: "

    Returns:
        Dict[str, Any]: The modified example with the task prefix added to the source language text
    """
    example[source_language] = f"{task_prefix} {example[source_language]}"
    return example

def load_and_train_test_split_dataset(dataset_name: str, test_size: float = .2) -> Tuple[Dataset, Dataset]:
    """
    Loads a dataset using its name and performs a train-test split with a specified split.

    Args:
        dataset_name (str): The name of the dataset to load
        test_size (float): Size of the test set (in percent). Default is .2 (20%)

    Returns:
        tuple: The train and test datasets:
            - train_dataset (Dataset): The training dataset
            - test_dataset (Dataset): The test dataset
    """
    dataset = load_dataset(dataset_name)["train"]

    train_test_split = dataset.train_test_split(test_size=test_size, seed=42)
    train_dataset = train_test_split["train"].select(range(5)).map(add_task_prefix)
    test_dataset = train_test_split["test"].select(range(5)).map(add_task_prefix)

    return train_dataset, test_dataset

def translation_tokenize_function(
    tokenizer: AutoTokenizer, examples: Dict[str, Any], max_length: int = 128,
    source_language: str = "en", target_language: str = "es"
) -> Dict[str, Any]:
    """
    Tokenizes a translation example using the provided tokenizer.
    Tokenizes the example from the specified source language to the 
    specified target language.

    Args:
        tokenizer (AutoTokenizer): The tokenizer used for encoding the text
        examples (dict): A dictionary containing input texts for tokenization
                         Should contain keys corresponding to the specified
                         source and target languages
        max_length (int): The maximum token length of tokenized example
        source_language (str): The language code for the source language.
                               Default is 'en' for English
        target_language (str): The language code for the target language 
                               Default is 'es' for Spanish.

    Returns:
        dict: A dictionary containing the tokenized example.
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
    dataset: Dataset, tokenizer: AutoTokenizer, tokenize_function: Callable[[AutoTokenizer, Dict[str, Any]], Dict[str, Any]]
) -> Dataset:
    """
    Tokenizes an entire dataset using the specified tokenizer and tokenization function.
    Maps the provided `tokenize_function` across all examples in the dataset.

    Args:
        dataset (Dataset): The dataset containing the raw text examples to be tokenized
        tokenizer (AutoTokenizer): The tokenizer used for encoding the text
        tokenize_function (function): A function that tokenizes an example using
                                      the given tokenizer

    Returns:
        Dataset: The tokenized dataset.
    """
    tokenized_dataset = dataset.map(lambda x: tokenize_function(tokenizer, x), batched=True)

    return tokenized_dataset

def postprocess_texts_for_bleu(texts: List[str]) -> List[str]:
    """
    Post-process a list of texts to remove special tokens (such as padding) and extra whitespace.
    This is needed for BLEU score computation.

    Args:
        texts (List[str]): A list of strings to be processed

    Returns:
        List[str]: A list of cleaned strings with special tokens removed.
    """
    texts = [text.replace("<pad>", "").replace("</s>", "").strip() for text in texts]
    return texts

def compute_metrics(eval_preds: EvalPrediction, tokenizer: AutoTokenizer):
    """
    Computes BLEU score for the model predictions.

    Args:
        eval_preds (EvalPrediction): The predictions from the model evaluation
        tokenizer (AutoTokenizer): The tokenizer used to decode token ids to text

    Returns:
        dict: A dictionary containing the BLEU score.
    """
    bleu = evaluate.load("sacrebleu")
    preds, labels = eval_preds.predictions, eval_preds.label_ids

    logits = preds[0]
    predictions = np.argmax(logits, axis=-1)

    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_predictions = postprocess_texts_for_bleu(decoded_predictions)
    decoded_labels = postprocess_texts_for_bleu(decoded_labels)

    decoded_labels = [[label] for label in decoded_labels]

    return bleu.compute(predictions=decoded_predictions, references=decoded_labels)

def fine_tune_model_lora(
    model_name: str, dataset_name: str, lora_config: LoraConfig, training_arguments: TrainingArguments
) -> None:
    """
    Fine-tunes a translation model using the Low-Rank Adaptation method (LoRA).

    Args:
        model_name (str): The name the pre-trained translation model
        dataset_name (str): The name of the dataset to use for fine-tuning
        lora_config (LoraConfig): The configuration for LoRA adaptation
        training_arguments (TrainingArguments): The arguments for the training process
    """
    tokenizer, model = load_model_and_tokenizer(model_name)
    train_dataset, test_dataset = load_and_train_test_split_dataset(dataset_name)

    tokenized_train = tokenize_dataset(train_dataset, tokenizer, translation_tokenize_function)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, translation_tokenize_function)

    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    trainer.train()
    trainer.save_model("./lora_fine_tuned_model")

    print(f"Evaluation results: {trainer.evaluate()}")
