import numpy as np
import evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset
from typing import List, Dict, Any

from src.config import MAX_LENGTH

def postprocess_texts_for_bleu(texts: List[str]) -> List[str]:
    """
    Post-process a list of texts to remove special tokens (such as padding)
    and extra whitespace.
    This is necessary for BLEU score computation.

    Args:
        texts (List[str]): A list of strings to be processed.

    Returns:
        (List[str]): A list of cleaned strings with special tokens removed.
    """
    texts = [
        text.replace("<pad>", "").replace("</s>", "").strip() for text in texts
    ]
    return texts

def compute_sacrebleu_score(
    predictions: np.ndarray, labels: np.ndarray, tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """
    Computes the SacreBleu score for a list of predicted translations and
    reference translations.
    
    Args:
        predictions (np.ndarray): A list containing the tokenized model output
            logits for predicted translations.

        labels (np.ndarray): A list of tokenized reference translations
            where each translation is already tokenized into token IDs.

        tokenizer (AutoTokenizer): The tokenizer used to decode token ids to text.

    Returns:
        (Dict[str, Any]): A dictionary containing the SacreBLEU metrics.
    """
    decoded_predictions = tokenizer.batch_decode(
        predictions, skip_special_tokens=True, max_length=MAX_LENGTH
    )
    decoded_labels = tokenizer.batch_decode(
        labels, skip_special_tokens=True, max_length=MAX_LENGTH
    )

    decoded_predictions = postprocess_texts_for_bleu(decoded_predictions)
    decoded_labels = postprocess_texts_for_bleu(decoded_labels)

    decoded_labels = [[label] for label in decoded_labels]

    sacrebleu = evaluate.load("sacrebleu")
    return sacrebleu.compute(predictions=decoded_predictions, references=decoded_labels)

def print_qualitative_analysis(
    predictions: np.ndarray, labels: np.ndarray, tokenizer: AutoTokenizer, model_name: str
) -> None:
    """
    Enables qualitative analysis by decoding and printing model predictions alongside their true labels.

    Args:
        predictions (np.ndarray): The predicted token IDs from the model output.

        labels (np.ndarray): The true token IDs (ground truth) for each example.

        tokenizer (AutoTokenizer): The tokenizer used to decode token IDs into text.

        model_name (str): Descriptive model name. Used for printing.
    """
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, max_length=MAX_LENGTH)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, max_length=MAX_LENGTH)

    for i in range(3):
        print(f"{model_name} prediction {i}: {decoded_predictions[i]}\n")
        print(f"{model_name} label {i}: {decoded_labels[i]}\n\n")


def bleu_score(
    model: AutoModelForSeq2SeqLM, 
    tokenizer: AutoTokenizer, 
    tokenized_test: Dataset, 
    performance: Dict[str, Any], 
    model_name: str
) -> Dict[str, Any]:
    """
    Compute and print BLEU score.

    Args:
        model (AutoModelForSeq2SeqLM): Model to compute the score for.

        tokenizer (AutoTokenizer): Tokenizer associated with the model.

        tokenized_test (Dataset): Tokenized test dataset.

        performance (Dict[str, Any]): Performance dictionary to be
            updated with the scores.
        
        model_name (str): Descriptive model name. Used for printing and
            as a key in the performance dictionary.

    Returns:
        (Dict[str, Any]): Updated performance dictionary.
    """
    # Convert to model input
    tokenized_inputs = tokenizer(
        tokenized_test['en'],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )

    tokenized_preds = model.generate(
        **tokenized_inputs,
        return_dict_in_generate=True,
        output_scores=True,
        max_length=MAX_LENGTH
    )
    predictions = tokenized_preds.sequences

    tokenized_labels = tokenizer(
        tokenized_test['es'],
        return_tensors="np",  # Use numpy format for compatibility with labels_np
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH
    )['input_ids']

    print_qualitative_analysis(predictions,tokenized_labels, tokenizer, model_name)

    score = compute_sacrebleu_score(predictions, tokenized_labels, tokenizer=tokenizer)
    performance[f'{model_name}'] = score

    return performance