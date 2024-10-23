import numpy as np
import evaluate
from transformers import AutoTokenizer
from typing import List, Dict, Any

from src.config import MAX_LENGTH

def postprocess_texts_for_bleu(texts: List[str]) -> List[str]:
    """
    Post-process a list of texts to remove special tokens (such as padding) and extra whitespace.
    This is needed for BLEU score computation.

    Args:
        texts (List[str]): A list of strings to be processed

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
    Computes the SacreBleu score for a list of predicted translations and reference translations.
    
    Args:
        predictions (np.ndarray): A list containing the tokenized model output
            logits for predicted translations.

        labels (np.ndarray): A list of tokenized reference translations
            where each translation is already tokenized into token IDs.

        tokenizer (AutoTokenizer): The tokenizer used to decode token ids to text.

    Returns:
        (Dict[str, Any]): A dictionary containing the SacreBLEU metrics.
    """
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, max_length=MAX_LENGTH)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, max_length=MAX_LENGTH)

    decoded_predictions = postprocess_texts_for_bleu(decoded_predictions)
    decoded_labels = postprocess_texts_for_bleu(decoded_labels)

    decoded_labels = [[label] for label in decoded_labels]

    sacrebleu = evaluate.load("sacrebleu")
    return sacrebleu.compute(predictions=decoded_predictions, references=decoded_labels)
