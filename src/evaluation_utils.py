import numpy as np
import evaluate

def compute_bleu_score(preds: np.ndarray, labels: np.ndarray, tokenizer: AutoTokenizer) -> dict:
    """Computes the BLEU score for a list of predicted translations and reference translations.
    
    Args:
        preds (list): A list containing the tokenized model output logits for predicted translations.
                      The first element is assumed to be the logits matrix.
        labels (list): A list of tokenized reference translations where each translation 
                       is already tokenized into label IDs.
        tokenizer (AutoTokenizer): The tokenizer used to decode token ids to text
                       
    Returns:
        dict: A dictionary containing the BLEU score.
    """
    bleu = evaluate.load("sacrebleu")
    
    logits = preds[0]
    preds = np.argmax(logits, axis=-1)

    decoded_predictions = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_predictions = postprocess_texts_for_bleu(decoded_predictions)
    decoded_labels = postprocess_texts_for_bleu(decoded_labels)

    decoded_labels = [[label] for label in decoded_labels]

    return bleu.compute(predictions=decoded_predictions, references=decoded_labels)