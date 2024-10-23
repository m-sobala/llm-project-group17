from src.utils import load_model_and_tokenizer
from src.config import MAX_LENGTH

def summarize_text(
        text: str,
        model : str = "google-t5/t5-small",
        max_length: int = 35,
        min_length: int = 20) -> str:
    """
    Summarizes the input text using a pre-trained sequence-to-sequence model.

    Args:
        text_file (str): name of file with input text that needs to be summarized.
        model (str, default="google-t5/t5-small"): A pre-trained model for text summarization.
        max_length (int, default=35): number representing maximum length of the summary.
        min_length (int, default=20): number representing minimum length of the summary.

    Returns:
        str: A summary (with a length in the specified range) of the input text generated by the model.
    """
    # Read the text from the input file
    # with open(text_file, 'r', encoding='utf-8') as file:
    #     text = file.read()

    # Load the summarization model
    model, tokenizer = load_model_and_tokenizer(model)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary
