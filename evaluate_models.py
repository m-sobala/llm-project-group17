from transformers import TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType
from datasets import Dataset
import numpy as np
import matplotlib.pyplot as plt

from src.config import MAX_LENGTH, ORDERED_SCORES

from src.utils import (
    load_model_and_tokenizer, load_and_train_test_split_dataset,
    tokenize_dataset, translation_tokenize_function,
    add_task_prefix
)
from src.fine_tuning_utils import fine_tune_model_lora, fine_tune_model_full
from src.evaluation_utils import compute_sacrebleu_score
from src.context import summarize_text


def get_model_scores(performance: dict, ordered_scores: list=ORDERED_SCORES) -> tuple:
    """
    Extract model names and their corresponding BLEU scores from the performance dictionary.

    Args:
        performance (dict): A dictionary containing performance data for different models. 
                            The keys represent model names, and the values contain various 
                            performance metrics, including the 'score' field (BLEU score).
        
        ordered_scores (list, optional): A list of model names specifying the order in which 
                                         to retrieve the BLEU scores. Defaults to ORDERED_SCORES.

    Returns:
        (tuple): A tuple containing:
            - models (List[str]): A list of model names in the order specified by ordered_scores.
            - bleu_scores (List[float]): A list of corresponding BLEU scores for each model.
    """
    models = []
    bleu_scores = []
    for key in ordered_scores:
        models.append(key)
        bleu_scores.append(performance[key]['score'])
    return (models, bleu_scores)

def plot_performance_models(models: list, bleu_scores: list, save_path: str = 'visuals/bleu_scores_plot.png'):
    """
    Plots the BLEU scores of different models in a bar chart and saves the plot as a PNG file.

    Parameters:
    models (list): A list of model names (strings) to be plotted on the x-axis.
    bleu_scores (list): A list of BLEU scores (floats) corresponding to each model, to be plotted on the y-axis.
    save_path (str): Path to save the plot image. Defaults to 'visuals/bleu_scores_plot.png'.
    """
    plt.figure(figsize=(10, 6))
    plt.bar(models, bleu_scores, color='#4169E1')
    plt.xlabel('Models', color='black')
    plt.ylabel('BLEU Score', color='black')
    plt.title('BLEU Scores for Different Models', color='black')
    plt.xticks(rotation=45, color='black')
    plt.yticks(color='black')
    plt.tight_layout()
    
    plt.savefig(save_path)

def print_qualitative_analysis(predictions: np.ndarray, labels: np.ndarray, tokenizer: AutoTokenizer, model_name: str):
    """
    Enables qualitative analysis by decoding and printing model predictions alongside their true labels.

    Args:
        predictions (np.ndarray): The predicted token IDs from the model output.
        labels (np.ndarray): The true token IDs (ground truth) for each example.
        tokenizer (AutoTokenizer): The tokenizer used to decode token IDs into text.
        model_name (str): The string name of the model

    """
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True, max_length=MAX_LENGTH)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, max_length=MAX_LENGTH)

    for i in range(3):
        print(f"{model_name} prediction {i}: {decoded_predictions[i]}")
        print(f"{model_name} label {i}: {decoded_labels[i]}")


def fine_tune_model(
    model: AutoModelForSeq2SeqLM, 
    tokenizer: AutoTokenizer, 
    tokenized_train: Dataset,
    tokenized_test: Dataset,
) -> None:
    """
    Fine tune model using both full and lora methods.

    Args:
        model (AutoModelForSeq2SeqLM): The pre-trained model to fine-tune.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        tokenized_train (Dataset): The tokenized training dataset to use for fine-tuning.
        tokenized_test (Dataset): The tokenized test dataset to use for fine-tuning.
    """
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.4,
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules="all-linear"
    )

    training_args = TrainingArguments(
        output_dir='./results', 
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1
    )

    fine_tune_model_full(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_train,
        eval_dataset=tokenized_test,
        training_arguments=training_args
    )

    fine_tune_model_lora(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_train,
        eval_dataset=tokenized_test,
        training_arguments=training_args,
        lora_config=lora_config
    )

def save_blue_score(
        model: AutoModelForSeq2SeqLM, 
        tokenizer: AutoTokenizer, 
        tokenized_test: Dataset, 
        performance: dict, 
        model_name: str) -> dict:
    
    #Convert to model takable input
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

def add_context(tokenizer: AutoTokenizer, test: Dataset) -> Dataset:
    test = test.map(
        lambda x: add_task_prefix(x, task_prefix=f"With the context: {summarize_text(x['en'])}. Translate the following text from English to Spanish: ")
    )
    return tokenize_dataset(test, tokenizer, translation_tokenize_function)


def main():
    model, tokenizer = load_model_and_tokenizer(
        model_name="google/flan-t5-small"
    )

    train_dataset, test_dataset = load_and_train_test_split_dataset(
        dataset_name="Iker/Document-Translation-en-es"
    )

    context_test_dataset = add_context(tokenizer, test_dataset)
    train_dataset = train_dataset.map(add_task_prefix)
    test_dataset = test_dataset.map(add_task_prefix)
    
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, translation_tokenize_function)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, translation_tokenize_function)
    tokenized_test_context = tokenize_dataset(context_test_dataset, tokenizer, translation_tokenize_function)

    performance = {}  # A dict keeping track of the blue scores for each model
    performance = save_blue_score(model, tokenizer, tokenized_test, performance, 'Baseline')
    performance = save_blue_score(model, tokenizer, tokenized_test_context, performance, 'Context')

    fine_tune_model(model, tokenizer, tokenized_train, tokenized_test)

    fine_tuned_model_full = AutoModelForSeq2SeqLM.from_pretrained("./full_fine_tuned_model")
    performance = save_blue_score(fine_tuned_model_full, tokenizer, tokenized_test, performance, 'FullFineTuning')
    performance = save_blue_score(fine_tuned_model_full, tokenizer, tokenized_test_context, performance, 'Context_FullFineTuning')

    fine_tuned_model_lora = AutoModelForSeq2SeqLM.from_pretrained("./lora_fine_tuned_model")
    performance = save_blue_score(fine_tuned_model_lora, tokenizer, tokenized_test, performance, 'LoraFineTuning')
    performance = save_blue_score(fine_tuned_model_lora, tokenizer, tokenized_test_context, performance, 'Context_LoraFineTuning')

    print(performance)
    models, scores = get_model_scores(performance)
    plot_performance_models(models, scores)

if __name__ == "__main__":
    main()