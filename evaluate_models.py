from transformers import TrainingArguments, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType
from datasets import Dataset

from src.config import MAX_LENGTH

from src.utils import (
    load_model_and_tokenizer, load_and_train_test_split_dataset,
    tokenize_dataset, translation_tokenize_function,
    add_task_prefix
)
from src.fine_tuning_utils import fine_tune_model_lora, fine_tune_model_full
from src.evaluation_utils import compute_sacrebleu_score
from src.context import summarize_text

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

if __name__ == "__main__":
    main()