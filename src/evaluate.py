from transformers import TrainingArguments, AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer
from peft import LoraConfig, TaskType
from datasets import Dataset

from utils import (
    load_model_and_tokenizer, load_and_train_test_split_dataset,
    tokenize_dataset, translation_tokenize_function,
    add_task_prefix
)
from fine_tuning_utils import fine_tune_model_lora
from evaluation_utils import compute_sacrebleu_score
from context import summarize_text

def fine_tune_model(model: AutoModelForSeq2SeqLM, 
                    tokenizer: AutoTokenizer, 
                    tokenized_train: Dataset):
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
        save_total_limit=1,
    )

    fine_tune_model_lora(
        model=model,
        tokenizer=tokenizer,
        dataset=tokenized_train,
        lora_config=lora_config,
        training_arguments=training_args
    )

def save_blue_score(
        model: AutoModelForSeq2SeqLM, 
        tokenizer: AutoTokenizer, 
        tokenized_test: Dataset, 
        performance: dict, 
        model_name: str) -> dict:
    tokenized_preds = model.generate(tokenized_test['en'])

    score = compute_sacrebleu_score(tokenized_preds, tokenized_test['es'], tokenizer=tokenizer)
    performance[f'{model_name}'] = score
    return performance

def add_context(tokenizer: AutoTokenizer, tokenized_X: Dataset) -> Dataset:
    new_x = []
    articles = tokenizer.decode(tokenized_X, skip_special_tokens=True)
    for article in articles:
        context = summarize_text(article)
        x = add_task_prefix(article, task_prefix=f"With the context: {context}, Translate the following text from English to Spanish: ")
        new_x.append(x)
    return tokenize_dataset(new_x, tokenizer, translation_tokenize_function)


def main():
    model, tokenizer = load_model_and_tokenizer(
        model_name="google/flan-t5-small"
    )

    train_dataset, test_dataset = load_and_train_test_split_dataset(
        dataset_name="Iker/Document-Translation-en-es"
    )

    tokenized_train = tokenize_dataset(train_dataset, tokenizer, translation_tokenize_function)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, translation_tokenize_function)
    tokenized_test_context = add_context(tokenizer, tokenized_test)

    performance = {}  # A dict keeping track of the blue scores for each model
    performance = save_blue_score(model, tokenizer, tokenized_test, performance, 'Baseline')
    performance = save_blue_score(model, tokenizer, tokenized_test_context, performance, 'Context')
    

    fine_tune_model(model, tokenizer, tokenized_train)
    fine_tuned_model = AutoConfig.from_pretrained("./lora_fine_tuned_model")

    performance = save_blue_score(fine_tuned_model, tokenizer, tokenized_test, performance, 'FineTuning')
    performance = save_blue_score(fine_tuned_model, tokenizer, tokenized_test_context, performance, 'Context_FineTuning')


if __name__ == "__main__":
    main()