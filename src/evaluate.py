from transformers import TrainingArguments
from peft import LoraConfig, TaskType

from utils import (
    load_model_and_tokenizer, load_and_train_test_split_dataset,
    tokenize_dataset, translation_tokenize_function
)
from fine_tuning_utils import fine_tune_model_lora

def main():
    model, tokenizer = load_model_and_tokenizer(
        model_name="google/flan-t5-small"
    )

    train_dataset, test_dataset = load_and_train_test_split_dataset(
        dataset_name="Iker/Document-Translation-en-es"
    )

    tokenized_train = tokenize_dataset(train_dataset, tokenizer, translation_tokenize_function)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, translation_tokenize_function)

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

if __name__ == "__main__":
    main()