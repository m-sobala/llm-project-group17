from fine_tuning_utils import fine_tune_model_lora
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

def main():
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
        model_name="google/flan-t5-small",
        dataset_name="Iker/Document-Translation-en-es",
        lora_config=lora_config,
        training_arguments=training_args
    )

if __name__ == "__main__":
    main()
