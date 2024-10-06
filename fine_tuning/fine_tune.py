from fine_tuning_utils import fine_tune_model_lora
from transformers import TrainingArguments
from peft import LoraConfig, TaskType

def main():
    lora_config = LoraConfig(
        r=16,  # rank
        lora_alpha=32,  # scaling factor
        lora_dropout=0.1,  # dropout rate
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules="all-linear"
    )

    training_args = TrainingArguments(
        output_dir='./results', 
        eval_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=1,
    )

    fine_tune_model_lora(
        translation_model_name="Helsinki-NLP/opus-mt-en-es",
        dataset_name="Iker/Document-Translation-en-es",
        lora_config=lora_config,
        training_arguments=training_args
    )

if __name__ == "__main__":
    main()
