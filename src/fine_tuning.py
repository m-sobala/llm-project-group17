import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Trainer, TrainingArguments, EvalPrediction
)
from peft import LoraConfig, TaskType, get_peft_model
from typing import Dict, Any

from src.evaluation import compute_sacrebleu_score

def compute_metrics(eval_preds: EvalPrediction, tokenizer: AutoTokenizer) -> Dict[str, Any]:
    """
    Computes SacreBLEU score for the model predictions.

    Args:
        eval_preds (EvalPrediction): The predictions from the model evaluation.

        tokenizer (AutoTokenizer): The tokenizer used to decode token ids to text.

    Returns:
        (Dict[str, Any]): A dictionary containing the SacreBLEU metrics.
    """
    preds, labels = eval_preds.predictions, eval_preds.label_ids

    logits = preds[0]
    predictions = np.argmax(logits, axis=-1)

    return compute_sacrebleu_score(predictions, labels, tokenizer)

def fine_tune_model_lora(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    eval_dataset: Dataset,
    training_arguments: TrainingArguments,
    lora_config: LoraConfig,
) -> None:
    """
    Fine-tune a translation model using the Low-Rank Adaptation method (LoRA).

    Args:
        model (AutoModelForSeq2SeqLM): The pre-trained model to fine-tune.

        tokenizer (AutoTokenizer): The tokenizer associated with the model.

        dataset (Dataset): The tokenized training dataset to use for fine-tuning.

        eval_dataset (Dataset): The tokenized evaluation dataset to use for fine-tuning.

        training_arguments (TrainingArguments): The arguments for the training process.

        lora_config (LoraConfig): The configuration for LoRA adaptation.
    """
    model = get_peft_model(model, lora_config)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    trainer.train()
    trainer.save_model("./lora_fine_tuned_model")

def fine_tune_model_full(
    model: AutoModelForSeq2SeqLM,
    tokenizer: AutoTokenizer,
    dataset: Dataset,
    eval_dataset: Dataset,
    training_arguments: TrainingArguments
) -> None:
    """
    Fine-tune a translation model. This is a full parameter fine-tuning.

    Args:
        model (AutoModelForSeq2SeqLM): The pre-trained model to fine-tune.

        tokenizer (AutoTokenizer): The tokenizer associated with the model.

        dataset (Dataset): The tokenized training dataset to use for fine-tuning.

        eval_dataset (Dataset): The tokenized evaluation dataset to use for fine-tuning.

        training_arguments (TrainingArguments): The arguments for the training process.
    """
    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda p: compute_metrics(p, tokenizer)
    )

    trainer.train()
    trainer.save_model("./full_fine_tuned_model")

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
        r=16,
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
