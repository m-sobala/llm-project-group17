import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSeq2SeqLM, AutoTokenizer,
    Trainer, TrainingArguments, EvalPrediction
)
from peft import LoraConfig, get_peft_model
from typing import Dict, Any

from src.evaluation_utils import compute_sacrebleu_score

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
