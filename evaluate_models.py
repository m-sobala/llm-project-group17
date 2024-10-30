from transformers import AutoModelForSeq2SeqLM

from src.fine_tuning import fine_tune_model
from src.evaluation import bleu_score
from src.context import add_context_to_dataset
from src.utils import (
    load_model_and_tokenizer, load_and_train_test_split_dataset,
    tokenize_dataset, translation_tokenize_function,
    add_task_prefix, plot_performance_models
)

def main():
    """The main method of this script."""
    model, tokenizer = load_model_and_tokenizer(
        model_name="google/flan-t5-small"
    )
    train_dataset, test_dataset = load_and_train_test_split_dataset(
        dataset_name="Iker/Document-Translation-en-es"
    )

    context_test_dataset = add_context_to_dataset(test_dataset)
    train_dataset = train_dataset.map(add_task_prefix)
    test_dataset = test_dataset.map(add_task_prefix)
    
    tokenized_train = tokenize_dataset(train_dataset, tokenizer, translation_tokenize_function)
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, translation_tokenize_function)
    tokenized_test_context = tokenize_dataset(context_test_dataset, tokenizer, translation_tokenize_function)

    performance = {}  # A dict keeping track of the blue scores for each model
    performance = bleu_score(model, tokenizer, tokenized_test, performance, 'Baseline')
    performance = bleu_score(model, tokenizer, tokenized_test_context, performance, 'Context')

    fine_tune_model(model, tokenizer, tokenized_train, tokenized_test)

    fine_tuned_model_full = AutoModelForSeq2SeqLM.from_pretrained("./full_fine_tuned_model")
    performance = bleu_score(fine_tuned_model_full, tokenizer, tokenized_test, performance, 'FullFineTuning')
    performance = bleu_score(fine_tuned_model_full, tokenizer, tokenized_test_context, performance, 'Context_FullFineTuning')

    fine_tuned_model_lora = AutoModelForSeq2SeqLM.from_pretrained("./lora_fine_tuned_model")
    performance = bleu_score(fine_tuned_model_lora, tokenizer, tokenized_test, performance, 'LoraFineTuning')
    performance = bleu_score(fine_tuned_model_lora, tokenizer, tokenized_test_context, performance, 'Context_LoraFineTuning')

    print(performance)
    plot_performance_models(performance)

if __name__ == "__main__":
    main()
