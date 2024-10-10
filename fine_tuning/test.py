from fine_tuning_utils import load_and_train_test_split_dataset
train, test = load_and_train_test_split_dataset("Iker/Document-Translation-en-es")
print(train[0])