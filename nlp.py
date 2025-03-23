import utils
import tokenizer
import model as md
import torch.nn as nn
from transformers import BertForTokenClassification
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.optim import AdamW
from sklearn.metrics import precision_score, accuracy_score, recall_score

# Stratified K Fold validation with K = 5 splits
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Number of epochs
EPOCHS = 2
# Batch size
BATCH_SIZE = 16
# Maximum length of the tokens
MAX_LENGTH = 512
# Learning rate of the optimizer
LEARNING_RATE = 5e-5


train_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 0}
val_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 0}
test_params = {"batch_size": BATCH_SIZE, "shuffle": False, "num_workers": 0}

data, labels_to_ids, ids_to_labels = utils.preprocess_dataset("data//NER dataset.csv")

# Split data into train and validation and test sets
train_validation_data, test_data = train_test_split(data, test_size=0.2, shuffle=False)
test_data = test_data.reset_index(drop=True)
test_data_tokenizer = tokenizer.Tokenizer(test_data, MAX_LENGTH, labels_to_ids)
test_dataloader = md.torch.utils.data.DataLoader(test_data_tokenizer, **test_params)


# Check if GPU is available if not use CPU
if md.torch.cuda.is_available():
    device = md.torch.device("cuda:0")
else:
    device = md.torch.device("cpu")

# Load the model and assign it to the device
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased", num_labels=len(labels_to_ids.keys())
).to(device)


# Optimizer
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

# Loss function
loss_function = nn.CrossEntropyLoss()
fold_counter = 0
training_average_accuracy = []
training_average_loss = []
validation_average_loss = []
validation_accuracies = []
validation_precisions = []
validation_recall = []
validation_f1_scores = []
for train_indices, validation_indices in skf.split(
    train_validation_data.Sentence, train_validation_data.Sentence_Word_Tags
):
    fold_counter += 1
    print(f"Beginning fold {fold_counter}")
    train_data = train_validation_data.iloc[train_indices]
    train_data = train_data.reset_index(drop=True)
    validation_data = train_validation_data.iloc[validation_indices]
    validation_data = validation_data.reset_index(drop=True)
    train_data_tokenizer = tokenizer.Tokenizer(train_data, MAX_LENGTH, labels_to_ids)
    val_data_tokenizer = tokenizer.Tokenizer(validation_data, MAX_LENGTH, labels_to_ids)

    # Convert train and validation datasets to torch dataloaders
    train_dataloader = md.torch.utils.data.DataLoader(
        train_data_tokenizer, **train_params
    )
    val_dataloader = md.torch.utils.data.DataLoader(val_data_tokenizer, **val_params)

    # Model training
    for epoch in range(0, EPOCHS):
        print(f"Epoch {epoch + 1} training")
        fold_train_accuracy, fold_train_loss = md.train(
            model, train_dataloader, optimizer, device, epoch
        )
        labels, preds, fold_validation_loss = md.evaluate(
            model, val_dataloader, device, ids_to_labels
        )
        precision = precision_score(labels, preds, average="macro")
        accuracy = accuracy_score(labels, preds)
        recall = recall_score(labels, preds, average="macro")
        f1_score = 2 * (precision * recall) / (precision + recall)
    training_average_accuracy.append(fold_train_accuracy)
    training_average_loss.append(fold_train_loss)
    validation_average_loss.append(fold_validation_loss)
    validation_accuracies.append(accuracy)
    validation_precisions.append(precision)
    validation_recall.append(recall)
    validation_f1_scores.append(f1_score)


# Test the model on the test dataset
labels, preds, _ = md.evaluate(model, test_dataloader, device, ids_to_labels, -1)
utils.print_metrics(labels, preds)
