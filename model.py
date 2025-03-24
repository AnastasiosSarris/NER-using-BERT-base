import torch
from sklearn.metrics import accuracy_score
from torch.nn.utils import clip_grad_norm_


def train(model, train_dataloader, optimizer, device, epoch):
    """
    Function to train the model
    Parameters:
     model: The model to train
     train_dataloader: The dataloader of the training data
     optimizer: The optimizer to use
     device: The device to assign the data
     epoch: The epoch number
     Returns:
     average_train_accuracy: The average accuracy of the training
     average_train_loss: The average loss of the training

    """

    # Set the model to train mode
    model.train()
    training_accuracy, training_loss = 0, 0
    training_steps = 0
    training_labels, training_predictions = [], []

    for _, batch in enumerate(train_dataloader):
        # Load each key of the tokenizer vocabulary and assign in to the device
        # in the type of long int
        batch_token_ids = batch["input_ids"].to(device, dtype=torch.long)
        batch_attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
        batch_labels = batch["labels"].to(device, dtype=torch.long)
        # Calculate the loss of the model and the matrix of each classes probabilities
        loss, logits = model(
            input_ids=batch_token_ids,
            attention_mask=batch_attention_mask,
            labels=batch_labels,
            token_type_ids=None,
            return_dict=False,
        )

        training_loss += loss.item()
        training_steps += 1

        if training_steps % 100 == 0:
            print(
                f"Training loss pres 100 training steps:{training_loss/training_steps}"
            )

        # Flatten the labels and the predictions
        flattened_labels = batch_labels.view(-1)
        active_logits = logits.view(-1, model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)

        # Select only the labels that are recognizable (not -100)
        recognizable_labels = flattened_labels != -100

        # Pick the labels of interest and the predictions of them
        labels = torch.masked_select(flattened_labels, recognizable_labels)
        predictions = torch.masked_select(flattened_predictions, recognizable_labels)

        training_labels.extend(labels)
        training_predictions.extend(predictions)
        training_accuracy += accuracy_score(
            labels.cpu().numpy(), predictions.cpu().numpy()
        )
        # Scale down the gradients to ensure a more stable training
        clip_grad_norm_(model.parameters(), 10)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    average_train_loss = training_loss / len(train_dataloader)
    average_train_accuracy = training_accuracy / len(train_dataloader)
    print(f"Epoch {epoch + 1} training loss: {average_train_loss}")
    print(f"Epoch {epoch + 1} training accuracy: {average_train_accuracy}")
    return average_train_accuracy, average_train_loss


def evaluate(model, val_dataloader, device, ids_to_labels, epoch_num):
    """

    Function to evaluate the model and make predictions
    Parameters:
    model: The model to evaluate
    val_dataloader: The dataloader of the validation data
    device: The device to assign the data
    ids_to_labels: The dictionary that maps the label id to label tag

    Returns:
    labels: The labels of the validation data
    predictions: The predictions of the validation data
    validation_loss: The loss of the validation data

    """

    # Set the model to evaluation mode
    model.eval()
    validating_loss = 0
    validating_accuracy = 0
    validating_steps = 0
    validation_labels, validation_predictions = [], []
    # Disable the gradient computation
    with torch.no_grad():
        for batch in val_dataloader:
            # Load each key of the tokenizer vocabulary and assign in to the device
            # in the type of long int
            batch_token_ids = batch["input_ids"].to(device, dtype=torch.long)
            batch_attention_mask = batch["attention_mask"].to(device, dtype=torch.long)
            batch_labels = batch["labels"].to(device, dtype=torch.long)
            (loss, logits) = model(
                batch_token_ids,
                attention_mask=batch_attention_mask,
                labels=batch_labels,
                token_type_ids=None,
                return_dict=False,
            )

            validating_loss += loss.item()
            validating_steps += 1

            if validating_steps % 100 == 0:
                print(
                    f"Validation loss pres 100 validation steps:{validating_loss/validating_steps}"
                )

            # Flatten the labels and the predictions
            flattened_labels = batch_labels.view(-1)
            active_logits = logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)

            # Select only the labels that are recognizable (not -100)
            active_accuracy = flattened_labels != -100

            # Pick the labels of interest and the predictions of them
            labels = torch.masked_select(flattened_labels, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            validation_labels.extend(labels)
            validation_predictions.extend(predictions)

            # Compute the validation accuracy of the model
            validating_accuracy += accuracy_score(
                labels.cpu().numpy(), predictions.cpu().numpy()
            )

        labels = [ids_to_labels[id.item()] for id in validation_labels]
        predictions = [ids_to_labels[id.item()] for id in validation_predictions]
        validation_loss = validating_loss / len(val_dataloader)
        validation_accuracy = validating_accuracy / len(val_dataloader)
        if epoch_num + 1 != 0:
            print(f"Validation loss: {validation_loss}")
            print(f"Validation accuracy: {validation_accuracy}")
        return labels, predictions, validation_loss
