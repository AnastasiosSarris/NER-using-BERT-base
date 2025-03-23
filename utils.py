import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

ENTITIES_LABELS = [
    "B-per",
    "I-per",
    "B-geo",
    "I-geo",
    "B-org",
    "I-org",
    "B-tim",
    "I-tim",
]


def preprocess_dataset(file_path):
    """
    Function that preprocesses the dataset to make it ready for
    the training prossess.
    Parameters:
        file_path (str): The path to the dataset
    Returns:
        data_in_sentences (pd.DataFrame): The dataset grouped by sentences
        labels_to_ids (dict): A dictionary that maps the labels to ids
        ids_to_labels (dict): A dictionary that maps the ids to labels
    """
    # Load the dataframe
    data = pd.read_csv(file_path, encoding="unicode_escape")
    # Rename all the values of the undesired classes to O since
    # in NER it means it does not belong to any entity
    data["Tag"] = data["Tag"].apply(lambda tag: tag if tag in ENTITIES_LABELS else "O")

    # Map the laels to ids and viceversa
    labels_to_ids, ids_to_labels = map_labels_to_ids(data)

    # Remove the Nan values and concat the words in the same sentence
    data = data.fillna(method="ffill")
    # Create two new columns, one with the sentence and the other with the tags
    data["Sentence"] = (
        data[["Sentence #", "Word", "Tag"]]
        .groupby(["Sentence #"])["Word"]
        .transform(lambda x: " ".join(x))
    )
    data["Sentence_Word_Tags"] = (
        data[["Sentence #", "Word", "Tag"]]
        .groupby(["Sentence #"])["Tag"]
        .transform(lambda x: ",".join(x))
    )
    # Create a new dataframe with the sentence number, the sentences and the tags
    data_in_sentences = data[
        ["Sentence #", "Sentence", "Sentence_Word_Tags"]
    ].drop_duplicates()
    data_in_sentences = data_in_sentences.reset_index(drop=True)

    return data_in_sentences, labels_to_ids, ids_to_labels


def map_labels_to_ids(dataset, column="Tag"):
    """
    Function that maps the labels to ids and viceversa
    Parameters:
        dataset (pd.DataFrame): The dataset that contains the labels
        column (str): The name of the column that contains the labels, which is "Tag" by default
    Returns:
        labels_to_ids (dict): A dictionary that maps the labels to ids
        ids_to_labels (dict): A dictionary that maps the ids to labels
    """
    labels_to_ids = {
        label: index for index, label in enumerate(dataset[column].unique())
    }
    ids_to_labels = {
        index: label for index, label in enumerate(dataset[column].unique())
    }
    return labels_to_ids, ids_to_labels


def print_metrics(labels, preds):
    """ "
    Function that prints the evaluation metrics of the model

    Parameters:
    labels: The true labels
    preds: The predicted labels

    """
    confusion_matrix = confusion_matrix(labels, preds)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, average="macro")
    f1_score = 2 * (precision * recall) / (precision + recall)
    print(f"Accuracy score for test dataset: {np.round(accuracy,2)}")
    print(f"Precision score: {np.round(precision,2)}")
    print(f"Recall score: {np.round(recall,2)}")
    print(f"F1 score: {np.round(f1_score,2)}")
