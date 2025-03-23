from transformers import BertTokenizer
import numpy as np
import torch
from torch.utils.data import Dataset


class Tokenizer(Dataset):
    """

    Class that implements the tokenization of the dataset using the BERT cased tokenizer.
    Parameters:
    dataset: The dataset to tokenize
    max_length: The maximum length of the tokenized sentences
    labels_to_ids: A dictionary that maps the labels to ids
    Returns:
    A Tokenizer object that contains a dictionary with the tokenized sentences, the attention masks and the labels
    """

    def __init__(self, dataset, max_length, labels_to_ids):
        self.dataset = dataset
        self.len = len(dataset)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        self.max_length = max_length
        self.labels_to_ids = labels_to_ids

    def __getitem__(self, index):
        # Get the sentences from the dataframe and tokenize them
        token_ids = []
        attention_masks = []
        sentence = self.dataset["Sentence"][index]
        sentence_labels = self.dataset["Sentence_Word_Tags"][index]
        batch_encoder = self.tokenizer.encode_plus(
            sentence,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        token_ids.append(batch_encoder["input_ids"])
        attention_masks.append(batch_encoder["attention_mask"])
        # Convert the labels to ids
        tokenized_labels = [
            self.labels_to_ids[label] for label in sentence_labels.split(",")
        ]
        # Create an empty array of -100 of length max_length
        encoded_labels = np.ones(self.max_length, dtype=int) * -100
        # Set the original label of the word to each tokenized wordpiece
        for index, _ in enumerate(tokenized_labels):
            encoded_labels[index] = tokenized_labels[index]

        # Save in a dictionary the tokenized labels, the tokens and the attention masks
        tensors = {
            key: torch.as_tensor(val).squeeze(0) for key, val in batch_encoder.items()
        }
        tensors["labels"] = torch.as_tensor(encoded_labels).squeeze(0)
        return tensors

    def __len__(self):
        return self.len
