import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils import ENTITIES_LABELS


TRAINING_ACCURACY_SCORES = [
    0.9487630867070832,
    0.9633948523901956,
    0.9717919090541378,
    0.9767809063391818,
    0.9814455072135622,
]
TRAINING_LOSS_SCORES = [
    0.14631907803983413,
    0.1015960768762046,
    0.0771914624302151,
    0.0627643029615624,
    0.05066800184610446,
]
VALIDATION_AVERAGE_LOSS = [
    0.1617323553112025,
    0.11859240872630229,
    0.08350671088167777,
    0.060722047529998234,
    0.05151941516451188,
]

VALIDATION_ACCURACY_SCORES = [
    0.9421135412648011,
    0.9587843563151174,
    0.971996973289879,
    0.9788862623058739,
    0.9812584702441808,
]
VALIDATION_PRECISION_SCORES = [
    0.7553591279807643,
    0.8108237578419841,
    0.8821687560101753,
    0.8931062405696453,
    0.915004582961281,
]
VALIDATION_RECALL_SCORES = [
    0.7185785598638499,
    0.7998469983172798,
    0.8585434193194197,
    0.911076990053173,
    0.9118543134250543,
]


VALIDATION_F1_SCORES = [
    0.7365099336840524,
    0.8052979746409707,
    0.8701957634763239,
    0.9020021155778127,
    0.9134267320014108,
]


# Plot class distribution of the original dataset
def plot_dataset_info():
    data = pd.read_csv("data/NER dataset.csv", encoding="unicode_escape")
    classes = data["Tag"].unique()
    classes_counts = []

    for class_name in classes:
        class_data = data[data.loc[:, "Tag"] == class_name]
        classes_counts.append(len(class_data))

    data["Tag"] = data["Tag"].apply(lambda tag: tag if tag in ENTITIES_LABELS else "O")
    sentence_classes = data["Tag"].unique()
    sentence_classes_counts = []
    sum_of_classes = 0
    for class_name in sentence_classes:
        class_data = data[data.loc[:, "Tag"] == class_name]
        sentence_classes_counts.append(len(class_data))
        if class_name != "O":
            sum_of_classes += len(class_data)

    figure, axis = plt.subplots(1, 3, figsize=(15, 10))
    figure.canvas.set_window_title("Classes distribution")
    figure.suptitle("Classes distribution")
    figure.subplots_adjust(wspace=0.5)
    axis[0].bar(classes, classes_counts)
    axis[0].plot(classes, classes_counts, color="red", marker="o", linestyle="dashdot")
    axis[0].set_ylabel("Count")
    axis[0].set_xlabel("Class")
    axis[0].set_ylim(min(classes_counts), max(classes_counts) + 10000)
    axis[0].set_title("Original dataset classes distribution")
    for tick in axis[0].get_xticklabels():
        tick.set_rotation(45)

    # Plot the class distribution after the classes replacement
    axis[1].bar(sentence_classes, sentence_classes_counts)
    axis[1].set_ylim(min(classes_counts), max(classes_counts) + 30000)
    axis[1].set_title("Classes distribution for specific NER")
    axis[1].set_ylabel("Count")
    axis[1].set_xlabel("Class")
    for tick in axis[1].get_xticklabels():
        tick.set_rotation(45)

    # Compare the class distribution of the "O" label and the rest of the labels
    axis[2].bar("O Label", sentence_classes_counts[0])
    axis[2].bar("Rest of the labels", sum_of_classes, color="#1f77b4")
    axis[2].set_title("Comparison of the 'O' label and the rest of the labels")
    axis[2].set_ylabel("Count")
    axis[2].set_xlabel("Label tag")
    for tick in axis[2].get_xticklabels():
        tick.set_rotation(17)

    plt.show()


# Metrics training and validation scores of the chosen model
def plot_metrics():
    validation_metrics = [
        VALIDATION_ACCURACY_SCORES,
        VALIDATION_RECALL_SCORES,
        VALIDATION_PRECISION_SCORES,
        VALIDATION_F1_SCORES,
    ]

    metrics_names = [
        "Accuracy",
        "Recall",
        "Precision",
        "F1 Score",
    ]

    for index, metric in enumerate(validation_metrics):
        print(f"Mean value for {metrics_names[index]}: {np.round(np.mean(metric), 2)}")
        print(
            f"Standard deviation for {metrics_names[index]}: {np.round(np.std(metric), 2)}"
        )
    # Plot the metrics
    plt.figure("Metrics boxplot")
    plt.boxplot(validation_metrics)
    plt.title("Statistical summary of metrics in validation dataset")
    plt.xticks(range(1, 5), metrics_names, rotation=10)
    plt.ylabel("Percentage score")
    plt.xlabel("Metrics")
    plt.show()


def plot_loss():
    plt.figure("Loss plot")
    plt.plot(TRAINING_LOSS_SCORES, label="Training loss")
    plt.plot(VALIDATION_AVERAGE_LOSS, label="Validation loss")
    plt.xticks(range(5), range(1, 6))
    plt.title("Training and validation loss")
    plt.xlabel("Fold")
    plt.ylabel("Average loss per fold")
    plt.legend()
    plt.show()


plot_dataset_info()
plot_metrics()
plot_loss()
