import warnings
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accuracy:
    def __init__(self, topk=(1,), task: str = "multi-label", debug: bool = False):
        self.topk = topk
        self.task = task
        self.accuracy_meter_1 = AverageMeter()
        self.balanced_accuracy_meter_1 = AverageMeter()
        self.auc_meter = AverageMeter()
        self.loss_meter = AverageMeter()
        self.debug = debug

    def __call__(self, outputs, targets, loss):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
            # accuracy
            accuracy = getACC(targets, outputs, self.task)
            self.accuracy_meter_1.update(accuracy, targets.shape[0])
            # balanced accuracy
            balanced_accuracy = getBalancedACC(targets, outputs, self.task)
            self.balanced_accuracy_meter_1.update(balanced_accuracy, targets.shape[0])
            # roc auc
            if len(np.unique(targets)) >= 2:
                auc = getAUC(targets, outputs, self.task, debug=self.debug)
                self.auc_meter.update(auc, targets.shape[0])
            else:
                if self.debug:
                    print(
                        "Accuracy -> __call__ -> ROC AUC score is not defined with only one class, skipping this batch.")
            # loss
            self.loss_meter.update(loss.item(), targets.shape[0])
        #
        return self.accuracy_meter_1.avg, self.balanced_accuracy_meter_1.avg, self.auc_meter.avg, self.loss_meter.avg


def getAUC(y_true, y_score, task, debug: bool = False):
    """AUC metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param debug: whatever to print debug logs or not
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()
    print_message_error = ""

    if task == "multi-label, binary-class":
        auc = 0
        for i in range(y_score.shape[1]):
            if len(np.unique(y_true[:, i])) >= 2:
                label_auc = roc_auc_score(y_true[:, i], y_score[:, i])
                auc += label_auc
                print_message_error = "getAUC -> [if] ROC AUC score is not defined with only one class, skipping this batch."
        ret = auc / y_score.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = roc_auc_score(y_true, y_score)
    else:
        auc = 0
        for i in range(y_score.shape[1]):
            y_true_binary = (y_true == i).astype(float)
            y_score_binary = y_score[:, i]
            if len(np.unique(y_true_binary)) >= 2:
                auc += roc_auc_score(y_true_binary, y_score_binary)
            else:
                print_message_error = "getAUC -> [else] ROC AUC score not defined with only one class, skipping this batch."

        ret = auc / y_score.shape[1]

    if debug and print_message_error:
        print(print_message_error)
    return ret


def getACC(y_true, y_score, task, threshold=0.5):
    """Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    return _get_acc(y_true, y_score, task, accuracy_score, threshold)


def getBalancedACC(y_true, y_score, task, threshold=0.5):
    """Balanced Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    return _get_acc(y_true, y_score, task, balanced_accuracy_score, threshold)


def _get_acc(y_true, y_score, task, accuracy_method, threshold=0.5):
    """Accuracy Evaluator.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param accuracy_method: the method to use to evaluate the score
    :param threshold: the threshold for multilabel and binary-class tasks
    """
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    if task == "multi-label, binary-class":
        y_pre = y_score > threshold
        acc = 0
        for label in range(y_true.shape[1]):
            label_acc = accuracy_method(y_true[:, label], y_pre[:, label])
            acc += label_acc
        ret = acc / y_true.shape[1]
    elif task == "binary-class":
        if y_score.ndim == 2:
            y_score = y_score[:, -1]
        else:
            assert y_score.ndim == 1
        ret = accuracy_method(y_true, y_score > threshold)
    else:
        ret = accuracy_method(y_true, np.argmax(y_score, axis=-1))

    return ret


def get_label_distribution(dataset_loader: DataLoader, num_classes: int, task_name: str):
    assert num_classes > 0

    # Inizializza il contatore delle etichette
    class_counts = Counter({i: 0 for i in range(num_classes)})

    # Iteriamo sul dataloader
    for _, labels in dataset_loader:
        match task_name.lower():
            # Caso: binary-class (etichetta 0 o 1)
            case "binary-class":
                if labels.dim() in [1, 2]:  # Singola etichetta binaria (0 o 1)
                    for label in labels:
                        class_counts[label.item()] += 1
                else:
                    raise ValueError("For binary-class task, labels should be 1D (binary labels).")
            # Caso: multi-label (più classi per esempio) binary-class (etichetta 0 o 1)
            case "multi-label, binary-class":
                if labels.dim() > 1:  # Etichette binarie per ogni classe
                    for label_list in labels:
                        for idx, label in enumerate(label_list):
                            if label == 1:
                                class_counts[idx] += 1
                else:
                    raise ValueError("For multi-label binary-class task, labels should be 2+D (binary labels).")
            # Caso: multi-class (una classe per esempio)
            case "multi-class":
                if labels.dim() == 2:  # Caso con tensor 2D (ad esempio shape [N, 1])
                    squeezed_labels = labels.squeeze()
                    if squeezed_labels.dim() == 1:
                        for label in squeezed_labels:  # Squeeze per ottenere un tensor 1D
                            class_counts[label.item()] += 1
                elif labels.dim() == 1:  # Caso con tensor 1D (se ogni esempio è già un singolo valore)
                    for label in labels:
                        class_counts[label.item()] += 1
                else:
                    raise ValueError("For multi-class task, labels should be 1D or 2D (single class per example).")
            # Caso: ordinal-regression (valori numerici, es. da 1 a 4)
            case "ordinal-regression" | "mnist" | "fashionmnist":
                if labels.dim() in [1, 2]:  # Etichetta numerica per esempio
                    for label in labels:
                        class_counts[label.item()] += 1
                else:
                    raise ValueError("For ordinal-regression and mnist tasks, labels should be 1D (numeric labels).")
            case _:
                raise ValueError(f"Unknown task name {task_name}")

    # Calcoliamo il numero totale di elementi
    total_elements = sum(class_counts.values()) + 1

    # Normalizziamo la distribuzione dividendo per il totale
    class_distribution_normalized = [class_counts[i] / total_elements for i in range(num_classes)]

    return class_distribution_normalized