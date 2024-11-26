from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)


def print_evaluation_metrics(y_pred, y_test):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))    


def warmup_scheduler(epoch, lr):
    if epoch < 10:
        return lr + 0.001
    else:
        return lr * 0.99