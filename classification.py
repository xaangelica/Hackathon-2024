from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def train_classifier(X_train, y_train):
    """
    Train the classifier using training data.

    Parameters:
    - X_train: numpy array of training features.
    - y_train: numpy array of training labels.

    Returns:
    - model: trained classifier.
    """
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    """
    Predict labels for test data using the trained model.

    Parameters:
    - model: trained classifier.
    - X_test: numpy array of test features.

    Returns:
    - y_pred: numpy array of predicted labels.
    """
    y_pred = model.predict(X_test)
    return y_pred

def evaluate_model(y_true, y_pred):
    """
    Evaluate the model's performance.

    Parameters:
    - y_true: numpy array of true labels.
    - y_pred: numpy array of predicted labels.

    Returns:
    - accuracy: float, accuracy score.
    """
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy
