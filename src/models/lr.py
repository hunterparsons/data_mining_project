from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss, accuracy_score

class LR:
    def __init__(self, X_train_s, y_train, X_val_s, y_val):
        self.X_train_s = X_train_s
        self.y_train = y_train
        self.X_val_s = X_val_s
        self.y_val = y_val

        self.train()

    def get_performance(self):
        predictions = self.get_predictions()
        return {'LogLoss': log_loss(self.y_val, predictions), 'Acc': accuracy_score(self.y_val, predictions > 0.5)}

    def get_predictions(self):
        return self.model.predict_proba(self.X_val_s)[:, 1]

    def train(self):
        self.model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', random_state=42)
        self.model.fit(self.X_train_s, self.y_train)