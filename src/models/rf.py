from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import log_loss, accuracy_score

class RF:
    def __init__(self, X_train_u, y_train, X_val_u, y_val):
        self.X_train_u = X_train_u
        self.y_train = y_train
        self.X_val_u = X_val_u
        self.y_val = y_val

        self.train()

    def get_performance(self):
        predictions = self.get_predictions()
        return {'LogLoss': log_loss(self.y_val, predictions), 'Acc': accuracy_score(self.y_val, predictions > 0.5)}

    def get_predictions(self):
        return self.model.predict_proba(self.X_val_u)[:, 1]

    def train(self):
        self.model = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=10, random_state=42, n_jobs=-1)
        self.model.fit(self.X_train_u, self.y_train)