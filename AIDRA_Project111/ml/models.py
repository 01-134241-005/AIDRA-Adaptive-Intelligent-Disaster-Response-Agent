
# ml/models.py
import random
import math

# =========================
# SIMPLE KNN CLASSIFIER
# =========================
class SimpleKNN:
    def __init__(self, k=3):
        self.k = k
        self.X = []
        self.y = []

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X_test):
        preds = []
        for xt in X_test:
            distances = []
            for i, xi in enumerate(self.X):
                d = math.sqrt(sum((xt[j] - xi[j])**2 for j in range(len(xt))))
                distances.append((d, self.y[i]))
            distances.sort(key=lambda x: x[0])
            k_nearest = distances[:self.k]
            votes = {}
            for _, label in k_nearest:
                votes[label] = votes.get(label, 0) + 1
            preds.append(max(votes, key=votes.get))
        return preds

# =========================
# SIMPLE NAIVE BAYES (Gaussian)
# =========================
class SimpleNaiveBayes:
    def fit(self, X, y):
        self.classes = list(set(y))
        self.class_priors = {}
        self.mean = {}
        self.var = {}
        for c in self.classes:
            X_c = [X[i] for i in range(len(X)) if y[i] == c]
            self.class_priors[c] = len(X_c) / len(X)
            self.mean[c] = []
            self.var[c] = []
            for j in range(len(X[0])):
                col = [row[j] for row in X_c]
                mean = sum(col) / len(col)
                var = sum((x - mean)**2 for x in col) / len(col)
                self.mean[c].append(mean)
                self.var[c].append(var if var > 0 else 1e-6)

    def gaussian_pdf(self, x, mean, var):
        exponent = math.exp(-((x - mean)**2) / (2 * var))
        return (1 / math.sqrt(2 * math.pi * var)) * exponent

    def predict(self, X_test):
        preds = []
        for xt in X_test:
            probs = {}
            for c in self.classes:
                prior = self.class_priors[c]
                likelihood = 1.0
                for j in range(len(xt)):
                    likelihood *= self.gaussian_pdf(xt[j], self.mean[c][j], self.var[c][j])
                probs[c] = prior * likelihood
            preds.append(max(probs, key=probs.get))
        return preds

# =========================
# EVALUATION METRICS
# =========================
def confusion_matrix(y_true, y_pred, classes):
    """Returns a dict-of-dict: matrix[actual][predicted] = count"""
    matrix = {c: {c2: 0 for c2 in classes} for c in classes}
    for t, p in zip(y_true, y_pred):
        if t in matrix and p in matrix[t]:
            matrix[t][p] += 1
    return matrix

def precision_recall_f1(y_true, y_pred, classes):
    """Per-class and macro-averaged Precision, Recall, F1."""
    cm = confusion_matrix(y_true, y_pred, classes)
    per_class = {}
    for c in classes:
        tp = cm[c][c]
        fp = sum(cm[other][c] for other in classes if other != c)
        fn = sum(cm[c][other] for other in classes if other != c)
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1}
    macro_p  = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_r  = sum(v["recall"]    for v in per_class.values()) / len(classes)
    macro_f1 = sum(v["f1"]        for v in per_class.values()) / len(classes)
    return {
        "per_class": per_class,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "confusion_matrix": cm
    }

def evaluate_model(model, X_test, y_test, classes):
    preds = model.predict(X_test)
    acc = sum(1 for t, p in zip(y_test, preds) if t == p) / len(y_test)
    report = precision_recall_f1(y_test, preds, classes)
    report["accuracy"] = acc
    return report

# =========================
# SYNTHETIC DATA GENERATION (200 per class)
# =========================
def generate_balanced_dataset():
    data = []
    for severity in ['Critical', 'Moderate', 'Minor']:
        for _ in range(200):
            if severity == 'Critical':
                dist = random.uniform(5, 10)
                hazard = random.choice([0, 1])
                injury = random.randint(4, 5)
                hr = random.randint(110, 140)
            elif severity == 'Moderate':
                dist = random.uniform(2, 7)
                hazard = random.choice([0, 1])
                injury = random.randint(2, 4)
                hr = random.randint(85, 115)
            else:
                dist = random.uniform(0, 4)
                hazard = 0
                injury = random.randint(1, 3)
                hr = random.randint(60, 95)
            data.append([dist, hazard, injury, hr, severity])
    random.shuffle(data)
    return data

# =========================
# TRAIN MODELS — returns rich dict
# =========================
CLASSES = ['Critical', 'Moderate', 'Minor']

def train_models():
    dataset = generate_balanced_dataset()
    X = [row[:4] for row in dataset]
    y = [row[4]  for row in dataset]

    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    knn = SimpleKNN(k=3)
    nb  = SimpleNaiveBayes()
    knn.fit(X_train, y_train)
    nb.fit(X_train,  y_train)

    knn_report = evaluate_model(knn, X_test, y_test, CLASSES)
    nb_report  = evaluate_model(nb,  X_test, y_test, CLASSES)

    best_name  = "KNN" if knn_report["accuracy"] >= nb_report["accuracy"] else "NaiveBayes"
    best_model = knn   if best_name == "KNN" else nb

    print(f"[ML] KNN  — Acc:{knn_report['accuracy']:.3f}  MacroF1:{knn_report['macro_f1']:.3f}")
    print(f"[ML] NB   — Acc:{nb_report['accuracy']:.3f}  MacroF1:{nb_report['macro_f1']:.3f}")
    print(f"[ML] Best model: {best_name}")

    return {
        "knn":       knn_report,
        "nb":        nb_report,
        "best_model_name": best_name,
        "best_model": best_model
    }

# =========================
# PREDICT SEVERITY FOR A VICTIM BASED ON LOCATION
# =========================
def predict_severity(model, victim_location, grid):
    x, y = victim_location
    dist   = abs(x) + abs(y)
    hazard = 1 if grid[x][y] == 'R' else 0
    if dist >= 4 or hazard:
        injury_sev = 5
        heart_rate = 130
    elif dist >= 2:
        injury_sev = 3
        heart_rate = 100
    else:
        injury_sev = 1
        heart_rate = 70
    return model.predict([[dist, hazard, injury_sev, heart_rate]])[0]