# train.py
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

kim_images = [f"Dataset/KimJongUn/K{i}.jpg" for i in range(1, 51)]
not_kim_images = [f"Dataset/NotKimJongUn/NK{i}.jpg" for i in range(1, 51)]

def load_dataset(image_size=(64, 64)):
    X = []
    y = []

    # label 1 = Kim Jong Un
    for path in kim_images:
        img = Image.open(path).convert("L").resize(image_size)
        arr = np.asarray(img, dtype=np.float32)/255.0
        X.append(arr.flatten())
        y.append(1)

    # label 0 = Not Kim Jong Un
    for path in not_kim_images:
        img = Image.open(path).convert("L").resize(image_size)
        arr = np.asarray(img, dtype=np.float32)/255.0
        X.append(arr.flatten())
        y.append(0)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y

def train_val_test_split(X, y, seed=2):
    np.random.seed(seed)

    # get indices of each class
    idx_kim = np.where(y == 1)[0]
    idx_not = np.where(y == 0)[0]

    # shuffle each class separately
    np.random.shuffle(idx_kim)
    np.random.shuffle(idx_not)

    # helper function for splitting
    def split_indices(indices):
        n = len(indices)
        n_train = int(0.6 * n)
        n_val = int(0.2 * n)
        return (
            indices[:n_train],
            indices[n_train:n_train+n_val],
            indices[n_train+n_val:]
        )

    # stratified splits
    kim_train, kim_val, kim_test = split_indices(idx_kim)
    not_train, not_val, not_test = split_indices(idx_not)

    # combine
    train_idx = np.concatenate([kim_train, not_train])
    val_idx   = np.concatenate([kim_val, not_val])
    test_idx  = np.concatenate([kim_test, not_test])

    # shuffle inside each split
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx]
    )

def plot_all_confusion_matrices(train_cm, val_cm, test_cm):
    cms = [train_cm, val_cm, test_cm]
    titles = ["Train", "Validation", "Test"]

    fig, axes = plt.subplots(1, 3, figsize=(12,4))

    for ax, cm, title in zip(axes, cms, titles):
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_title(f"{title} Confusion Matrix")
        ax.set_xticks([0,1])
        ax.set_yticks([0,1])
        ax.set_xticklabels(["Pred 0", "Pred 1"])
        ax.set_yticklabels(["True 0", "True 1"])

        # annotate values
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

        fig.colorbar(im, ax=ax)   # <-- IMPORTANT: colorbar per subplot

    plt.tight_layout()
    plt.show()


SEED = 2
np.random.seed(SEED)

# ---------- hyperparameters ----------
IMAGE_SIZE = (64, 64)
INPUT_DIM = IMAGE_SIZE[0] * IMAGE_SIZE[1]  # 4096
HIDDEN_DIM = 128
LR = 0.1
LAMBDA = 0.001
EPOCHS = 1500
BATCH_SIZE = 16
MODEL_PATH = "model.npz"
# -------------------------------------

# activation function
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv(a):
    return a * (1 - a)

# loss function
def binary_cross_entropy(y_true, y_pred, eps=1e-12):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# metrics
def predict_labels(X, params):
    W1, b1, W2, b2 = params
    z1 = X.dot(W1) + b1
    a1 = sigmoid(z1)
    z2 = a1.dot(W2) + b2
    y_hat = sigmoid(z2).reshape(-1)
    labels = (y_hat >= 0.5).astype(np.int32)
    return labels, y_hat

def mean_absolute_error(y_true, y_pred_labels):
    return np.mean(np.abs(y_true - y_pred_labels))

def accuracy(y_true, y_pred_labels):
    return np.mean(y_true == y_pred_labels)

def confusion_matrix(y_true, y_pred_labels):
    tp = np.sum((y_true == 1) & (y_pred_labels == 1))
    tn = np.sum((y_true == 0) & (y_pred_labels == 0))
    fp = np.sum((y_true == 0) & (y_pred_labels == 1))
    fn = np.sum((y_true == 1) & (y_pred_labels == 0))
    return np.array([[tn, fp], [fn, tp]])

# ---------- load data ----------
X, y = load_dataset(image_size=IMAGE_SIZE)
print("Loaded dataset:", X.shape, y.shape)

X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, seed=SEED)
print("Split sizes -> train:", len(X_train), "val:", len(X_val), "test:", len(X_test))

# ---------- initialize parameters ----------
def init_params(input_dim, hidden_dim):
    W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros((1, hidden_dim), dtype=np.float32)
    W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros((1, 1), dtype=np.float32)
    return W1.astype(np.float32), b1.astype(np.float32), W2.astype(np.float32), b2.astype(np.float32)

W1, b1, W2, b2 = init_params(INPUT_DIM, HIDDEN_DIM)

# ---------- training loop ----------
best_val_loss = np.inf
best_params = None

train_loss_history = []
val_loss_history = []
train_acc_history = []
val_acc_history = []

n_train = X_train.shape[0]
for epoch in range(1, EPOCHS + 1):
    perm = np.random.permutation(n_train)
    X_train_shuf = X_train[perm]
    y_train_shuf = y_train[perm].reshape(-1, 1)

    for i in range(0, n_train, BATCH_SIZE):
        X_batch = X_train_shuf[i:i+BATCH_SIZE]
        y_batch = y_train_shuf[i:i+BATCH_SIZE]
        m = X_batch.shape[0]

        # forward
        z1 = X_batch.dot(W1) + b1
        a1 = sigmoid(z1)
        z2 = a1.dot(W2) + b2
        y_hat = sigmoid(z2)

        # backward
        dz2 = (y_hat - y_batch) / m
        dW2 = a1.T.dot(dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)

        da1 = dz2.dot(W2.T)
        dz1 = da1 * sigmoid_deriv(a1)
        dW1 = X_batch.T.dot(dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # L2 regularization
        dW2 += LAMBDA * W2
        dW1 += LAMBDA * W1

        # gradient descent
        W2 -= LR * dW2
        b2 -= LR * db2
        W1 -= LR * dW1
        b1 -= LR * db1

    # --- end epoch: compute losses ---
    _, train_probs = predict_labels(X_train, (W1, b1, W2, b2))
    train_loss = binary_cross_entropy(y_train, train_probs)
    _, val_probs = predict_labels(X_val, (W1, b1, W2, b2))
    val_loss = binary_cross_entropy(y_val, val_probs)

    train_loss_history.append(train_loss)
    val_loss_history.append(val_loss)
    train_acc_history.append(accuracy(y_train, (train_probs >= 0.5).astype(int)))
    val_acc_history.append(accuracy(y_val, (val_probs >= 0.5).astype(int)))

    if epoch % 50 == 0 or epoch == 1:
        print(f"Epoch {epoch:04d}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

# ---------- save best model ----------
if best_params is None:
    best_params = (W1, b1, W2, b2)

np.savez(MODEL_PATH,
         W1=best_params[0], b1=best_params[1],
         W2=best_params[2], b2=best_params[3])
print("Saved best model to", MODEL_PATH)

# ---------- evaluation ----------
def evaluate_set(Xs, ys, params, name="set"):
    labels, probs = predict_labels(Xs, params)
    mae = mean_absolute_error(ys, labels)
    acc = accuracy(ys, labels)
    cm = confusion_matrix(ys, labels)
    loss = binary_cross_entropy(ys, probs)
    return {"loss": loss, "acc": acc, "mae": mae, "cm": cm}

print("\n--- Evaluation of BEST model on all splits ---")
train_metrics = evaluate_set(X_train, y_train, best_params, name="Train")
val_metrics   = evaluate_set(X_val, y_val, best_params, name="Validation")
test_metrics  = evaluate_set(X_test, y_test, best_params, name="Test")

# --- Plot confusion matrices ---
plot_all_confusion_matrices(train_metrics["cm"], val_metrics["cm"], test_metrics["cm"])

print("\nMean errors (MAE) and Accuracy:")
print(f" Train: MAE={train_metrics['mae']:.4f}, Acc={train_metrics['acc']:.4f}")
print(f" Val  : MAE={val_metrics['mae']:.4f}, Acc={val_metrics['acc']:.4f}")
print(f" Test : MAE={test_metrics['mae']:.4f}, Acc={test_metrics['acc']:.4f}")

# ---------- plot training curves ----------
fig, axes = plt.subplots(1, 2, figsize=(12,4))

# --- Loss subplot ---
axes[0].plot(train_loss_history, label="Train Loss")
axes[0].plot(val_loss_history, label="Validation Loss")
axes[0].set_title("Loss vs Epochs")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Loss")
axes[0].legend()

# --- Accuracy subplot ---
axes[1].plot(train_acc_history, label="Train Accuracy")
axes[1].plot(val_acc_history, label="Validation Accuracy")
axes[1].set_title("Accuracy vs Epochs")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.show()




