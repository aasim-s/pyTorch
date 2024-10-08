import torch
from torch import nn
from pathlib import Path
import matplotlib.pyplot as plt

steps = {
    1: "data (prepare and load)",
    2: "build model",
    3: "fitting the model (training)",
    4: "making predictions and evaluating a model (inference)",
    5: "saving and loading a model"
}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")


print(f"----{steps[1]}----")

weight = 0.7
bias = 0.3

# create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

print(f"{X[:10]} {y[:10]}")

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

print(f"train and test set length {len(X_train)},{len(X_test)}")


def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=None):
    """Plots training and test data and compares predictions."""

    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    plt.legend(prop={"size": 14})
    plt.show()


print(f"----{steps[2]}----")


class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(
            1, dtype=torch.float), requires_grad=True)
        self.bias = nn.Parameter(torch.randn(
            1, dtype=torch.float), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


torch.manual_seed(42)
model_0 = LinearRegressionModel()
model_0.to(device)

print(list(model_0.parameters()))
print(model_0.state_dict())

with torch.inference_mode():
    y_preds = model_0(X_test)

print(f"Numbers of testing samples: {len(X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"untrained Predicted values:\n{y_preds}")

# plot_predictions(predictions=y_preds)

print(f"----{steps[3]}----")

loss_fn = nn.L1Loss()  # same as MAE loss
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr=0.01)

EPOCHS = 200

train_loss_values = []
test_loss_values = []
epoch_count = []

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

for epoch in range(EPOCHS):
    model_0.train()
    y_pred = model_0(X_train)

    loss = loss_fn(y_pred, y_train)
    optimizer.zero_grad()  # sets the gradients to zero

    loss.backward()
    optimizer.step()

    model_0.eval()  # put model in evaluation mode
    with torch.inference_mode():
        test_pred = model_0(X_test)

        test_loss = loss_fn(test_pred, y_test.type(torch.float))
        # predictions come in float dtype hence make y_test as float

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(
                f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()

# plt.show()
print("The model learned the following values for weights and bias:")
print(model_0.state_dict())
print("\nAnd the original values for weights and bias are:")
print(f"weights: {weight}, bias: {bias}")

print(f"----{steps[4]}----")
model_0.eval()
with torch.inference_mode():
    y_preds = model_0(X_test)

plot_predictions(predictions=y_preds)

print(f"----{steps[5]}----")

print("create Model directory")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

print("create model save path")
MODEL_NAME = "01.1_pyTorch_workflow_fundamentals_model_0.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

print(f"save mode to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)

# Instantiate a new instance of model
# loaded_model_0 = LinearRegressionModel()

# Load the state_dict of our saved model
# loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
