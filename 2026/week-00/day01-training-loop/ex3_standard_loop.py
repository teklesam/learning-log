"""Ex3: the standard loop. A small MLP classifier on synthetic 'patients'.
Features ~ e.g. age, BP, cholesterol, ... ; label = disease yes/no (non-linear rule)."""
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split
torch.manual_seed(0)

n = 2000
X = torch.randn(n, 6)
logit = 1.5 * X[:, 0] - X[:, 1] + X[:, 2] * X[:, 3] + 0.5 * X[:, 4] ** 2 - 0.5
y = torch.bernoulli(torch.sigmoid(logit)).float()

ds = TensorDataset(X, y)
train_ds, val_ds = random_split(ds, [1600, 400])
train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=256)

class MLP(nn.Module):
    def __init__(self, d_in=6, d_hidden=32):
        super().__init__()
        # TODO: two hidden layers with ReLU, output ONE logit (no sigmoid here. Why?)
        self.net = ...
    def forward(self, x):
        return self.net(x).squeeze(-1)

model = MLP()
loss_fn = nn.BCEWithLogitsLoss()      # takes logits; numerically stable
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def evaluate(dl):
    # TODO: switch to eval mode and disable grad tracking
    ...
    correct, total = 0, 0
    for xb, yb in dl:
        p = torch.sigmoid(model(xb))
        correct += ((p > 0.5).float() == yb).sum().item(); total += len(yb)
    return correct / total

for epoch in range(30):
    model.train()
    for xb, yb in train_dl:
        # TODO: the 5 lines (forward, loss, zero_grad, backward, step)
        ...
    if epoch % 5 == 0:
        print(f"epoch {epoch:2d}  val acc {evaluate(val_dl):.3f}")

acc = evaluate(val_dl)
print("final val acc:", round(acc, 3))
assert acc > 0.70, "should reach >0.70. Is the loop complete?"
torch.save(model.state_dict(), "mlp.pt")
print("✅ Ex3 done (model saved for Ex4)")
