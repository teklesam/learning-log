"""Ex2: fit y = 2x + 1 with plain tensors. No nn, no optim. You ARE the optimiser."""
import torch
torch.manual_seed(0)

X = torch.randn(200, 1)
y = 2 * X + 1 + 0.1 * torch.randn(200, 1)

w = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)
lr = 0.1

for epoch in range(100):
    pred = X * w + b
    loss = ((pred - y) ** 2).mean()
    # TODO: backprop
    ...
    with torch.no_grad():   # updates must not be tracked by autograd. Why?
        # TODO: gradient-descent update for w and b
        ...
        # TODO: zero the gradients of w and b
        ...
    if epoch % 20 == 0:
        print(f"epoch {epoch:3d}  loss {loss.item():.4f}  w {w.item():.3f}  b {b.item():.3f}")

assert abs(w.item() - 2) < 0.05 and abs(b.item() - 1) < 0.05, "not converged. Check the update/zeroing"
print("✅ Ex2 done")
