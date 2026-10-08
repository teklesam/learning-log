"""Ex1: autograd. Compute gradients by hand, then check against PyTorch."""
import torch

# f(w) = (w*x - y)^2 with x=3, y=5, at w=2
x, y = torch.tensor(3.0), torch.tensor(5.0)
w = torch.tensor(2.0, requires_grad=True)

f = (w * x - y) ** 2
# TODO: call the method that computes df/dw
...

# TODO: work out df/dw BY HAND (chain rule) and put the number here
manual_grad = None

print("autograd:", w.grad.item(), "| by hand:", manual_grad)
assert abs(w.grad.item() - manual_grad) < 1e-6, "check your chain rule"

# Bonus: why does this print a DIFFERENT number the second time? (gradient accumulation)
f = (w * x - y) ** 2; f.backward()
print("after a second backward without zeroing:", w.grad.item())
print("✅ Ex1 done")
