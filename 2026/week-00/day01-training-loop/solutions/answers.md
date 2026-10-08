# Solutions (only after 10 honest minutes)
- **Ex1:** `f.backward()`; df/dw = 2(wx - y)·x = 2(6-5)·3 = **6**. Second backward without zeroing gives 12: gradients **accumulate**, which is why every loop calls `zero_grad()`.
- **Ex2:** `loss.backward()`; `w -= lr * w.grad; b -= lr * b.grad`; `w.grad.zero_(); b.grad.zero_()`. The `no_grad` stops the update itself from becoming part of the graph.
- **Ex3:** `self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, 1))`. In `evaluate`: `model.eval()` + wrap the loop in `with torch.no_grad():` (or decorate the function with `@torch.no_grad()`). Loop: `loss = loss_fn(model(xb), yb); opt.zero_grad(); loss.backward(); opt.step()`. There's no sigmoid in the model because `BCEWithLogitsLoss` applies it internally in a numerically stable way.
- **Ex4:** `m = (p >= lo) & ((p < hi) | (hi == 1.0))`; `ece += (m.sum().item() / len(p)) * abs(conf - obs)`.
