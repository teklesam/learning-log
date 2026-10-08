# Day 1: a PyTorch training loop from scratch (offline, about 60–75 min)

**Goal:** by the end you can write a training loop **without looking anything up**, and explain every line.
Everything runs offline on a laptop CPU (synthetic data, no downloads). Python 3 + `torch` only.

| Block | Time | File |
|---|---|---|
| 1. Autograd by hand | 15 min | `ex1_autograd.py` |
| 2. Linear regression with no `nn`, no `optim` | 15 min | `ex2_manual_loop.py` |
| 3. The standard loop (`nn.Module`, `DataLoader`, `optim`) | 20 min | `ex3_standard_loop.py` |
| 4. Clinical twist: calibration check on your classifier | 15 min | `ex4_calibration.py` |

**Rules:** fill in every `# TODO`. Run each file with `python3 exN_*.py`. The `assert`s tell you whether you got it right.
Only open `solutions/` once you've tried for 10 minutes.

**Afterwards (on the train back):** write `../2026-10-08.md` with *Did / Understood / Didn't understand / Next*, then commit.

## The 5 lines that every training loop is made of
```python
for xb, yb in loader:           # 1. get a batch
    pred = model(xb)            # 2. forward
    loss = loss_fn(pred, yb)    # 3. loss
    opt.zero_grad(); loss.backward()   # 4. clear old grads, backprop
    opt.step()                  # 5. update the weights
```
**Explain to yourself:** why `zero_grad()`? Why `model.eval()` + `torch.no_grad()` at test time?
