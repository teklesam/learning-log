"""Ex4: the clinician's question. When the model says 80%, is it right 80% of the time?
Build a reliability table + Expected Calibration Error (ECE) for the Ex3 model."""
import torch
from ex3_standard_loop import MLP, val_ds   # re-runs Ex3 quickly (or paste your model)

model = MLP(); model.load_state_dict(torch.load("mlp.pt")); model.eval()
X = torch.stack([val_ds[i][0] for i in range(len(val_ds))])
y = torch.stack([val_ds[i][1] for i in range(len(val_ds))])
with torch.no_grad():
    p = torch.sigmoid(model(X))

bins = torch.linspace(0, 1, 11)
ece = 0.0
print(" bin         n   mean_pred  observed")
for lo, hi in zip(bins[:-1], bins[1:]):
    # TODO: mask for predictions in [lo, hi)  (include 1.0 in the last bin)
    m = ...
    if m.sum() == 0: continue
    conf, obs = p[m].mean().item(), y[m].mean().item()
    # TODO: add this bin's contribution to ECE = (n_bin / N) * |conf - obs|
    ece += ...
    print(f"[{lo:.1f},{hi:.1f})  {int(m.sum()):4d}   {conf:.3f}     {obs:.3f}")
print(f"ECE = {ece:.3f}")
print("Reflect: would you trust this model's 0.8 at the bedside? What would you change? (Think temperature scaling.)")
print("✅ Ex4 done")
