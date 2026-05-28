"""
Parsea TB events del run SAC-GMM (152 demos, open_drawer)
y genera plots PNG de cada métrica.
"""
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import os
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
LOG_DIR = THIS_DIR / "tb_logs"
PLOTS_DIR = THIS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

print(f"Reading from: {LOG_DIR}")
acc = EventAccumulator(str(LOG_DIR))
acc.Reload()

tags = acc.Tags()["scalars"]
print(f"\nMétricas disponibles ({len(tags)}):")
for t in tags:
    print(f"  - {t}")

# Plot one figure per metric + one combined figure for eval vs train return
for tag in tags:
    events = acc.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    plt.figure(figsize=(10, 5))
    plt.plot(steps, values, linewidth=0.8)
    plt.title(f"{tag}  (n={len(values)} points)")
    plt.xlabel("global step")
    plt.ylabel(tag.split("/")[-1])
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    safe = tag.replace("/", "_").replace(" ", "_")
    out = PLOTS_DIR / f"{safe}.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  ✓ {out.name}  (min={min(values):.4f}, max={max(values):.4f}, last={values[-1]:.4f})")

# Combined plot: eval return + train episode return overlaid (if both exist)
def get_scalar(tag):
    if tag not in tags:
        return None, None
    ev = acc.Scalars(tag)
    return [e.step for e in ev], [e.value for e in ev]

train_steps, train_vals = get_scalar("train_episode-return")
eval_steps, eval_vals = get_scalar("eval_episode-avg-return")

if train_steps and eval_steps:
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(train_steps, train_vals, alpha=0.3, color="tab:blue", label="train/episode-return")
    ax1.plot(eval_steps, eval_vals, color="tab:red", linewidth=2, label="eval/episode-avg-return")
    ax1.set_xlabel("global step")
    ax1.set_ylabel("return")
    ax1.set_title("SAC-GMM open_drawer (152 demos) — train vs eval return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    out = PLOTS_DIR / "summary_train_vs_eval.png"
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  ✓ {out.name}  (combined)")

print(f"\nPlots guardados en: {PLOTS_DIR}")
