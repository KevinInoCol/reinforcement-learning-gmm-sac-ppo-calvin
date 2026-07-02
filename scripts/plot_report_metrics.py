"""
Genera las curvas para el informe del proyecto a partir de Weights & Biases
(vía wandb.Api(), NO en tiempo real: refleja lo último sincronizado a W&B).

Las 5 métricas pedidas, para GMM+SAC y GMM+PPO (mismo project unificado,
distinguidos por `group`):

    1. Recompensa acumulada por episodio   (train)
    2. Recompensa media de evaluación       (eval)
    3. Policy / Actor loss
    4. Value / Critic loss
    5. Entropy loss   (PPO: literal;  SAC: loss_alpha = pérdida de temperatura α)

Cada métrica -> 1 figura con 2 subplots (GMM+SAC | GMM+PPO). NO se superponen
los dos métodos porque sus ejes/algoritmos no son comparables directamente.

Requisitos en el env (ya instalados): wandb, matplotlib, pandas.
Login en Mac:  `wandb login`  (o exportar WANDB_API_KEY).

Uso:
    python scripts/plot_report_metrics.py
    python scripts/plot_report_metrics.py --project Project-RL-Manipulator-Arm \\
        --entity <tu_usuario_o_team>
    # forzar runs específicos (por nombre o id) si hay varios por group:
    python scripts/plot_report_metrics.py --sac_run <name_o_id> --ppo_run <name_o_id>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb

ROOT = Path(__file__).resolve().parent.parent

# Nombre de la métrica W&B por método. (clave_interna -> {sac, ppo})
METRICS = {
    "reward_episodic": {
        "title": "Cumulative reward per episode",
        "ylabel": "return / episode",
        "sac": "train_episode-return",
        "ppo": "rollout/ep_rew_mean",
    },
    "reward_eval": {
        "title": "Mean evaluation reward",
        "ylabel": "mean return (eval)",
        "sac": "eval_episode-avg-return",
        "ppo": "eval/mean_reward",
    },
    "loss_actor": {
        "title": "Policy / Actor loss",
        "ylabel": "loss",
        "sac": "loss_actor",
        "ppo": "train/policy_gradient_loss",
    },
    "loss_critic": {
        "title": "Value / Critic loss",
        "ylabel": "loss",
        "sac": "loss_critic",
        "ppo": "train/value_loss",
    },
    "loss_entropy": {
        "title": "Entropy loss  (SAC: H = -E[log pi], policy entropy)",
        "ylabel": "loss / entropy",
        "sac": "loss_entropy",
        "ppo": "train/entropy_loss",
    },
}

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
})


def pick_run(api, path, group, run_ref):
    """Devuelve un run de W&B: por nombre/id explícito, o el más reciente del group."""
    if run_ref:
        # primero intento por id directo
        try:
            return api.run(f"{path}/{run_ref}")
        except Exception:
            pass
        runs = [r for r in api.runs(path) if r.name == run_ref or r.id == run_ref]
        if runs:
            return runs[0]
        print(f"⚠️  No encontré run '{run_ref}' en {path}.")
        return None
    runs = list(api.runs(path, filters={"group": group}, order="-created_at"))
    if not runs:
        print(f"⚠️  No hay runs con group='{group}' en {path}.")
        return None
    if len(runs) > 1:
        print(f"ℹ️  {len(runs)} runs en group='{group}'; uso el más reciente: "
              f"{runs[0].name} ({runs[0].id}). Forzá otro con --{group.split('_')[-1]}_run.")
    return runs[0]


def fetch(run, key):
    """Serie (x=_step, y=key) de un run; [] si la métrica no existe."""
    if run is None:
        return [], []
    xs, ys = [], []
    try:
        for row in run.scan_history(keys=[key, "_step"]):
            y, x = row.get(key), row.get("_step")
            if y is None or x is None:
                continue
            try:
                y = float(y)  # W&B serializa -inf como string "-Infinity"
            except (TypeError, ValueError):
                continue
            if not np.isfinite(y):  # descarta -inf/inf/nan (placeholders de eval en SAC)
                continue
            xs.append(x)
            ys.append(y)
    except Exception as e:
        print(f"⚠️  Error leyendo '{key}' de {run.name}: {e}")
    return xs, ys


def mov_avg(y, w):
    y = np.asarray(y, float)
    if w <= 1 or len(y) < w:
        return y
    return np.convolve(y, np.ones(w) / w, mode="valid")


# inner env.step() por outer step (gmm_window = max_steps/adapt_per_episode = 64/4)
N_INNER = 16


def env_scale(run):
    """Factor para pasar el `_step` de W&B a environment steps (simulador), y así
    comparar SAC vs PPO en la MISMA unidad. SAC loguea `eval_total-env-steps` (ya
    son inner env.step); PPO loguea `global_step` (outer) -> x N_INNER = inner."""
    if run is None:
        return 1.0
    s = run.summary
    mx = s.get("_step")
    if not mx:
        return 1.0
    if s.get("eval_total-env-steps"):
        total = s["eval_total-env-steps"]
    elif s.get("global_step") is not None:
        total = s["global_step"] * N_INNER
    else:
        return 1.0
    return float(total) / float(mx)


def plot_metric(ax, run, key, color, smooth, scale=1.0):
    xs, ys = fetch(run, key)
    if not xs:
        ax.text(0.5, 0.5, f"no data\n({key})", ha="center", va="center",
                transform=ax.transAxes, color="gray")
        return False
    xs = [x * scale for x in xs]  # W&B step -> environment steps (simulador)
    ax.plot(xs, ys, color=color, alpha=0.30, marker="o", markersize=2, linewidth=0.8)
    ya = mov_avg(ys, smooth)
    ax.plot(xs[len(xs) - len(ya):], ya, color=color, label=f"moving average (w={smooth})")
    ax.set_xlabel("environment steps (simulator)")
    ax.legend(fontsize=8)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="Project-RL-Manipulator-Arm")
    ap.add_argument("--entity", default=None, help="usuario o team W&B (default: el tuyo).")
    ap.add_argument("--sac_group", default="gmm_sac")
    ap.add_argument("--ppo_group", default="gmm_ppo")
    ap.add_argument("--sac_run", default=None, help="nombre o id de un run SAC específico.")
    ap.add_argument("--ppo_run", default=None, help="nombre o id de un run PPO específico.")
    ap.add_argument("--smooth", type=int, default=9, help="ventana de media móvil.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else (ROOT / "Output_Training" / "report_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    entity = args.entity or api.default_entity
    path = f"{entity}/{args.project}"
    print(f"📡 W&B: {path}")

    sac_run = pick_run(api, path, args.sac_group, args.sac_run)
    ppo_run = pick_run(api, path, args.ppo_group, args.ppo_run)
    sac_scale = env_scale(sac_run)
    ppo_scale = env_scale(ppo_run)
    sac_env = (sac_run.summary.get("_step", 0) or 0) * sac_scale if sac_run else 0
    ppo_env = (ppo_run.summary.get("_step", 0) or 0) * ppo_scale if ppo_run else 0
    print(f"   GMM+SAC: {sac_run.name if sac_run else '—'}  (~{sac_env/1e6:.2f}M env steps)")
    print(f"   GMM+PPO: {ppo_run.name if ppo_run else '—'}  (~{ppo_env/1e6:.2f}M env steps)\n")

    for fname, m in METRICS.items():
        # Misma UNIDAD (env steps) en ambos subplots, pero cada uno con su propio
        # rango: SAC converge con ~6x menos interacción y con sharex quedaría
        # aplastado a la izquierda e ilegible. La comparación se hace leyendo
        # las escalas de los ejes (misma unidad).
        fig, (axs, axp) = plt.subplots(1, 2, figsize=(13, 4.5))
        ok_s = plot_metric(axs, sac_run, m["sac"], "tab:blue", args.smooth, sac_scale)
        ok_p = plot_metric(axp, ppo_run, m["ppo"], "tab:orange", args.smooth, ppo_scale)
        axs.set_title(f"GMM+SAC — {m['sac']}")
        axp.set_title(f"GMM+PPO — {m['ppo']}")
        axs.set_ylabel(m["ylabel"])
        fig.suptitle(m["title"], fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        path_png = out_dir / f"{fname}.png"
        fig.savefig(path_png.as_posix(), dpi=150)
        plt.close(fig)
        flag = "✓" if (ok_s or ok_p) else "∅"
        print(f"   {flag} {path_png.name}")

    print(f"\n📊 Figuras en: {out_dir}")


if __name__ == "__main__":
    main()
