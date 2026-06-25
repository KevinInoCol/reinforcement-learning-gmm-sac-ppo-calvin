"""
Visualiza la variabilidad de la posición INICIAL del end-effector entre episodios.

No carga ningún modelo: solo construye el entorno y hace N resets (con el mismo
esquema de seeds que la evaluación real, seed_everything(seed+s)), capturando la
posición XYZ del gripper en cada arranque. Genera:

  1. Marcadores persistentes (esferas) en el simulador, uno por episodio, de modo
     que en UNA sola escena/imagen se ve la nube de puntos de inicio (idea: aunque
     el desplazamiento por episodio sea chico, acumulados se ven claramente).
  2. Un scatter top-down (x,y) con matplotlib  ->  figura lista para el paper.
  3. Estadística del spread (rango y std en cm)  ->  para decidir si es "suficiente".
  4. (opcional) Una captura de cámara del simulador con todos los marcadores.

Uso típico en Mac (ver en vivo en el simulador + figuras):

    python3 scripts/viz_start_positions.py \\
        --skill calvin_open_drawer --env calvin_scene_D \\
        --num_seeds 3 --num_episodes 20 \\
        --gui --hold

Preview de OTRA magnitud de ruido sin tocar el experimento real:

    python3 scripts/viz_start_positions.py --ee_noise 0.03,0.03,0.0
    python3 scripts/viz_start_positions.py --ee_noise 0.06,0.06,0.0   # más vistoso

Headless (RECOD): omití --gui; igual genera el scatter PNG y la estadística.
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]  # scripts -> repo root
root = sac_gmm_path.parents[0]
sys.path.insert(0, sac_gmm_path.as_posix())
sys.path.insert(0, os.path.join(root, "calvin_env"))

import numpy as np
import hydra
from hydra import compose, initialize_config_dir
from pytorch_lightning import seed_everything

from sac_gmm.utils.env_maker import make_env


# Paleta para distinguir seeds (RGBA)
_SEED_COLORS = [
    [0.90, 0.10, 0.10, 1.0],  # rojo
    [0.10, 0.45, 0.90, 1.0],  # azul
    [0.10, 0.70, 0.20, 1.0],  # verde
    [0.85, 0.55, 0.10, 1.0],  # naranja
    [0.55, 0.20, 0.75, 1.0],  # violeta
]


def setup_camera(p):
    dist = float(os.environ.get("SACGMM_CAM_DIST", "1.2"))
    yaw = float(os.environ.get("SACGMM_CAM_YAW", "50"))
    pitch = float(os.environ.get("SACGMM_CAM_PITCH", "-30"))
    target = [float(v) for v in os.environ.get("SACGMM_CAM_TARGET", "0.0,-0.2,0.5").split(",")]
    try:
        p.resetDebugVisualizerCamera(
            cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch,
            cameraTargetPosition=target,
        )
    except Exception as e:
        print(f"⚠️  No se pudo setear la cámara: {e}")
    return dist, yaw, pitch, target


def add_marker(p, pos, rgba, radius=0.012):
    """Esfera visual (sin colisión) fija en el mundo, persiste entre resets."""
    try:
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
        return p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs, basePosition=list(pos))
    except Exception as e:
        print(f"⚠️  No se pudo crear marcador: {e}")
        return None


def capture_image(p, path, cam, w=1280, h=960):
    """Captura best-effort de la escena con todos los marcadores."""
    dist, yaw, pitch, target = cam
    try:
        view = p.computeViewMatrixFromYawPitchRoll(target, dist, yaw, pitch, 0, 2)
        proj = p.computeProjectionMatrixFOV(60.0, w / h, 0.1, 100.0)
        img = p.getCameraImage(w, h, view, proj)
        rgb = np.reshape(img[2], (h, w, 4))[:, :, :3].astype(np.uint8)
        import matplotlib.pyplot as plt
        plt.imsave(path.as_posix(), rgb)
        print(f"🖼️  Captura del simulador: {path}")
    except Exception as e:
        print(f"⚠️  No se pudo capturar imagen del simulador: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", default="calvin_open_drawer")
    ap.add_argument("--env", default="calvin_scene_D")
    ap.add_argument("--num_seeds", type=int, default=3, help="Grupos de seeds (seed+s).")
    ap.add_argument("--num_episodes", type=int, default=20, help="Episodios por seed.")
    ap.add_argument("--seed", type=int, default=42, help="Seed base (igual que la eval real).")
    ap.add_argument("--ee_noise", default=None,
                    help="Override 'x,y,z' del ruido (solo para preview; no toca el config).")
    ap.add_argument("--gui", action="store_true", help="Abrir el simulador.")
    ap.add_argument("--hold", action="store_true",
                    help="Con --gui, dejar la ventana abierta al final (Ctrl+C para salir).")
    ap.add_argument("--step_delay", type=float, default=0.15,
                    help="Pausa entre resets para ver aparecer cada marcador (con --gui).")
    ap.add_argument("--save_image", action="store_true",
                    help="Capturar imagen de la escena con todos los marcadores.")
    ap.add_argument("--output_dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else (sac_gmm_path / "Output_Inference" / "start_viz")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{args.skill}_{ts}"

    # === Hydra (mismo armado que gmm_ppo_eval) ===
    config_dir = str(sac_gmm_path / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        overrides = [
            f"skill={args.skill}",
            f"env={args.env}",
            "agent=sac_calvin",
            "obs_space=[pos]",
        ]
        if args.gui:
            overrides += [
                "show_gui=true",
                "env.calvin_env.env.show_gui=true",
                "env.calvin_env.env.use_egl=false",
            ]
        cfg = compose(config_name="sac_train", overrides=overrides)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    env = make_env(cfg.env, cfg.skill, datamodule.dataset.start)

    if args.ee_noise is not None:
        env.ee_noise = np.array([float(v) for v in args.ee_noise.split(",")])
    print(f"⚙️  ee_noise = {np.asarray(env.ee_noise).tolist()}  "
          f"(skill={args.skill}, env={args.env})")

    cam = None
    if args.gui:
        cam = setup_camera(env.p)

    # === Loop de resets: misma semilla que la eval real ===
    starts = []          # [(seed_idx, ep, [x,y,z]), ...]
    for s in range(args.num_seeds):
        seed_everything(args.seed + s, workers=True)
        color = _SEED_COLORS[s % len(_SEED_COLORS)]
        for ep in range(args.num_episodes):
            env.reset()
            pos = env.last_start_ee_pos
            if pos is None:
                print("⚠️  last_start_ee_pos es None (¿obs sin 'position'?). Salto.")
                continue
            starts.append((s, ep, list(pos)))
            if args.gui:
                add_marker(env.p, pos, color)
                if args.step_delay > 0:
                    time.sleep(args.step_delay)
            print(f"[seed {s} ep {ep:02d}] EE start = "
                  f"({pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f})")

    if not starts:
        print("❌ No se registró ninguna posición inicial.")
        return

    arr = np.array([p for _, _, p in starts])  # (N,3)

    # === Estadística del spread (¿es suficiente?) ===
    rng_cm = (arr.max(0) - arr.min(0)) * 100.0
    std_cm = arr.std(0) * 100.0
    print("\n=== Variabilidad del inicio del EE ===")
    print(f"  N posiciones: {len(arr)}")
    print(f"  rango (cm):  x={rng_cm[0]:.2f}  y={rng_cm[1]:.2f}  z={rng_cm[2]:.2f}")
    print(f"  std   (cm):  x={std_cm[0]:.2f}  y={std_cm[1]:.2f}  z={std_cm[2]:.2f}")

    # === Scatter top-down (figura para el paper) ===
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        for s in range(args.num_seeds):
            pts = np.array([p for si, _, p in starts if si == s])
            if len(pts):
                ax.scatter(pts[:, 0] * 100, pts[:, 1] * 100, s=40, alpha=0.75,
                           color=_SEED_COLORS[s % len(_SEED_COLORS)][:3],
                           label=f"seed {args.seed + s}")
        ax.scatter([arr[:, 0].mean() * 100], [arr[:, 1].mean() * 100],
                   marker="x", s=120, color="k", label="media")
        ax.set_xlabel("x (cm)")
        ax.set_ylabel("y (cm)")
        ax.set_title(f"Posición inicial del EE — {args.skill}\n"
                     f"ee_noise={np.asarray(env.ee_noise).tolist()}  "
                     f"(rango x={rng_cm[0]:.1f}cm, y={rng_cm[1]:.1f}cm)")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend()
        scatter_path = out_dir / f"start_scatter_{tag}.png"
        fig.tight_layout()
        fig.savefig(scatter_path.as_posix(), dpi=150)
        print(f"📈 Scatter guardado: {scatter_path}")
    except Exception as e:
        print(f"⚠️  No se pudo generar el scatter: {e}")

    if args.save_image and args.gui and cam is not None:
        capture_image(env.p, out_dir / f"start_scene_{tag}.png", cam)

    if args.gui and args.hold:
        print("\n🖱️  Ventana del simulador abierta con todos los marcadores. "
              "Orbitá con el mouse. Ctrl+C para cerrar.")
        try:
            while env.p.isConnected():
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n👋 Cerrando.")


if __name__ == "__main__":
    main()
