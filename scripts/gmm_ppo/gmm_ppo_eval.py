"""
Evaluación de un modelo GMM+PPO entrenado con Stable-Baselines3.

Carga un .zip de PPO entrenado sobre GMMActionWrapper, corre N episodios
con GUI opcional, graba video y guarda métricas (mismo formato que el
resto del pipeline).

Uso típico (Mac, con GUI + video):

    python3 scripts/gmm_ppo/gmm_ppo_eval.py \\
        --model checkpoints/gmm_ppo_open_drawer_best.zip \\
        --skill calvin_open_drawer \\
        --env calvin_scene_D \\
        --num_episodes 5 \\
        --show_gui --record_video --step_delay 0.1
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[1]  # scripts/gmm_ppo → scripts → repo root
root = sac_gmm_path.parents[0]
sys.path.insert(0, sac_gmm_path.as_posix())
sys.path.insert(0, os.path.join(root, "calvin_env"))
sys.path.insert(0, cwd_path.as_posix())  # local copy del wrapper en scripts/gmm_ppo/

import numpy as np
import hydra
from hydra import compose, initialize_config_dir
from pytorch_lightning import seed_everything

from stable_baselines3 import PPO
from sac_gmm.utils.env_maker import make_env
# IMPORTANTE: import desde la copia LOCAL en scripts/gmm_ppo/, no desde src/sac_gmm/envs/calvin/.
# Esto aisla las modificaciones GMM+PPO del original (usado por SAC-GMM si se reentrena).
from gmm_action_wrapper import GMMActionWrapper


def setup_camera(pb_client):
    try:
        dist = float(os.environ.get("SACGMM_CAM_DIST", "1.2"))
        yaw = float(os.environ.get("SACGMM_CAM_YAW", "50"))
        pitch = float(os.environ.get("SACGMM_CAM_PITCH", "-30"))
        target = [float(v) for v in os.environ.get("SACGMM_CAM_TARGET", "0.0,-0.2,0.5").split(",")]
        pb_client.resetDebugVisualizerCamera(
            cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch,
            cameraTargetPosition=target,
        )
        print(f"🎥 Camera: dist={dist}, yaw={yaw}, pitch={pitch}, target={target}")
    except Exception as e:
        print(f"⚠️  No se pudo setear camera: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--skill", default="calvin_open_drawer")
    parser.add_argument("--env", default="calvin_scene_D")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Episodios por seed.")
    parser.add_argument("--num_seeds", type=int, default=1,
                        help="Número de seeds (grupos independientes de num_episodes) "
                             "para reportar media ± varianza, igual que agent_eval_record.py.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed base; cada grupo s usa seed+s (default 42, "
                             "mismo que GMM/GMM+SAC → mismas posiciones iniciales).")
    # Consistentes con el entrenamiento corregido (igualados a SAC: 4 adapt/episodio).
    parser.add_argument("--n_inner_steps", type=int, default=16)
    parser.add_argument("--max_outer_steps", type=int, default=4)
    parser.add_argument("--mu_change_range", type=float, default=0.03)
    parser.add_argument("--show_gui", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--use_egl", action="store_true")
    parser.add_argument("--step_delay", type=float,
                        default=float(os.environ.get("SACGMM_STEP_DELAY", "0.0")),
                        help="Pausa entre steps en segundos. También se puede setear con "
                             "la env var SACGMM_STEP_DELAY=0.05 (el flag tiene prioridad).")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    assert model_path.exists(), f"Modelo no encontrado: {model_path}"

    out_dir = Path(args.output_dir) if args.output_dir else (sac_gmm_path / "Output_Inference")
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "results_table").mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = "GMM_PPO"
    run_id = f"{agent_name}_{args.skill}_{timestamp}"

    # === Hydra ===
    config_dir = str(sac_gmm_path / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        overrides = [
            f"skill={args.skill}",
            f"env={args.env}",
            "agent=sac_calvin",
            "obs_space=[pos]",
        ]
        if args.show_gui:
            overrides.append("show_gui=true")
            overrides.append("env.calvin_env.env.show_gui=true")
            if not args.use_egl:
                overrides.append("env.calvin_env.env.use_egl=false")
        cfg = compose(config_name="sac_train", overrides=overrides)

    # === Env ===
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    raw_env = make_env(cfg.env, cfg.skill, datamodule.dataset.start)
    env = GMMActionWrapper(
        raw_env,
        gmm_cfg=cfg.gmm,
        kp_mock_cfg=cfg.kp_mock,
        demos_target=datamodule.dataset.goal,
        n_inner_steps=args.n_inner_steps,
        max_outer_steps=args.max_outer_steps,
        mu_change_range=args.mu_change_range,
        priors_change_range=0.0,
        modify_covariances=False,
    )

    if args.show_gui:
        setup_camera(raw_env.p)

    print(f"📂 Cargando modelo: {model_path}")
    model = PPO.load(str(model_path), env=env)
    print(f"✅ Modelo cargado.")

    # === Recording ===
    pb_client = None
    log_id = None
    tmp_video_path = None
    final_video_path = out_dir / "videos" / f"eval_{run_id}.mp4"

    if args.record_video and args.show_gui:
        try:
            pb_client = raw_env.p
            tmp_dir = Path(tempfile.gettempdir())
            tmp_video_path = tmp_dir / f"gmm_ppo_eval_{os.getpid()}_{final_video_path.name}"
            log_id = pb_client.startStateLogging(
                pb_client.STATE_LOGGING_VIDEO_MP4, tmp_video_path.as_posix(),
            )
            print(f"📹 Grabando → {final_video_path}")
        except Exception as e:
            print(f"⚠️  No se pudo iniciar grabación: {e}")
            log_id = None

    # === Eval loop (bucle externo de seeds, igual que agent_eval_record.py) ===
    per_seed_acc, per_seed_return, per_seed_length = [], [], []
    total_successes, total_episodes = 0, 0
    for s in range(args.num_seeds):
        # seed+s controla el ruido gaussiano de la posición inicial del brazo
        # (env.reset → sample_base_shift → np.random.normal), reproducible por seed.
        seed_everything(args.seed + s, workers=True)
        successes, seed_returns, seed_lengths = 0, [], []
        for ep in range(args.num_episodes):
            obs, _ = env.reset()
            ep_return, outer_steps, done = 0.0, 0, False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += reward
                outer_steps += 1
                if args.step_delay > 0:
                    time.sleep(args.step_delay)
                done = terminated or truncated
            success = bool(info.get("success", False)) if info else False
            if success:
                successes += 1
            seed_returns.append(ep_return)
            # Outer steps × n_inner_steps = total sim steps
            seed_lengths.append(outer_steps * args.n_inner_steps)
            print(f"[seed {s+1}/{args.num_seeds}] Episode {ep+1}/{args.num_episodes}: "
                  f"return={ep_return:.2f}, outer_steps={outer_steps}, success={success}")
        seed_acc = successes / args.num_episodes
        per_seed_acc.append(seed_acc)
        per_seed_return.append(float(np.mean(seed_returns)))
        per_seed_length.append(float(np.mean(seed_lengths)))
        total_successes += successes
        total_episodes += args.num_episodes
        print(f"[seed {s+1}/{args.num_seeds}] accuracy={seed_acc:.3f} "
              f"({successes}/{args.num_episodes})")

    if log_id is not None and pb_client is not None:
        try:
            pb_client.stopStateLogging(log_id)
            if tmp_video_path and tmp_video_path.exists():
                shutil.move(tmp_video_path.as_posix(), final_video_path.as_posix())
                size_mb = final_video_path.stat().st_size / (1024 * 1024)
                print(f"✅ Video guardado: {final_video_path}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"⚠️  Error al detener grabación: {e}")

    accuracy_mean = float(np.mean(per_seed_acc))
    accuracy_var = float(np.var(per_seed_acc))
    mean_return = float(np.mean(per_seed_return))
    mean_length = float(np.mean(per_seed_length))

    print(f"\n=== Resumen GMM+PPO ===")
    print(f"Accuracy:  mean={accuracy_mean:.4f}, var={accuracy_var:.4f}, per_seed={per_seed_acc}")
    print(f"Aciertos:  {total_successes}/{total_episodes}")
    print(f"Return:    {mean_return:.2f}")
    print(f"Length:    {mean_length:.2f} sim steps")

    # === Save JSON + CSV ===
    json_path = out_dir / "results_table" / f"eval_{run_id}.json"
    payload = {
        "timestamp": timestamp,
        "agent": agent_name,
        "skill": args.skill,
        "env": args.env,
        "num_eval_episodes": args.num_episodes,
        "num_eval_seeds": args.num_seeds,
        "n_components_gmm": env.K,
        "n_inner_steps": args.n_inner_steps,
        "max_outer_steps": args.max_outer_steps,
        "mu_change_range": args.mu_change_range,
        "model_path": str(model_path),
        "video_path": str(final_video_path) if log_id is not None else None,
        "accuracy_per_seed": per_seed_acc,
        "accuracy_mean": accuracy_mean,
        "accuracy_var": accuracy_var,
        "return_per_seed": per_seed_return,
        "return_mean": mean_return,
        "length_per_seed": per_seed_length,
        "length_mean": mean_length,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"📊 JSON guardado: {json_path}")

    csv_path = out_dir / "results_table" / "eval_results.csv"
    headers = [
        "timestamp", "agent", "skill", "num_eval_episodes", "num_eval_seeds",
        "n_components_gmm", "accuracy_mean", "accuracy_var", "accuracy_per_seed",
        "return_mean", "length_mean", "chk_dir", "video_path",
    ]
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(headers)
        writer.writerow([
            timestamp, agent_name, args.skill, args.num_episodes, args.num_seeds,
            env.K, accuracy_mean, accuracy_var, json.dumps(per_seed_acc),
            mean_return, mean_length, str(model_path),
            str(final_video_path) if log_id is not None else "",
        ])
    print(f"📊 CSV actualizado: {csv_path}")


if __name__ == "__main__":
    main()
