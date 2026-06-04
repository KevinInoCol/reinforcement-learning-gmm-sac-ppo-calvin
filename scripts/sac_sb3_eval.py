"""
Evaluación de un modelo SAC entrenado con Stable-Baselines3.

Carga un .zip de SB3, ejecuta N episodios en CalvinSkillEnv con GUI opcional,
graba video MP4 vía PyBullet, y guarda métricas en CSV/JSON
(misma estructura que agent_eval_record.py).

Uso típico (con GUI + video en Mac):

    python3 scripts/sac_sb3_eval.py \\
        --model checkpoints/sac_sb3_open_drawer_best.zip \\
        --skill calvin_open_drawer \\
        --env calvin_scene_D \\
        --num_episodes 5 \\
        --show_gui \\
        --record_video \\
        --step_delay 0.1

Si NO querés GUI (headless en cluster):

    python3 scripts/sac_sb3_eval.py \\
        --model logs/sac_sb3/calvin_open_drawer/best/best_model.zip \\
        --num_episodes 10
"""
import os
import sys
import argparse
import datetime
import json
import shutil
import tempfile
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]
sys.path.insert(0, sac_gmm_path.as_posix())
sys.path.insert(0, os.path.join(root, "calvin_env"))

import numpy as np
import gym as old_gym
import gymnasium as gym
import hydra
from hydra import compose, initialize_config_dir

from stable_baselines3 import SAC
from sac_gmm.utils.env_maker import make_env


# Reusamos el wrapper del training script
from scripts.sac_train_sb3 import CalvinGymWrapper


def setup_camera(pb_client):
    """Posición de cámara default zoom-in (igual que en agent_eval_record.py)."""
    try:
        dist = float(os.environ.get("SACGMM_CAM_DIST", "1.2"))
        yaw = float(os.environ.get("SACGMM_CAM_YAW", "50"))
        pitch = float(os.environ.get("SACGMM_CAM_PITCH", "-30"))
        target_str = os.environ.get("SACGMM_CAM_TARGET", "0.0,-0.2,0.5")
        target = [float(v) for v in target_str.split(",")]
        pb_client.resetDebugVisualizerCamera(
            cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch,
            cameraTargetPosition=target,
        )
        print(f"🎥 Camera: dist={dist}, yaw={yaw}, pitch={pitch}, target={target}")
    except Exception as e:
        print(f"⚠️  No se pudo setear camera: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path al .zip del modelo SB3")
    parser.add_argument("--skill", default="calvin_open_drawer")
    parser.add_argument("--env", default="calvin_scene_D")
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--show_gui", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--use_egl", action="store_true", help="Force EGL (no-Mac)")
    parser.add_argument("--step_delay", type=float, default=0.0,
                        help="Wall-clock sleep per env step (slow-motion).")
    parser.add_argument("--output_dir", default=None,
                        help="Default: Output_Inference/")
    args = parser.parse_args()

    model_path = Path(args.model).expanduser().resolve()
    assert model_path.exists(), f"Modelo no encontrado: {model_path}"

    out_dir = Path(args.output_dir) if args.output_dir else (sac_gmm_path / "Output_Inference")
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    (out_dir / "results_table").mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agent_name = "SAC_SB3"
    run_id = f"{agent_name}_{args.skill}_{timestamp}"

    # === Compose configs via Hydra ===
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

    # === Build env ===
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    raw_env = make_env(cfg.env, cfg.skill, datamodule.dataset.start)
    env = CalvinGymWrapper(raw_env, max_steps=cfg.skill.max_steps)

    # === Set camera before recording ===
    if args.show_gui:
        setup_camera(raw_env.p)

    # === Load model ===
    print(f"📂 Cargando modelo: {model_path}")
    model = SAC.load(str(model_path), env=env)
    print(f"✅ Modelo cargado. policy={type(model.policy).__name__}")

    # === Setup video recording ===
    pb_client = None
    log_id = None
    tmp_video_path = None
    final_video_path = out_dir / "videos" / f"eval_{run_id}.mp4"

    if args.record_video and args.show_gui:
        try:
            pb_client = raw_env.p
            tmp_dir = Path(tempfile.gettempdir())
            tmp_video_path = tmp_dir / f"sacgmm_sb3_eval_{os.getpid()}_{final_video_path.name}"
            log_id = pb_client.startStateLogging(
                pb_client.STATE_LOGGING_VIDEO_MP4,
                tmp_video_path.as_posix(),
            )
            print(f"📹 Grabando → {final_video_path}")
        except Exception as e:
            print(f"⚠️  No se pudo iniciar grabación: {e}")
            log_id = None

    # === Eval loop ===
    import time
    successes, returns, lengths = 0, [], []
    for ep in range(args.num_episodes):
        obs, _ = env.reset()
        ep_return, ep_length, done = 0.0, 0, False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            if args.step_delay > 0:
                time.sleep(args.step_delay)
            done = terminated or truncated

        success = bool(info.get("success", False)) if info else False
        if success:
            successes += 1
        returns.append(ep_return)
        lengths.append(ep_length)
        print(f"Episode {ep+1}/{args.num_episodes}: return={ep_return:.2f}, "
              f"length={ep_length}, success={success}")

    # === Stop recording ===
    if log_id is not None and pb_client is not None:
        try:
            pb_client.stopStateLogging(log_id)
            if tmp_video_path and tmp_video_path.exists():
                shutil.move(tmp_video_path.as_posix(), final_video_path.as_posix())
                size_mb = final_video_path.stat().st_size / (1024 * 1024)
                print(f"✅ Video guardado: {final_video_path}  ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"⚠️  Error al detener grabación: {e}")

    accuracy = successes / args.num_episodes
    mean_return = float(np.mean(returns))
    mean_length = float(np.mean(lengths))

    print(f"\n=== Resumen ===")
    print(f"Accuracy:  {accuracy:.2f} ({successes}/{args.num_episodes})")
    print(f"Return:    {mean_return:.2f}")
    print(f"Length:    {mean_length:.2f}")

    # === Save JSON + append CSV (same format as agent_eval_record.py) ===
    json_path = out_dir / "results_table" / f"eval_{run_id}.json"
    payload = {
        "timestamp": timestamp,
        "agent": agent_name,
        "skill": args.skill,
        "env": args.env,
        "num_eval_episodes": args.num_episodes,
        "num_eval_seeds": 1,
        "model_path": str(model_path),
        "video_path": str(final_video_path) if log_id is not None else None,
        "accuracy_per_seed": [accuracy],
        "accuracy_mean": accuracy,
        "accuracy_var": 0.0,
        "return_per_seed": returns,
        "return_mean": mean_return,
        "length_per_seed": lengths,
        "length_mean": mean_length,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"📊 JSON guardado: {json_path}")

    # CSV append
    import csv
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
            timestamp, agent_name, args.skill, args.num_episodes, 1,
            "n/a", accuracy, 0.0, json.dumps([accuracy]),
            mean_return, mean_length, str(model_path),
            str(final_video_path) if log_id is not None else "",
        ])
    print(f"📊 CSV actualizado: {csv_path}")


if __name__ == "__main__":
    main()
