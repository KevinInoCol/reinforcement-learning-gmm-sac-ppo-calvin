"""
Entrenamiento de GMM+PPO usando Stable-Baselines3.

Variante de SAC-GMM donde se reemplaza SAC por PPO para predecir Δθ
sobre los parámetros del GMM. El GMM se carga del .npy guardado por
gmm_train.py y permanece congelado durante el entrenamiento.

Uso típico (cluster):

    python3 scripts/gmm_ppo/gmm_ppo_train_sb3.py \\
        --total_timesteps 200000 \\
        --skill calvin_open_drawer \\
        --env calvin_scene_D \\
        --n_inner_steps 32 \\
        --max_outer_steps 2 \\
        --eval_freq 1000 \\
        --n_eval_episodes 10

Nota sobre timesteps: en este entrenamiento, cada "step" del agente PPO
corresponde a `n_inner_steps` pasos del simulador. Con max_outer_steps=2
y n_inner_steps=32, un episodio dura 2 acciones del PPO = 64 pasos sim.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[1]  # scripts/gmm_ppo → scripts → repo root
root = sac_gmm_path.parents[0]
sys.path.insert(0, sac_gmm_path.as_posix())
sys.path.insert(0, os.path.join(root, "calvin_env"))
sys.path.insert(0, cwd_path.as_posix())  # local copy del wrapper en scripts/gmm_ppo/

import hydra
from hydra import compose, initialize_config_dir

import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from sac_gmm.utils.env_maker import make_env
# IMPORTANTE: import desde la copia LOCAL en scripts/gmm_ppo/, no desde src/sac_gmm/envs/calvin/.
# Esto aisla las modificaciones GMM+PPO del original (usado por SAC-GMM si se reentrena).
from gmm_action_wrapper import GMMActionWrapper


class TimeLimitCallback(BaseCallback):
    """Detiene el training después de `max_seconds` wall-clock.

    Útil para benchmarks: "¿cuántos timesteps avanzo en X segundos?".
    """

    def __init__(self, max_seconds: float, verbose: int = 1):
        super().__init__(verbose)
        self.max_seconds = max_seconds
        self.start_time = None

    def _on_step(self) -> bool:
        if self.start_time is None:
            self.start_time = time.time()
        elapsed = time.time() - self.start_time
        if elapsed >= self.max_seconds:
            if self.verbose:
                print(
                    f"\n⏰ Time limit alcanzado: {elapsed:.0f}s "
                    f"≥ {self.max_seconds:.0f}s. Stopping training."
                )
            return False  # SB3 stops cuando un callback retorna False
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--skill", type=str, default="calvin_open_drawer")
    parser.add_argument("--env", type=str, default="calvin_scene_D")
    # Igualados a SAC-GMM: adapt_per_episode=4 => gmm_window=16 (n_inner_steps),
    # 4 decisiones por episodio (max_outer_steps). max_steps episodio = 4*16 = 64.
    parser.add_argument("--n_inner_steps", type=int, default=16)
    parser.add_argument("--max_outer_steps", type=int, default=4)
    parser.add_argument("--mu_change_range", type=float, default=0.03)
    # Eval más fino para una curva creíble (vs 5 ep/10 puntos del baseline).
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--n_eval_episodes", type=int, default=20)
    parser.add_argument("--save_freq", type=int, default=5000,
                        help="Frecuencia (en steps) para guardar checkpoints intermedios.")
    # PPO hyperparams
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Rollout length antes de cada update PPO")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_range", type=float, default=0.2)
    # CLAVE para recompensa dispersa: entropy bonus > 0 para que PPO explore
    # (análogo a la maximización de entropía de SAC). El baseline usaba 0.0
    # y por eso la política se estancaba sin explorar.
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--max_seconds", type=float, default=None,
                        help="Wall-clock time limit (segundos). Si se alcanza antes "
                             "de total_timesteps, el training se detiene y guarda.")
    # === Weights & Biases (opcional) ===
    # En RECOD los nodos de cómputo NO tienen internet => se usa modo offline y
    # luego `wandb sync <dir>` desde el headnode (con internet) para subirlo.
    parser.add_argument("--wandb", action="store_true",
                        help="Activa logging a W&B (offline por defecto).")
    parser.add_argument("--wandb_project", type=str, default="gmm-ppo")
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Nombre del run; por defecto se autogenera.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (
        sac_gmm_path / "logs" / "gmm_ppo" / args.skill
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[gmm_ppo] out_dir = {out_dir}")

    # === Compose configs via Hydra ===
    # Detectar Mac (Darwin) y deshabilitar EGL (Linux-only).
    import platform
    is_mac = platform.system() == "Darwin"

    config_dir = str(sac_gmm_path / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        overrides = [
            f"skill={args.skill}",
            f"env={args.env}",
            "agent=sac_calvin",
            "obs_space=[pos]",
        ]
        if is_mac:
            # En Mac no existe eglRendererPlugin; forzar pybullet DIRECT mode
            overrides.append("env.calvin_env.env.use_egl=false")
            print("[gmm_ppo] macOS detectado → use_egl=false")
        cfg = compose(config_name="sac_train", overrides=overrides)

    # === Crear datamodule (para start position) y env ===
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    print(f"[gmm_ppo] Creando CalvinSkillEnv para skill={args.skill}")
    raw_env = make_env(cfg.env, cfg.skill, datamodule.dataset.start)

    # === Envolver con GMMActionWrapper ===
    # demos_target viene del dataset (np.mean del último frame de las demos);
    # kp_mock_cfg viene de Hydra (config/kp_mock/default.yaml, inyectado via skill_setup).
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

    print(f"[gmm_ppo] obs_space={env.observation_space}")
    print(f"[gmm_ppo] action_space={env.action_space}  (Δθ del GMM)")

    # Sanity check ANTES de wrappear con Monitor (env.reset devuelve (obs, info)
    # en gymnasium; Monitor cambia esa firma).
    obs, _ = env.reset()
    print(f"[gmm_ppo] obs.shape={obs.shape}, ejemplo={obs}")

    # Wrap con Monitor — necesario para que SB3 reporte ep_rew_mean / ep_len_mean
    # correctamente y para que EvalCallback no tire UserWarning.
    env = Monitor(env)

    # === PPO de SB3 ===
    # device='cpu': para MlpPolicy (no CNN), SB3 recomienda CPU — la
    # transferencia GPU↔CPU del rollout buffer es más cara que el forward MLP.
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=64,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,  # >0 => exploración (crítico en reward dispersa)
        verbose=1,
        device="cpu",
        tensorboard_log=str(out_dir / "tb"),
    )
    print(f"[gmm_ppo] ent_coef={args.ent_coef}  n_steps={args.n_steps}  "
          f"n_inner={args.n_inner_steps}  max_outer={args.max_outer_steps}")

    # === Callbacks ===
    eval_callback = EvalCallback(
        env,
        best_model_save_path=str(out_dir / "best"),
        log_path=str(out_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    ckpt_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="gmm_ppo",
    )

    # Construir la lista de callbacks (opcionalmente con time limit)
    callbacks = [eval_callback, ckpt_callback]
    if args.max_seconds is not None:
        callbacks.append(TimeLimitCallback(max_seconds=args.max_seconds, verbose=1))
        print(f"[gmm_ppo] Time limit: {args.max_seconds:.0f}s "
              f"(~{args.max_seconds/3600:.2f}h)")

    # === W&B (opcional) ===
    # sync_tensorboard=True mirrorea automáticamente las métricas que SB3 ya
    # escribe en TensorBoard. En RECOD se corre con WANDB_MODE=offline (nodos
    # sin internet) y luego `wandb sync <out_dir>/wandb/offline-run-*` en headnode.
    # Envuelto en try/except: si W&B falla, el entrenamiento NO se cae.
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name,
                dir=str(out_dir),
                sync_tensorboard=True,
                config=vars(args),
                save_code=False,
            )
            print(f"[gmm_ppo] W&B activo: project={args.wandb_project} "
                  f"mode={os.environ.get('WANDB_MODE', 'online')} "
                  f"dir={out_dir}/wandb")
        except Exception as e:
            print(f"[gmm_ppo] WARN: no se pudo iniciar W&B ({e}); sigo sin W&B.")
            wandb_run = None

    start_t = time.time()
    print(f"[gmm_ppo] Iniciando training: total_timesteps={args.total_timesteps}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        log_interval=10,
    )
    elapsed = time.time() - start_t
    print(f"[gmm_ppo] Training terminó en {elapsed:.0f}s "
          f"(~{elapsed/3600:.2f}h)")

    final_path = out_dir / "gmm_ppo_final.zip"
    model.save(str(final_path))
    print(f"[gmm_ppo] Modelo final guardado en {final_path}")

    if wandb_run is not None:
        try:
            wandb_run.finish()
            print(f"[gmm_ppo] W&B run cerrado. Para subirlo desde headnode:\n"
                  f"   wandb sync {out_dir}/wandb/offline-run-*")
        except Exception:
            pass


if __name__ == "__main__":
    main()
