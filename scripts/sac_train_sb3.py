"""
SAC puro entrenado con Stable-Baselines3 sobre CalvinSkillEnv.

Reemplazo del `sac_train.py` original (que está roto en el repo).
Usa la implementación oficial de SAC de SB3 (sb3 2.x) — testeada y mantenida.

Uso:
    python3 scripts/sac_train_sb3.py [--total_timesteps N] [--out_dir DIR]

El script:
  1. Compone configs vía Hydra (skill=calvin_open_drawer, env=calvin_scene_D)
  2. Crea CalvinSkillEnv vía make_env existente
  3. Envuelve con CalvinGymWrapper para exponer interfaz gym/gymnasium pura
  4. Entrena SAC de SB3
  5. Guarda checkpoints + TensorBoard logs

Diseño del wrapper:
  - Observation: solo "position" (3D) — equivalente a state_type='pos' del paper
  - Action: 3D position delta (el env acepta 3D actions y agrega gripper internamente)
  - Reward: sparse 0/1 desde el env
  - Episode termina a los 64 pasos (skill.max_steps) o cuando done
"""
import os
import sys
from pathlib import Path
import argparse

cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]
sys.path.insert(0, sac_gmm_path.as_posix())
sys.path.insert(0, os.path.join(root, "calvin_env"))

import numpy as np

# Old gym (lo que usa CalvinSkillEnv internamente)
import gym as old_gym
# Gymnasium (lo que SB3 2.x espera)
import gymnasium as gym

import hydra
from hydra import compose, initialize_config_dir

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

from sac_gmm.utils.env_maker import make_env


class CalvinGymWrapper(gym.Env):
    """Wrapper de CalvinSkillEnv para SB3 (gymnasium API).

    - Convierte Dict obs → np.ndarray (solo "position", 3D)
    - Action 3D continua (el env interno la expande con orientation=0 y gripper=-1)
    - Convierte 4-tuple step de gym → 5-tuple de gymnasium
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, calvin_env, max_steps=64):
        super().__init__()
        self.env = calvin_env
        self.max_steps = max_steps
        self._step_count = 0
        # Gymnasium spaces (compatibles con SB3 2.x)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def _extract_obs(self, obs):
        if isinstance(obs, dict):
            pos = obs.get("position", np.zeros(3))
            return np.asarray(pos, dtype=np.float32)
        return np.asarray(obs, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # gymnasium reset() retorna (obs, info)
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        self._step_count = 0
        return self._extract_obs(obs), {}

    def step(self, action):
        # gym retorna (obs, reward, done, info)
        # gymnasium quiere (obs, reward, terminated, truncated, info)
        obs, reward, done, info = self.env.step(np.asarray(action, dtype=np.float32))
        self._step_count += 1
        truncated = self._step_count >= self.max_steps
        terminated = bool(done) and not truncated  # done por éxito vs timeout
        # En sparse-reward, done=True puede ser por éxito o por max_steps; ambos OK
        return (
            self._extract_obs(obs),
            float(reward),
            terminated,
            truncated,
            info or {},
        )

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_timesteps", type=int, default=200_000)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--skill", type=str, default="calvin_open_drawer")
    parser.add_argument("--env", type=str, default="calvin_scene_D")
    parser.add_argument("--eval_freq", type=int, default=2000)
    parser.add_argument("--n_eval_episodes", type=int, default=10)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else (sac_gmm_path / "logs" / "sac_sb3" / args.skill)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[sac_train_sb3] out_dir = {out_dir}")

    # Compose configs via Hydra (lo justo para crear el env)
    config_dir = str(sac_gmm_path / "config")
    with initialize_config_dir(version_base="1.1", config_dir=config_dir):
        cfg = compose(
            config_name="sac_train",
            overrides=[
                f"skill={args.skill}",
                f"env={args.env}",
                "agent=sac_calvin",  # solo para que el datamodule resuelva
                "obs_space=[pos]",   # forzamos solo position
            ],
        )

    # Datamodule para start_position
    datamodule = hydra.utils.instantiate(cfg.datamodule)

    # Crear env CALVIN
    print(f"[sac_train_sb3] Creando CalvinSkillEnv para skill={args.skill}")
    raw_env = make_env(cfg.env, cfg.skill, datamodule.dataset.start)

    # Envolver para gym/SB3
    env = CalvinGymWrapper(raw_env, max_steps=cfg.skill.max_steps)
    print(f"[sac_train_sb3] obs_space={env.observation_space}, act_space={env.action_space}")

    # Sanity: un reset + un step
    obs, _ = env.reset()
    print(f"[sac_train_sb3] obs.shape={obs.shape}, ejemplo={obs}")

    # SB3 SAC con hiperparámetros similares al paper
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(out_dir / "tb"),
    )

    # Callbacks
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
        save_freq=5000,
        save_path=str(out_dir / "checkpoints"),
        name_prefix="sac_sb3",
    )

    print(f"[sac_train_sb3] Iniciando training: total_timesteps={args.total_timesteps}")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[eval_callback, ckpt_callback],
        log_interval=10,
    )

    final_path = out_dir / "sac_sb3_final.zip"
    model.save(str(final_path))
    print(f"[sac_train_sb3] Modelo final guardado en {final_path}")


if __name__ == "__main__":
    main()
