"""
Wrapper de gym que transforma `CalvinSkillEnv` en un env donde la acción
del agente RL es un Δθ (correcciones a los parámetros del GMM) en lugar de
una velocidad directa.

Mecánica (idéntica al loop del paper SAC-GMM):

    para cada paso del env wrappeado:
        recibir Δθ (Δμ, Δπ, ΔΣ) del agente RL
        crear GMM_temporal = GMM_base + Δθ
        for inner_step in range(N):
            ξ̇ = GMM_temporal.predict(pose_actual)
            ξ̇ ← env.step(ξ̇)
            reward += r
            si done: break
        return (obs, reward_acumulado, terminated, truncated, info)

Esto permite que cualquier algoritmo RL on-action-space=Δθ (SAC, PPO, ...) sea
usado como refiner del GMM. Reutilizable entre experimentos.

Compatible con gymnasium (SB3 2.x). Devuelve obs Box(3,) por defecto
(solo position). El GMM se carga desde el .npy guardado por gmm_train.py.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig


class GMMActionWrapper(gym.Env):
    """
    Wraps CalvinSkillEnv para que la acción sea Δθ (correcciones al GMM)
    y cada step ejecute internamente N pasos del simulador con el GMM
    modificado.

    Args:
        calvin_env: instancia de CalvinSkillEnv ya construida.
        gmm_cfg: DictConfig con la configuración del GMM (target + skill).
                 Debe poderse instanciar con hydra.utils.instantiate(gmm_cfg)
                 y luego llamar .load_model().
        n_inner_steps: cuántos pasos del simulador se ejecutan por cada
                       acción del agente RL (N del paper). Default 32.
        max_outer_steps: cuántas decisiones puede tomar el agente RL en un
                         episodio. Equivalente a `skill.max_steps // n_inner_steps`.
                         Default 2 (= 64/32).
        mu_change_range: rango ± para Δμ. Acota la acción del agente RL.
                         Default 0.03 (paper).
        priors_change_range: rango ± para Δπ. Default 0.0 (paper no los modifica).
        modify_covariances: si True, también se aceptan ΔΣ en la acción.
                            Default False (paper no los modifica).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        calvin_env,
        gmm_cfg: DictConfig,
        kp_mock_cfg: DictConfig,
        demos_target: np.ndarray,
        n_inner_steps: int = 32,
        max_outer_steps: int = 2,
        mu_change_range: float = 0.03,
        priors_change_range: float = 0.0,
        modify_covariances: bool = False,
    ):
        super().__init__()
        self.env = calvin_env
        self.n_inner_steps = n_inner_steps
        self.max_outer_steps = max_outer_steps
        self.mu_change_range = mu_change_range
        self.priors_change_range = priors_change_range
        self.modify_covariances = modify_covariances

        # === Cargar el GMM base ===
        self.gmm_base = hydra.utils.instantiate(gmm_cfg)
        self.gmm_base.load_model()

        # === Keypoint mock para detectar target dinámico ===
        # Replica la lógica del CALVINAgent canónico (src/sac_gmm/agents/agent.py:52):
        # el target del GMM no es estático sino keypoint actual del objeto + un shift
        # que alinea la pose actual con la pose media del dataset (`demos_target`).
        self.demos_target = np.asarray(demos_target, dtype=np.float32)
        self.kp_mock = hydra.utils.instantiate(
            kp_mock_cfg, env_is_source=self.env.is_source
        )
        self.kp_target_shift = None  # se calcula en reset(), depende de gt_keypoint
        # gmm.priors: (K,), gmm.means: (K, 2*S), gmm.covariances: (K, 2*S, 2*S)
        # state_type='pos' → S=3 → means es (K, 6), covariances (K, 6, 6)
        self.K = self.gmm_base.n_components
        self.state_dim = self.gmm_base.dim  # 3 para state_type='pos'

        # === Calcular dims de Δθ ===
        # Δπ: K dims (si priors_change_range > 0)
        # Δμ: K * 2 * state_dim dims (cada media es 2*state_dim = 6 floats)
        # ΔΣ: K * (2*state_dim) * (2*state_dim) dims (si modify_covariances)
        self._dim_dpi = self.K if self.priors_change_range > 0 else 0
        self._dim_dmu = self.K * 2 * self.state_dim  # paper: solo μ
        self._dim_dsigma = (
            self.K * (2 * self.state_dim) * (2 * self.state_dim)
            if self.modify_covariances
            else 0
        )
        action_dim = self._dim_dpi + self._dim_dmu + self._dim_dsigma

        # Action space: vector en [-1, 1] que escalamos internamente al range
        # correspondiente. SB3 trabaja bien con bounds simétricos en [-1, 1].
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )

        # Observation space: position 3D (igual que SAC SB3 puro)
        self.observation_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # Estado interno del wrapper
        self._outer_step_count = 0
        self._gmm_active = None  # copia mutable del GMM base
        self._obs_dict = None

    def detect_target(self) -> np.ndarray:
        """Posición target para el GMM, replicando GMMAgent.detect_target.

        Equivalente a `src/sac_gmm/agents/calvin/gmm_agent.py:47-54`:
            keypoint_out = self.kp_mock.keypoint(np.zeros(1))
            keypoint_out = self.kp_mock.to_world(keypoint_out).squeeze()
            keypoint_pos = keypoint_out[: self.kp_mock.dim - 1]
            return keypoint_pos + self.kp_target_shift

        El GMM se entrenó con `goal_centered=True` (X = pos - goal_demo), por lo
        que en inferencia hay que pasarle `position - target`, donde target =
        keypoint actual del objeto + shift al frame del dataset.
        """
        keypoint_out = self.kp_mock.keypoint(np.zeros(1))
        keypoint_out = self.kp_mock.to_world(keypoint_out).squeeze()
        keypoint_pos = keypoint_out[: self.kp_mock.dim - 1]
        return np.asarray(keypoint_pos + self.kp_target_shift, dtype=np.float32)

    def _extract_obs(self, obs_dict) -> np.ndarray:
        """Extrae la posición 3D del dict de obs de CalvinSkillEnv."""
        if isinstance(obs_dict, dict):
            pos = obs_dict.get("position", np.zeros(3))
            return np.asarray(pos, dtype=np.float32)
        return np.asarray(obs_dict, dtype=np.float32)

    def _decode_action(self, action: np.ndarray):
        """Convierte la acción en [-1,1]^action_dim a Δπ, Δμ, ΔΣ."""
        action = np.asarray(action, dtype=np.float64).flatten()
        idx = 0

        if self._dim_dpi > 0:
            dpi = action[idx: idx + self._dim_dpi] * self.priors_change_range
            idx += self._dim_dpi
        else:
            dpi = None

        dmu = action[idx: idx + self._dim_dmu] * self.mu_change_range
        idx += self._dim_dmu
        dmu = dmu.reshape(self.K, 2 * self.state_dim)

        if self._dim_dsigma > 0:
            dsigma = action[idx: idx + self._dim_dsigma].reshape(
                self.K, 2 * self.state_dim, 2 * self.state_dim
            )
        else:
            dsigma = None

        return dpi, dmu, dsigma

    def _apply_delta_to_gmm(self, dpi, dmu, dsigma):
        """Aplica Δθ a una copia fresca del GMM base."""
        gmm = copy.deepcopy(self.gmm_base)

        # gmm.means viene en shape generic (K, 2*S). Aplicar Δμ:
        if hasattr(gmm, "means"):
            gmm.means = gmm.means + dmu

        if dpi is not None and hasattr(gmm, "priors"):
            new_priors = gmm.priors + dpi
            # Re-normalizar para que sumen 1 (clipping negativos a 0)
            new_priors = np.clip(new_priors, 1e-6, None)
            new_priors = new_priors / new_priors.sum()
            gmm.priors = new_priors

        if dsigma is not None and hasattr(gmm, "covariances"):
            # Sumar y proyectar a PSD (símmetrize + clip eigenvalues)
            new_cov = gmm.covariances + 0.5 * (dsigma + dsigma.transpose(0, 2, 1))
            # Por simplicidad, no se hace projection PSD aquí (puede romper);
            # mejor mantener modify_covariances=False por default.
            gmm.covariances = new_cov

        # Para BayesianGMM, reconstruir el objeto sklearn interno:
        if hasattr(gmm, "reshape_params"):
            try:
                gmm.reshape_params(to="gmr-specific")
            except Exception:
                pass

        return gmm

    def reset(self, *, seed: Optional[int] = None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self._obs_dict = self.env.reset()
        self._outer_step_count = 0
        self._gmm_active = copy.deepcopy(self.gmm_base)

        # === Keypoint mock reset (replica de Agent.reset_mock en agent.py:74-78) ===
        # 1) Anclar el keypoint en la pose ground-truth del objeto (la pone skill_env
        #    en su reset, vía set_gt_keypoint()).
        gt_keypoint = self.env.gt_keypoint
        self.kp_mock.reset_gt(gt_keypoint)
        # 2) shift = pose-media-dataset - pose-actual-init: corrige la diferencia
        #    entre el frame de demos y el frame del episodio actual.
        self.kp_target_shift = self.demos_target - np.asarray(
            self.kp_mock.init_pos, dtype=np.float32
        )
        # 3) Muestrear pos inicial (sin ruido si env_is_source, con ruido si no).
        self.kp_mock.reset_position()

        return self._extract_obs(self._obs_dict), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        # 1) Decodificar Δθ desde la acción del agente
        dpi, dmu, dsigma = self._decode_action(action)

        # 2) Aplicar Δθ → nuevo GMM activo para los próximos N pasos
        self._gmm_active = self._apply_delta_to_gmm(dpi, dmu, dsigma)

        # 3) Rolloutear N pasos del simulador
        accumulated_reward = 0.0
        last_info = {}
        terminated = False
        # Target dinámico: keypoint actual del objeto + shift (replica CALVINAgent
        # canónico: se recomputa una vez por outer step, no por inner step).
        target_pos = self.detect_target()

        for inner in range(self.n_inner_steps):
            pos = self._extract_obs(self._obs_dict)
            try:
                dx = self._gmm_active.predict(pos - target_pos)
            except Exception:
                # Fallback: si el GMM tiene problemas (PSD violado, etc.), no
                # movernos.
                dx = np.zeros(3, dtype=np.float32)

            dx = np.clip(np.asarray(dx, dtype=np.float32), -1.0, 1.0)
            obs_dict, reward, done, info = self.env.step(dx)
            self._obs_dict = obs_dict
            accumulated_reward += float(reward)
            last_info = info or {}
            if done:
                terminated = True
                break

        # 4) Avanzar el outer step counter
        self._outer_step_count += 1
        truncated = (self._outer_step_count >= self.max_outer_steps) and not terminated

        obs_out = self._extract_obs(self._obs_dict)
        return obs_out, accumulated_reward, terminated, truncated, last_info

    def render(self):
        return self.env.render()

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()
