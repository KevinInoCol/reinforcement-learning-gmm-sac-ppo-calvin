"""
Copia de agent_eval.py con grabación de video y guardado de métricas.

Uso:
    python3 scripts/agent_eval_record.py [overrides...]

Salida (por default todo bajo Output_Inference/):
    Output_Inference/
      ├─ videos/eval_<agent>_<skill>_<YYYYmmdd_HHMMSS>.mp4
      └─ results_table/
           ├─ eval_results.csv        (acumulado, una fila por run)
           └─ eval_<agent>_<skill>_<ts>.json   (detalle por run)

Override de paths con env vars:
    SACGMM_OUTPUT_DIR    parent común (default: Output_Inference/)
    SACGMM_VIDEO_PATH    archivo MP4 específico
    SACGMM_RESULTS_DIR   carpeta de resultados
    SACGMM_RESULTS_CSV   CSV específico

Requisitos:
    - PyBullet con soporte MP4 (necesita ffmpeg en PATH; en Mac:
      `brew install ffmpeg`)
    - show_gui=true en los overrides (la grabación funciona en modo GUI)
"""
import os
import sys
import shutil
import tempfile
import json
import wandb
import hydra
import logging
import csv
import datetime
import torch
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning import seed_everything
from sac_gmm.utils.utils import print_system_env_info, setup_logger, get_last_checkpoint
from hydra.core.hydra_config import HydraConfig
from hydra import compose

from sac_gmm.models.sac_gmm_model import SACGMM
from sac_gmm.models.kis_gmm_model import KISGMM


cwd_path = Path(__file__).absolute().parents[0]
sac_gmm_path = cwd_path.parents[0]
root = sac_gmm_path.parents[0]

# This is to access the locally installed repo clone when using slurm
sys.path.insert(0, sac_gmm_path.as_posix())  # sac_gmm
sys.path.insert(0, os.path.join(root, "calvin_env"))  # root/calvin_env
sys.path.insert(0, root.as_posix())  # root


logger = logging.getLogger(__name__)


@rank_zero_only
def log_rank_0(*args, **kwargs):
    # when using ddp, only log with rank 0 process
    logger.info(*args, **kwargs)


OUTPUT_DIR = Path(os.environ.get("SACGMM_OUTPUT_DIR", sac_gmm_path / "Output_Inference"))


def _setup_camera(agent):
    """Ajusta la cámara de PyBullet para arrancar zoom-in en la mesa+robot.

    Override con env vars:
        SACGMM_CAM_DIST    default 1.2
        SACGMM_CAM_YAW     default 50
        SACGMM_CAM_PITCH   default -30
        SACGMM_CAM_TARGET  default "0.0,-0.2,0.5"  (x,y,z separados por coma)
    """
    try:
        pb_client = agent.env.p
        dist = float(os.environ.get("SACGMM_CAM_DIST", "1.2"))
        yaw = float(os.environ.get("SACGMM_CAM_YAW", "50"))
        pitch = float(os.environ.get("SACGMM_CAM_PITCH", "-30"))
        target_str = os.environ.get("SACGMM_CAM_TARGET", "0.0,-0.2,0.5")
        target = [float(v) for v in target_str.split(",")]
        pb_client.resetDebugVisualizerCamera(
            cameraDistance=dist,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=target,
        )
        log_rank_0(
            f"🎥 Cámara: dist={dist}, yaw={yaw}, pitch={pitch}, target={target}"
        )
    except Exception as e:
        log_rank_0(f"⚠️  No se pudo ajustar la cámara: {e}")


def _resolve_video_path(cfg: DictConfig) -> Path:
    """Determina dónde guardar el MP4."""
    env_path = os.environ.get("SACGMM_VIDEO_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"eval_{cfg.agent.name}_{cfg.skill.name}_{timestamp}.mp4"
    videos_dir = OUTPUT_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    return videos_dir / fname


def _start_video(agent, video_path: Path):
    """Arranca el state-logging MP4 de PyBullet.

    PyBullet pasa la ruta a ffmpeg SIN escapar espacios, así que si el
    destino final tiene espacios (ej. "UNICAMP DOCTORADO"), grabamos
    primero a /tmp con un nombre sin espacios y luego en _stop_video
    movemos al destino real.
    """
    # Ruta temporal sin espacios para PyBullet/ffmpeg
    tmp_dir = Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"sacgmm_eval_{os.getpid()}_{video_path.name}"
    try:
        pb_client = agent.env.p  # BulletClient o módulo pybullet
        log_id = pb_client.startStateLogging(
            pb_client.STATE_LOGGING_VIDEO_MP4,
            tmp_path.as_posix(),
        )
        log_rank_0(f"📹 Grabando video → {video_path}")
        log_rank_0(f"   (archivo temporal: {tmp_path})")
        return pb_client, log_id, tmp_path
    except Exception as e:
        log_rank_0(f"⚠️  No se pudo iniciar grabación: {e}")
        log_rank_0(
            "   Si querés video MP4 nativo de PyBullet, instalá ffmpeg "
            "(`brew install ffmpeg` en Mac) y reintenta."
        )
        return None, None, None


def _save_results(cfg: DictConfig, results: dict, video_path: Path):
    """Guarda resultados en JSON (por-run) y CSV (acumulado).

    Override paths con env vars:
      SACGMM_OUTPUT_DIR    (default: Output_Inference/)
      SACGMM_RESULTS_DIR   (default: Output_Inference/results_table/)
      SACGMM_RESULTS_CSV   (default: Output_Inference/results_table/eval_results.csv)
    """
    results_dir = Path(os.environ.get("SACGMM_RESULTS_DIR", OUTPUT_DIR / "results_table"))
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{cfg.agent.name}_{cfg.skill.name}_{timestamp}"

    # 1) JSON per-run (detalle completo)
    json_path = results_dir / f"eval_{run_id}.json"
    payload = {
        "timestamp": timestamp,
        "agent": cfg.agent.name,
        "skill": cfg.skill.name,
        "env": str(cfg.env.env_name) if hasattr(cfg.env, "env_name") else None,
        "num_eval_episodes": cfg.num_eval_episodes,
        "num_eval_seeds": cfg.num_eval_seeds,
        "n_components_gmm": cfg.skill.n_components,
        "chk_dir": cfg.chk_dir if cfg.chk_dir else None,
        "video_path": video_path.as_posix() if video_path else None,
        **results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    log_rank_0(f"📊 JSON guardado: {json_path}")

    # 2) CSV acumulado (para tabla)
    csv_path = Path(os.environ.get("SACGMM_RESULTS_CSV", results_dir / "eval_results.csv"))
    csv_headers = [
        "timestamp", "agent", "skill", "num_eval_episodes", "num_eval_seeds",
        "n_components_gmm", "accuracy_mean", "accuracy_var", "accuracy_per_seed",
        "return_mean", "length_mean", "chk_dir", "video_path",
    ]
    write_header = not csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(csv_headers)
        writer.writerow([
            timestamp,
            cfg.agent.name,
            cfg.skill.name,
            cfg.num_eval_episodes,
            cfg.num_eval_seeds,
            cfg.skill.n_components,
            results.get("accuracy_mean"),
            results.get("accuracy_var"),
            json.dumps(results.get("accuracy_per_seed", [])),
            results.get("return_mean"),
            results.get("length_mean"),
            cfg.chk_dir if cfg.chk_dir else "",
            video_path.as_posix() if video_path else "",
        ])
    log_rank_0(f"📊 CSV actualizado: {csv_path}")


def _stop_video(pb_client, log_id, tmp_path: Path, video_path: Path):
    if pb_client is None or log_id is None:
        return
    try:
        pb_client.stopStateLogging(log_id)
        # Mover del archivo temporal al destino final (con espacios OK)
        if tmp_path is not None and tmp_path.exists():
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(tmp_path.as_posix(), video_path.as_posix())
            size_mb = video_path.stat().st_size / (1024 * 1024)
            log_rank_0(f"✅ Video guardado: {video_path}  ({size_mb:.1f} MB)")
        else:
            log_rank_0(
                f"⚠️  PyBullet terminó la grabación pero no encontré {tmp_path}.\n"
                f"   Verificá que ffmpeg esté en el PATH. "
                f"Usá Cmd+Shift+5 como alternativa."
            )
    except Exception as e:
        log_rank_0(f"⚠️  Error al detener grabación: {e}")


def run_test(cfg: DictConfig) -> None:
    log_rank_0(
        f"Evaluating {cfg.agent.name} for {cfg.agent.skill.name} skill with the config:\n{OmegaConf.to_yaml(cfg)}"
    )
    log_rank_0(print_system_env_info())

    if cfg.agent.name == "GMM":
        actor = None
        agent = hydra.utils.instantiate(cfg.agent)
    elif cfg.agent.name in ["SACGMM", "KISGMM"]:
        chk = Path(cfg.chk_dir)
        if chk is not None:
            if cfg.agent.name == "SACGMM":
                model = SACGMM
            elif cfg.agent.name == "KISGMM":
                model = KISGMM
            model = model.load_from_checkpoint(checkpoint_path=chk.as_posix(), agent=cfg.agent).to(cfg.device)
            agent = model.agent
            actor = model.actor
        else:
            raise ValueError("Model not loaded correctly.")

    agent.num_eval_episodes = cfg.num_eval_episodes

    # === Ajustar cámara antes de empezar la grabación ===
    _setup_camera(agent)

    # === Arrancar grabación ===
    video_path = _resolve_video_path(cfg)
    pb_client, log_id, tmp_path = _start_video(agent, video_path)

    accs, returns, lengths = [], [], []
    try:
        for s in range(cfg.num_eval_seeds):
            seed_everything(cfg.seed + s, workers=True)
            eval_accuracy, eval_return, eval_length = agent.evaluate(actor, device=cfg.device)
            accs.append(float(eval_accuracy))
            returns.append(float(eval_return) if eval_return is not None else None)
            lengths.append(float(eval_length) if eval_length is not None else None)
    finally:
        # Asegura cerrar la grabación incluso si hay excepción
        _stop_video(pb_client, log_id, tmp_path, video_path)

    accs_np = np.array(accs)
    mean_acc = float(np.mean(accs_np))
    var_acc = float(np.var(accs_np))

    results = {
        "accuracy_per_seed": accs,
        "accuracy_mean": mean_acc,
        "accuracy_var": var_acc,
        "return_per_seed": returns,
        "return_mean": float(np.mean([r for r in returns if r is not None])) if any(r is not None for r in returns) else None,
        "length_per_seed": lengths,
        "length_mean": float(np.mean([l for l in lengths if l is not None])) if any(l is not None for l in lengths) else None,
    }

    log_rank_0(f"=== Resumen de evaluación ===")
    log_rank_0(f"  Accuracy:  mean={mean_acc:.4f}, var={var_acc:.4f}, per_seed={accs}")
    log_rank_0(f"  Return:    {results['return_mean']}")
    log_rank_0(f"  Length:    {results['length_mean']}")

    _save_results(cfg, results, video_path)


@hydra.main(version_base="1.1", config_path=str(sac_gmm_path / "config"), config_name="agent_eval")
def eval_agent(cfg: DictConfig) -> None:
    run_test(cfg)


if __name__ == "__main__":
    eval_agent()
