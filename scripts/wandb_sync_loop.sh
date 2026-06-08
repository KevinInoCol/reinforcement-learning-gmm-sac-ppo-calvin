#!/bin/bash
# Loop de sync de W&B para correr en el HEADNODE de RECOD (que sí tiene internet).
# Los entrenamientos corren en modo offline en los nodos de cómputo (sin internet);
# este loop sube al dashboard el run offline MÁS RECIENTE cada N segundos, para
# verlo "de rato en rato" casi en vivo.
#
# Uso (en headnode, dentro de tmux para que sobreviva a la desconexión):
#   tmux new-session -d -s wandbsync 'bash ~/Project-Reinforment-Learning/scripts/wandb_sync_loop.sh 180'
#   tmux attach -t wandbsync     # para verlo
#   tmux kill-session -t wandbsync   # para pararlo
source ~/miniconda/etc/profile.d/conda.sh
conda activate UNICAMP-Project-RL-SAC-GMM

INTERVAL="${1:-180}"
echo "[wandb_sync_loop] sincronizando el run offline más reciente cada ${INTERVAL}s. Ctrl-C para parar."
while true; do
    latest=$(ls -dt ~/wandb/wandb/offline-run-*/ 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "[wandb_sync_loop] $(date '+%H:%M:%S') sync $latest"
        wandb sync "$latest" 2>&1 | tail -3
    else
        echo "[wandb_sync_loop] $(date '+%H:%M:%S') aún no hay offline-run en ~/wandb/wandb/"
    fi
    sleep "$INTERVAL"
done
