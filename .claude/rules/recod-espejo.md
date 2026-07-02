# Regla: RECOD es un espejo de lo local (Mac = fuente de verdad)

**El repositorio local en Mac es la ÚNICA fuente de verdad. RECOD solo ejecuta.**

## Qué significa en la práctica

1. **Todo archivo de código/config/sbatch se crea y edita en Mac**, se commitea y
   pushea a `kevin main`, y llega a RECOD **solo** vía `git pull`. Nunca crear ni
   editar archivos directamente en RECOD (ni con `cat > archivo` por SSH).
2. Si por urgencia algo se creó directo en RECOD (p. ej. un sbatch), hay que
   **espejarlo de vuelta al repo inmediatamente**: copiarlo a Mac, commitear,
   pushear, y en RECOD borrar la copia untracked y hacer `pull` para que quede
   trackeada.
3. **Excepciones (lo único que puede vivir solo en RECOD):** salidas de ejecución —
   `logs/` de SLURM, runs offline de W&B, checkpoints intermedios de training, y
   los ajustes locales de máquina en archivos ya trackeados (`config/setup.yaml`
   con device/root propios de cada máquina, que NUNCA se commitean desde ninguna
   de las dos).
4. Antes de lanzar un job en RECOD, verificar que el sbatch que se va a usar
   **existe en el repo** (`git ls-files | grep sbatch`). Si no está, primero
   espejarlo (regla 2).
5. Al hacer `git pull` en RECOD usar `--no-rebase`; si el pull falla por
   "untracked working tree files would be overwritten", casi siempre son salidas
   generadas en RECOD que ya subimos desde Mac: verificar que sean idénticas/
   regenerables y borrar las copias untracked de RECOD antes de reintentar.

## Por qué

- Evita divergencia silenciosa entre lo que creemos que corre y lo que corre.
- Todo experimento queda reproducible desde el repo (clave para el paper).
- Ya nos pasó: `run_sac_gmm.sbatch` vivía solo en RECOD y hubo que
  reconstruir su procedencia; y `logs/gmm_ppo_8h/` fue sobrescrito por un run
  nuevo porque el sbatch (solo-RECOD) apuntaba a la misma carpeta.
