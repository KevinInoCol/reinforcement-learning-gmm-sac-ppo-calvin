"""
Figuras DIDÁTICAS / ILUSTRATIVAS para a apresentação (dados sintéticos, NÃO são
do projeto). Replica a série de 3 gráficos de Imitation Learning com GMM/GMR:
  ① COLETA DE DEMONSTRAÇÕES
  ② MODELAGEM COM GMM
  ③ REPRODUÇÃO COM GMR
+ um diagrama do pipeline SAC-GMM vs PPO-GMM.

Saída: "Figuras para Presentacion/".
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "Figuras para Presentacion"
OUT.mkdir(exist_ok=True)

BG = "#f2f9f3"          # fundo verde bem claro
XLABEL = "Posição X no Workspace do Robô"
YLABEL = "Posição Y no Workspace do Robô"
XLIM = (-0.6, 8.6)
YLIM = (-0.1, 4.6)

DEMO_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff9f1c"]   # 1 vermelho 2 azul 3 verde 4 laranja
DEMO_START_MARKERS = ["o", "D", "^", "s"]
GMM_COLORS = ["#e377c2", "#17becf", "#1f77b4", "#bcbd22", "#ff9896", "#2ca02c", "#9467bd"]

rng = np.random.default_rng(7)

# ---- Curva "verdadeira" (skill) em forma de S ----
xg = np.linspace(0.45, 7.25, 80)
yg = 2.5 + 1.05 * np.sin(0.83 * xg - 0.08)


def make_demos():
    demos = []
    offs = [0.18, -0.05, -0.22, 0.05]
    for k in range(4):
        walk = np.cumsum(rng.normal(0, 0.05, len(xg)))
        walk -= np.linspace(walk[0], walk[-1], len(xg))   # detrend
        yd = yg + 0.9 * walk + rng.normal(0, 0.10, len(xg)) + offs[k]
        xd = xg + rng.normal(0, 0.04, len(xg))
        demos.append((xd, yd))
    return demos


def gmm_components():
    """Centros + elipses ao longo da curva (orientadas pela tangente)."""
    xc = np.array([0.8, 1.9, 2.9, 3.9, 4.9, 5.9, 6.9])
    yc = 2.5 + 1.05 * np.sin(0.83 * xc - 0.08)
    slope = 1.05 * 0.83 * np.cos(0.83 * xc - 0.08)
    ang = np.degrees(np.arctan(slope))
    w = np.array([1.9, 1.7, 1.8, 1.9, 1.8, 1.7, 1.9])
    h = np.array([1.25, 1.15, 1.2, 1.25, 1.2, 1.15, 1.2])
    return xc, yc, ang, w, h


def base_ax(ax, title, banner_color):
    ax.set_facecolor(BG)
    ax.set_xlim(*XLIM); ax.set_ylim(*YLIM)
    ax.set_xlabel(XLABEL, fontsize=11, fontweight="bold")
    ax.set_ylabel(YLABEL, fontsize=11, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45, color="0.7")
    ax.set_axisbelow(True)
    ax.text(0.5, 1.07, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=15, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.45", facecolor=banner_color,
                      edgecolor="white", linewidth=2.5))


def fig_demos(save=True, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    base_ax(ax, "①  COLETA DE DEMONSTRAÇÕES", "#e0392b")
    demos = make_demos()
    for k, (xd, yd) in enumerate(demos):
        ax.plot(xd, yd, color=DEMO_COLORS[k], lw=1.4, alpha=0.85,
                label=f"Demonstração {k+1}")
        ax.scatter(xd[0], yd[0], marker=DEMO_START_MARKERS[k], s=120,
                   color=DEMO_COLORS[k], edgecolor="k", zorder=5)
        ax.scatter(xd[-1], yd[-1], marker="X", s=110,
                   color=DEMO_COLORS[k], edgecolor="k", zorder=5)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    if standalone:
        fig.patch.set_facecolor("white"); plt.tight_layout()
        plt.savefig(OUT / "01_coleta_demonstracoes.png", dpi=150, facecolor="white")
        plt.close()
        print("   ✓ 01_coleta_demonstracoes.png")


def fig_gmm(save=True, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    base_ax(ax, "②  MODELAGEM COM GMM", "#2b6fe0")
    demos = make_demos()
    for xd, yd in demos:
        ax.scatter(xd, yd, s=6, color="0.45", alpha=0.5, zorder=2)
    xc, yc, ang, w, h = gmm_components()
    for i in range(len(xc)):
        e = Ellipse((xc[i], yc[i]), width=w[i], height=h[i], angle=ang[i],
                    facecolor=GMM_COLORS[i % len(GMM_COLORS)], alpha=0.30,
                    edgecolor=GMM_COLORS[i % len(GMM_COLORS)], lw=1.5, zorder=3)
        ax.add_patch(e)
        ax.scatter(xc[i], yc[i], s=45, facecolor="white",
                   edgecolor=GMM_COLORS[i % len(GMM_COLORS)], lw=2, zorder=4)
    ax.scatter([], [], s=10, color="0.45", label="Pontos demonstrados")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    if standalone:
        fig.patch.set_facecolor("white"); plt.tight_layout()
        plt.savefig(OUT / "02_modelagem_gmm.png", dpi=150, facecolor="white")
        plt.close()
        print("   ✓ 02_modelagem_gmm.png")


def fig_gmr(save=True, ax=None):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(9, 5))
    base_ax(ax, "③  REPRODUÇÃO COM GMR", "#27ae60")
    # demos e elipses esmaecidas ao fundo
    for xd, yd in make_demos():
        ax.plot(xd, yd, color="0.6", lw=1.0, alpha=0.30,
                label="_nolegend_")
    xc, yc, ang, w, h = gmm_components()
    for i in range(len(xc)):
        ax.add_patch(Ellipse((xc[i], yc[i]), width=w[i], height=h[i], angle=ang[i],
                     facecolor=GMM_COLORS[i % len(GMM_COLORS)], alpha=0.13,
                     edgecolor="none", zorder=2))
    # trajetória reproduzida (GMR): curva suave verde grossa
    ax.plot(xg, yg, color="#2ecc71", lw=8, solid_capstyle="round",
            zorder=4, label="Trajetória Reproduzida (GMR)")
    # setas ao longo do caminho
    idxs = np.linspace(8, len(xg) - 8, 7).astype(int)
    for i in idxs:
        ax.annotate("", xy=(xg[i + 2], yg[i + 2]), xytext=(xg[i], yg[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#1e8449", lw=2.2), zorder=5)
    ax.plot([], [], color="0.6", lw=1.2, alpha=0.6, label="Demonstrações originais")
    # início (círculo) e alvo (X)
    ax.scatter(xg[0], yg[0], s=300, facecolor="#2ecc71", edgecolor="k", lw=2, zorder=6)
    ax.scatter(xg[-1], yg[-1], marker="X", s=420, facecolor="#1e8449",
               edgecolor="k", lw=2, zorder=6)
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: "Demonstr" not in labels[i])
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper right", fontsize=8, framealpha=0.9)
    if standalone:
        fig.patch.set_facecolor("white"); plt.tight_layout()
        plt.savefig(OUT / "03_reproducao_gmr.png", dpi=150, facecolor="white")
        plt.close()
        print("   ✓ 03_reproducao_gmr.png")


def fig_serie():
    fig, axes = plt.subplots(1, 3, figsize=(22, 5.2))
    fig_demos(ax=axes[0]); fig_gmm(ax=axes[1]); fig_gmr(ax=axes[2])
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    plt.savefig(OUT / "00_serie_completa.png", dpi=140, facecolor="white")
    plt.close()
    print("   ✓ 00_serie_completa.png")


# ---------------------------------------------------------------------------
# Diagrama do pipeline SAC-GMM vs PPO-GMM
# ---------------------------------------------------------------------------
def box(ax, xy, w, h, text, fc, ec="k", fs=11, tc="k", bold=True):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                 facecolor=fc, edgecolor=ec, lw=2, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, fontweight="bold" if bold else "normal", color=tc, zorder=4)


def arrow(ax, p1, p2, text="", color="#333", rad=0.0, fs=9, off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=20,
                 lw=2.2, color=color, connectionstyle=f"arc3,rad={rad}", zorder=2))
    if text:
        mx, my = (p1[0] + p2[0]) / 2 + off[0], (p1[1] + p2[1]) / 2 + off[1]
        ax.text(mx, my, text, ha="center", va="center", fontsize=fs,
                style="italic", color=color, zorder=5)


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 7.4); ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(6.5, 7.05, "Como funcionam SAC-GMM e PPO-GMM",
            ha="center", fontsize=17, fontweight="bold")
    ax.text(6.5, 6.65, "(o RL refina os parâmetros θ de um GMM aprendido por demonstração)",
            ha="center", fontsize=10, style="italic", color="0.35")

    # Demonstrações -> GMM
    box(ax, (0.4, 5.3), 2.6, 0.9, "Demonstrações\nHumanas", "#fde2cf")
    box(ax, (4.2, 5.25), 3.0, 1.0, "GMM\nSistema Dinâmico  (θ)", "#d6e4ff")
    arrow(ax, (3.0, 5.75), (4.2, 5.75), "ajusta (EM)", off=(0, 0.18))

    # GMM -> Ambiente
    box(ax, (4.2, 2.7), 3.0, 1.0, "Ambiente CALVIN\n(N passos de simulação)", "#e8f6e9")
    arrow(ax, (5.7, 5.25), (5.7, 3.7), "gera velocidade  ξ̇", rad=0.0, off=(1.55, 0))

    # Ambiente -> Agente RL
    box(ax, (9.0, 2.65), 3.4, 1.15, "Agente RL", "#fff3cd", fs=12)
    arrow(ax, (7.2, 3.2), (9.0, 3.2),
          "estado  s$_t$  +  recompensa\nesparsa  r$_t$ (+10 / 0)", off=(0, 0.42), fs=8.5)

    # Agente RL -> GMM (correção Δθ) — laço de refinamento
    arrow(ax, (10.7, 3.8), (6.7, 6.25), "Δθ:  ajusta  μ, Σ, π  do GMM",
          color="#c0392b", rad=-0.32, off=(0.2, 0.35))

    # Dois algoritmos de RL (variantes)
    box(ax, (8.55, 0.55), 1.95, 1.4,
        "SAC\noff-policy\nreplay buffer\nmax-entropia", "#d5f5e3", fs=9)
    box(ax, (10.75, 0.55), 1.95, 1.4,
        "PPO\non-policy\nclipping\nentropy bonus", "#fdebd0", fs=9)
    arrow(ax, (9.5, 1.95), (10.0, 2.65), color="#1e8449", rad=0.0)
    arrow(ax, (11.7, 1.95), (11.4, 2.65), color="#b9770e", rad=0.0)
    ax.text(9.52, 0.28, "SAC-GMM", ha="center", fontsize=9, fontweight="bold", color="#1e8449")
    ax.text(11.72, 0.28, "PPO-GMM", ha="center", fontsize=9, fontweight="bold", color="#b9770e")

    # nota do laço (perto da curva vermelha do Δθ, para não colidir com 'gera velocidade')
    ax.text(9.6, 5.6, "laço de refinamento\n(a cada N passos)", ha="center",
            fontsize=9, style="italic", color="#c0392b")

    plt.tight_layout()
    plt.savefig(OUT / "diagrama_SACGMM_vs_PPOGMM.png", dpi=150, facecolor="white")
    plt.close()
    print("   ✓ diagrama_SACGMM_vs_PPOGMM.png")


if __name__ == "__main__":
    print("Gerando figuras ilustrativas ...")
    fig_demos(); fig_gmm(); fig_gmr(); fig_serie(); fig_pipeline()
    print(f"\nListo. Figuras em: {OUT}")
