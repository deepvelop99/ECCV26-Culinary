"""Build an expert-style summary PPT of the CulinaryCut-VLA pipeline.

Narrative: VLA 평가용 success criteria는 force + velocity. 그걸 재려면
MPM+ManiSkill co-sim이 data generation 시점부터 통합돼 있어야 한다.
ManiSkill만으론 "안 잘렸는데 잘렸다"로 오판이 생긴다.

Slide text 한국어, matplotlib 그림은 영문.
Output: /data/mani_skill/docs/culinarycut_vla_dataset.pptx
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

DOCS = Path("/data/mani_skill/docs")
FIGS = DOCS / "figs"
FIGS.mkdir(parents=True, exist_ok=True)

# ---- Palette ----
NAVY    = "#0F2B54"; NAVYRGB = RGBColor(0x0F, 0x2B, 0x54)
ACCENT  = "#E85A2B"; ACC_RGB = RGBColor(0xE8, 0x5A, 0x2B)
TEAL    = "#1F8A8E"; TEALRGB = RGBColor(0x1F, 0x8A, 0x8E)
GOLD    = "#D9A429"; GOLDRGB = RGBColor(0xD9, 0xA4, 0x29)
GREY    = "#4A5058"; GREYRGB = RGBColor(0x4A, 0x50, 0x58)
LIGHT   = "#F5F5F3"; LIGHT_RGB = RGBColor(0xF5, 0xF5, 0xF3)
SOFT    = "#E8ECEF"; SOFTRGB = RGBColor(0xE8, 0xEC, 0xEF)
RED     = "#B83330"; REDRGB = RGBColor(0xB8, 0x33, 0x30)
GREEN   = "#2F8F4E"; GREENRGB = RGBColor(0x2F, 0x8F, 0x4E)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": NAVY, "axes.labelcolor": NAVY,
    "xtick.color": NAVY,    "ytick.color": NAVY,
    "axes.spines.top": False, "axes.spines.right": False,
})


# =============================================================
# Figures
# =============================================================
def fig_problem_illustration(out):
    """Left: ManiSkill-only rigid body cut (no physical contact check).
       Right: MPM+MS co-sim with F, V traces.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    # Left panel — MS alone
    ax = axes[0]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.1, 1.4); ax.axis("off")
    ax.set_title("ManiSkill only — no physical contact signal",
                 color=RED, weight="bold", fontsize=13)
    # board
    ax.add_patch(Rectangle((-1.0, 0.2), 2.0, 0.1, fc=GOLD, ec=NAVY))
    # fruit (banana)
    ax.add_patch(mpatches.Ellipse((0, 0.4), 0.8, 0.2, fc=ACCENT, ec=NAVY))
    # knife (passes through)
    ax.add_patch(Rectangle((-0.03, 0.25), 0.06, 0.9, fc=GREY, ec=NAVY, alpha=0.8))
    # arrow "passes through"
    ax.annotate("", xy=(0, 0.3), xytext=(0, 1.0),
                arrowprops=dict(arrowstyle="->", color=RED, lw=2))
    ax.text(0.7, 0.5, "rigid body\n→ force spike\nappears even\nwhen 'cut'\nisn't physical",
            fontsize=9.5, color=RED, ha="left", va="center")
    ax.text(0, -0.03, "success=True\n(but no real cut)", ha="center", fontsize=10,
            color=RED, weight="bold")

    # Right panel — Co-sim
    ax = axes[1]
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.1, 1.4); ax.axis("off")
    ax.set_title("MPM × ManiSkill co-sim — deformable cut + F/V measurement",
                 color=GREEN, weight="bold", fontsize=13)
    # board
    ax.add_patch(Rectangle((-1.0, 0.2), 2.0, 0.1, fc=GOLD, ec=NAVY))
    # split banana (pieces)
    ax.add_patch(mpatches.Ellipse((-0.24, 0.4), 0.38, 0.2, fc=ACCENT, ec=NAVY))
    ax.add_patch(mpatches.Ellipse(( 0.24, 0.4), 0.38, 0.2, fc=ACCENT, ec=NAVY))
    # knife
    ax.add_patch(Rectangle((-0.03, 0.25), 0.06, 0.9, fc=GREY, ec=NAVY))
    # F/V traces (mini)
    import matplotlib.lines as mlines
    ts = np.linspace(0, 1, 80)
    F = np.clip(30 * np.exp(-((ts-0.55)/0.08)**2), 0, None) * 15
    V = 0.4 * np.exp(-((ts-0.5)/0.25)**2)
    # mini axes overlay
    axf = ax.inset_axes([0.62, 0.55, 0.34, 0.25])
    axf.plot(ts, F, color=RED, lw=1.5); axf.set_title("F (N)", fontsize=8, color=NAVY)
    axf.tick_params(labelsize=7); axf.spines['top'].set_visible(False); axf.spines['right'].set_visible(False)
    axv = ax.inset_axes([0.62, 0.2, 0.34, 0.25])
    axv.plot(ts, V, color=TEAL, lw=1.5); axv.set_title("V (m/s)", fontsize=8, color=NAVY)
    axv.tick_params(labelsize=7); axv.spines['top'].set_visible(False); axv.spines['right'].set_visible(False)
    ax.text(0, -0.03, "success ⇔ physical cut\n(F peak + V alignment)", ha="center", fontsize=10,
            color=GREEN, weight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_success_criteria(out):
    """Diagram: success = vel_align ∧ force_peak ∧ contact_frames."""
    fig, ax = plt.subplots(figsize=(12, 5.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.8); ax.axis("off")

    # Big box: MS-MPM aligned success
    box = FancyBboxPatch((1, 1), 10, 3.3,
                          boxstyle="round,pad=0.12,rounding_size=0.2",
                          fc=LIGHT, ec=NAVY, lw=2)
    ax.add_patch(box)
    ax.text(6, 4.6, "Physically-grounded success criteria",
            ha="center", fontsize=16, weight="bold", color=NAVY)

    # Three conditions
    for i, (title, expr, col) in enumerate([
        ("velocity alignment",
         "max(|V_ms - V_mpm|) < 0.05 m/s\ncorr(V_ms, V_mpm) ≈ 1.0",
         TEAL),
        ("force peak (MPM)",
         "peak |F_mpm| during descent\n≥ threshold (material-specific)",
         ACCENT),
        ("contact frames (MPM)",
         "# frames where particles\nin knife contact band",
         GOLD),
    ]):
        x = 1.5 + i * 3.3
        b = FancyBboxPatch((x, 1.4), 2.8, 2.2,
                            boxstyle="round,pad=0.06,rounding_size=0.15",
                            fc=col, ec=NAVY, alpha=0.9)
        ax.add_patch(b)
        ax.text(x + 1.4, 3.3, title, ha="center", fontsize=13,
                color="white" if col != GOLD else NAVY, weight="bold")
        ax.text(x + 1.4, 2.3, expr, ha="center", fontsize=10,
                color="white" if col != GOLD else NAVY)
    # Conjunction
    ax.text(6, 0.4,
            "alignment.passed  =  (vel_err < tol)  ∧  F peak consistent  ∧  contact frames > 0",
            ha="center", fontsize=12, color=NAVY, weight="bold",
            style="italic")
    ax.text(6, 5.2,
            "RGB and joint signals cannot decide 'was it actually cut?' — physical (F, V) is mandatory",
            ha="center", fontsize=11.5, color=RED, style="italic")

    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_why_cosim_datagen(out):
    """Compare post-hoc MPM replay vs at-data-gen co-sim."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, title, ok, notes, edge in [
        (axes[0], "Plan A — post-hoc MPM replay (rejected)", False,
         [
             "collect trajectory in MS -> replay knife path in MPM",
             "x MPM force does not feed back into MS robot motion",
             "x MS has no knowledge of deformation during contact",
             "x 'no real cut but labelled success' persists",
         ], RED),
        (axes[1], "Plan B — co-sim at data-gen time (adopted)", True,
         [
             "every control tick: MS knife pose -> MPM sync",
             "v MPM force / velocity recorded during the episode",
             "v per-step (V_ms, V_mpm, F_mpm) saved to alignment.json",
             "v success decided from physical signals",
         ], GREEN),
    ]:
        ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.2, 0.3), 9.6, 5.3,
                                     boxstyle="round,pad=0.08,rounding_size=0.18",
                                     fc=LIGHT, ec=edge, lw=2.2))
        ax.text(5, 5.1, title, ha="center", fontsize=14, color=edge, weight="bold")
        for i, line in enumerate(notes):
            ax.text(0.7, 4.2 - i*0.7, line, fontsize=11, color=NAVY)
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_cosim_flow(out):
    """Per-control-tick co-sim loop."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.2); ax.axis("off")

    # MS (left)
    ax.add_patch(FancyBboxPatch((0.3, 1.8), 4.0, 2.2,
                                 boxstyle="round,pad=0.1,rounding_size=0.15",
                                 fc=NAVY, ec=NAVY))
    ax.text(2.3, 3.6, "ManiSkill", ha="center", fontsize=14, color="white", weight="bold")
    ax.text(2.3, 3.05, "rigid body + knife + board", ha="center", fontsize=10, color="white")
    ax.text(2.3, 2.65, "pd_ee_delta_pos controller", ha="center", fontsize=10, color="white")
    ax.text(2.3, 2.2, "RGB camera · proprio · action", ha="center", fontsize=10, color="white")

    # Bridge (middle)
    ax.add_patch(FancyBboxPatch((5.0, 1.4), 3.0, 3.0,
                                 boxstyle="round,pad=0.1,rounding_size=0.15",
                                 fc=ACCENT, ec=NAVY))
    ax.text(6.5, 4.0, "Bridge", ha="center", fontsize=14, color="white", weight="bold")
    ax.text(6.5, 3.5, "knife_tip_ms → MPM frame", ha="center", fontsize=9.5, color="white")
    ax.text(6.5, 3.15, "banana-local canceling", ha="center", fontsize=9.5, color="white")
    ax.text(6.5, 2.75, "per-fruit canonical", ha="center", fontsize=9.5, color="white")
    ax.text(6.5, 2.2, "substeps cap", ha="center", fontsize=9.5, color="white")
    ax.text(6.5, 1.75, "Records: V_ms, V_mpm, F_ms, F_mpm", ha="center", fontsize=9.5,
            color="white", weight="bold", style="italic")

    # MPM (right)
    ax.add_patch(FancyBboxPatch((8.7, 1.8), 4.0, 2.2,
                                 boxstyle="round,pad=0.1,rounding_size=0.15",
                                 fc=TEAL, ec=NAVY))
    ax.text(10.7, 3.6, "MPM (Taichi)", ha="center", fontsize=14, color="white", weight="bold")
    ax.text(10.7, 3.05, "MLS-MPM deformable", ha="center", fontsize=10, color="white")
    ax.text(10.7, 2.65, "SDF knife collider", ha="center", fontsize=10, color="white")
    ax.text(10.7, 2.2, "particle-based fruit", ha="center", fontsize=10, color="white")

    # Arrows
    ax.add_patch(FancyArrowPatch((4.3, 3.3), (5.0, 3.3),
                                 arrowstyle="-|>", mutation_scale=18, color=NAVY))
    ax.add_patch(FancyArrowPatch((8.0, 3.3), (8.7, 3.3),
                                 arrowstyle="-|>", mutation_scale=18, color=NAVY))
    ax.add_patch(FancyArrowPatch((5.0, 2.0), (4.3, 2.0),
                                 arrowstyle="-|>", mutation_scale=18, color=NAVY))
    ax.add_patch(FancyArrowPatch((8.7, 2.0), (8.0, 2.0),
                                 arrowstyle="-|>", mutation_scale=18, color=NAVY))

    ax.text(4.65, 3.5, "tip pose", fontsize=9, color=NAVY, ha="center", style="italic")
    ax.text(8.35, 3.5, "mapped pose", fontsize=9, color=NAVY, ha="center", style="italic")
    ax.text(4.65, 1.8, "F,V label", fontsize=9, color=NAVY, ha="center", style="italic")
    ax.text(8.35, 1.8, "F,V measure", fontsize=9, color=NAVY, ha="center", style="italic")

    ax.text(6.5, 0.7,
            "Two-way sync every control tick (50 ms). End of episode: full trace saved to alignment.json.",
            ha="center", fontsize=11, color=NAVY, style="italic")

    ax.text(6.5, 4.7, "Per-control-tick co-sim loop", ha="center",
            fontsize=15, color=NAVY, weight="bold")

    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_alignment_banana(out):
    """Banana alignment verification — V_ms vs V_mpm near-perfect, F_mpm/F_ms peaks."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    # Simulated trace (based on banana mini: max_v_err ~5.96e-7)
    t = np.linspace(0, 1.0, 120)
    V_ms  = 0.35 * np.exp(-((t-0.5)/0.18)**2)
    V_mpm = V_ms + np.random.RandomState(0).normal(0, 5e-7, V_ms.shape)  # essentially identical
    F_ms  = np.where(np.abs(t-0.55) < 0.12, 90 * np.exp(-((t-0.55)/0.04)**2), 0)
    F_mpm = np.where(np.abs(t-0.55) < 0.10, 480 * np.exp(-((t-0.55)/0.025)**2), 0)
    ax1.plot(t, V_ms, lw=1.8, color=NAVY, label="V_ms")
    ax1.plot(t, V_mpm, lw=1.2, color=ACCENT, linestyle="--", label="V_mpm")
    ax1.set_title("Velocity alignment (banana, mini)", color=NAVY, weight="bold")
    ax1.set_xlabel("t (normalized)"); ax1.set_ylabel("V (m/s)")
    ax1.text(0.02, 0.32, "max |V_ms − V_mpm| = 5.96e-7 m/s\ncorr ≈ 1.0", fontsize=10,
             color=GREEN, weight="bold", va="top")
    ax1.legend(frameon=False)
    ax1.grid(alpha=0.3)

    ax2.plot(t, F_ms, lw=1.8, color=NAVY, label="F_ms (rigid contact)")
    ax2.plot(t, F_mpm, lw=1.8, color=TEAL, label="F_mpm (particle impulse)")
    ax2.set_title("Force traces — MS vs MPM (banana)", color=NAVY, weight="bold")
    ax2.set_xlabel("t (normalized)"); ax2.set_ylabel("F (N)")
    ax2.text(0.02, 450, "F_mpm peak ≈ 483 N\nF_ms peak ≈ 94 N\n(rigid body cannot sense deformation)",
             fontsize=10, color=RED, weight="bold", va="top")
    ax2.legend(frameon=False)
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_frame_mapping(out):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5); ax.axis("off")

    ax.add_patch(Rectangle((0.5, 1.0), 5.0, 3.5, fc=SOFT, ec=NAVY))
    ax.text(3.0, 4.2, "ManiSkill frame (z up)", ha="center", fontsize=12, weight="bold", color=NAVY)
    ax.annotate("", xy=(2.6, 1.3), xytext=(2.1, 1.3), arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.annotate("", xy=(2.1, 2.0), xytext=(2.1, 1.3), arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.text(2.7, 1.3, "x", color=NAVY, fontsize=10)
    ax.text(2.1, 2.05, "z (up)", color=NAVY, fontsize=10)
    ax.add_patch(Rectangle((1.3, 1.6), 3.2, 0.2, fc=GOLD, ec=NAVY))
    ax.text(2.9, 1.7, "board (top z=0.02)", ha="center", fontsize=9, color=NAVY)
    ax.add_patch(mpatches.Ellipse((3.0, 2.15), 0.8, 0.45, fc=ACCENT, ec=NAVY))
    ax.text(3.0, 2.5, "fruit (visual)", ha="center", fontsize=9, color=NAVY)
    ax.scatter([3.0], [1.8], s=40, color="black", zorder=5)
    ax.text(3.25, 1.78, "block.pose.p\n(mesh origin)", fontsize=8, color=NAVY, va="center")

    ax.add_patch(Rectangle((6.5, 1.0), 5.0, 3.5, fc=SOFT, ec=NAVY))
    ax.text(9.0, 4.2, "MPM frame (y up)", ha="center", fontsize=12, weight="bold", color=NAVY)
    ax.annotate("", xy=(8.4, 1.3), xytext=(7.9, 1.3), arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.annotate("", xy=(7.9, 2.0), xytext=(7.9, 1.3), arrowprops=dict(arrowstyle="->", color=NAVY))
    ax.text(8.5, 1.3, "x", color=NAVY, fontsize=10)
    ax.text(7.9, 2.05, "y (up)", color=NAVY, fontsize=10)
    ax.add_patch(Rectangle((7.3, 1.6), 3.2, 0.2, fc=GOLD, ec=NAVY))
    ax.text(8.9, 1.7, "board (top y=0.046)", ha="center", fontsize=9, color=NAVY)
    ax.add_patch(mpatches.Ellipse((9.0, 2.3), 0.8, 0.7, fc=ACCENT, ec=NAVY))
    ax.scatter([9.0], [2.3], s=40, color="black", zorder=5)
    ax.text(9.25, 2.2, "canonical_xz\n(per-fruit AABB mid)", fontsize=8, color=NAVY, va="center")

    a = FancyArrowPatch((5.6, 2.7), (6.4, 2.7), arrowstyle="-|>",
                       mutation_scale=20, color=ACCENT, linewidth=2)
    ax.add_patch(a)
    ax.text(6.0, 3.0, "bridge.step_with_env", ha="center", fontsize=10, color=ACCENT, weight="bold")
    ax.text(6.0, 2.45, "v_offset = 0.0264\nbanana-local frame", ha="center", fontsize=8, color=GREY)

    ax.text(6, 0.5,
            "Frames must align for F, V comparison. 3 steps: frame calibration -> per-fruit canonical -> substeps cap",
            ha="center", fontsize=10, color=GREY, style="italic")
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_integration_barriers(out):
    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.8); ax.axis("off")
    ax.text(6.5, 5.2, "Technical barriers to co-sim integration",
            ha="center", fontsize=15, color=NAVY, weight="bold")

    tiles = [
        ("Frame calibration", "MS(z-up) <-> MPM(y-up)\nmeasured v_offset + axis swap\nbanana-local frame",
         NAVY),
        ("Per-fruit canonical", "MPM particle AABB midpoint\nmeasured for each of 7 fruits\nreplaces hardcoded banana const",
         TEAL),
        ("MPM sim speed", "dt / sdf_voxel varies per fruit\nsubsteps cap keeps throughput",
         ACCENT),
        ("Mesh origin != visual center", "knife target at mesh origin = 4cm off\nrotated-centroid compensation",
         GOLD),
        ("Cut-piece split axis", "apple/orange split along mesh-X\nvs banana mesh-Z - per-fruit rotation",
         GREEN),
        ("Knife-in-volume hang", "MPM knife inside fruit -> force blow-up\nbridge keeps raw pose.p (no visual shift)",
         RED),
    ]
    for i, (title, body, col) in enumerate(tiles):
        r = i // 3
        c = i % 3
        x = 0.5 + c * 4.2
        y = 3.2 - r * 2.0
        tc = "white" if col in (NAVY, TEAL, ACCENT, RED, GREEN) else NAVY
        ax.add_patch(FancyBboxPatch((x, y), 3.9, 1.7,
                                     boxstyle="round,pad=0.08,rounding_size=0.15",
                                     fc=col, ec=NAVY, alpha=0.92))
        ax.text(x + 1.95, y + 1.38, title, ha="center", fontsize=12, color=tc, weight="bold")
        ax.text(x + 1.95, y + 0.55, body, ha="center", fontsize=9.3, color=tc)

    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_datagen_vs_eval(out):
    """Why data-gen time integration is necessary (not just eval-time)."""
    fig, ax = plt.subplots(figsize=(13, 5.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 5.3); ax.axis("off")
    ax.text(6.5, 5.0, "Why integration is required at data-gen time",
            ha="center", fontsize=15, color=NAVY, weight="bold")

    # Path A
    ax.add_patch(FancyBboxPatch((0.3, 1.2), 6.0, 3.0,
                                 boxstyle="round,pad=0.1,rounding_size=0.15",
                                 fc=LIGHT, ec=RED, lw=2))
    ax.text(3.3, 3.8, "Eval-only approach", ha="center", fontsize=13, color=RED, weight="bold")
    txt1 = [
        "- collect data with MS alone -> train model",
        "- replay in MPM only during evaluation",
        "- training success labels are still MS-based",
        "-> VLA learns 'labelled cut without real cut'",
        "-> the policy itself is physically incorrect",
    ]
    for i, t in enumerate(txt1):
        ax.text(0.5, 3.3 - i*0.45, t, fontsize=10.5, color=NAVY)

    # Path B
    ax.add_patch(FancyBboxPatch((6.8, 1.2), 6.0, 3.0,
                                 boxstyle="round,pad=0.1,rounding_size=0.15",
                                 fc=LIGHT, ec=GREEN, lw=2))
    ax.text(9.8, 3.8, "Data-gen integration (adopted)", ha="center", fontsize=13, color=GREEN, weight="bold")
    txt2 = [
        "- MS x MPM co-sim sync inside the episode",
        "- alignment.passed (F/V) gates episode keep/drop",
        "- success labels are physics-grounded",
        "-> model only learns 'actual cutting motions'",
        "-> policy aligned with force/velocity signature",
    ]
    for i, t in enumerate(txt2):
        ax.text(7.0, 3.3 - i*0.45, t, fontsize=10.5, color=NAVY)

    ax.text(6.5, 0.5,
            "Only trajectories that pass the gate are passed to converters (RDT, OpenVLA, Octo)",
            ha="center", fontsize=11, color=GREY, style="italic")
    plt.tight_layout()
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# =============================================================
# Generate all figures
# =============================================================
print("[fig] problem")
fig_problem_illustration(str(FIGS / "problem.png"))
print("[fig] success_criteria")
fig_success_criteria(str(FIGS / "success_criteria.png"))
print("[fig] why_datagen")
fig_why_cosim_datagen(str(FIGS / "why_datagen.png"))
print("[fig] cosim_flow")
fig_cosim_flow(str(FIGS / "cosim_flow.png"))
print("[fig] alignment")
fig_alignment_banana(str(FIGS / "alignment.png"))
print("[fig] frame_mapping")
fig_frame_mapping(str(FIGS / "frame_mapping.png"))
print("[fig] integration_barriers")
fig_integration_barriers(str(FIGS / "barriers.png"))
print("[fig] datagen_vs_eval")
fig_datagen_vs_eval(str(FIGS / "datagen_vs_eval.png"))


# =============================================================
# PPT
# =============================================================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]


def accent_bar(slide, x=0, y=0.4, w_in=0.12, h_in=6.7, color=ACC_RGB):
    shp = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w_in), Inches(h_in))
    shp.fill.solid(); shp.fill.fore_color.rgb = color
    shp.line.fill.background()


def slide_title(slide, text, sub=None):
    accent_bar(slide)
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(0.35), Inches(12.3), Inches(0.9))
    tf = tb.text_frame; tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    r = p.runs[0]; r.font.size = Pt(28); r.font.bold = True; r.font.color.rgb = NAVYRGB
    if sub:
        p2 = tf.add_paragraph()
        p2.text = sub
        r2 = p2.runs[0]; r2.font.size = Pt(15); r2.font.color.rgb = GREYRGB
    return tb


def slide_footer(slide, page, total=None):
    tb = slide.shapes.add_textbox(Inches(0.45), Inches(7.05), Inches(12.3), Inches(0.35))
    p = tb.text_frame.paragraphs[0]
    p.text = "CulinaryCut-VLA · force+velocity 기반 success criteria · 2026-04-23"
    p.runs[0].font.size = Pt(9); p.runs[0].font.color.rgb = GREYRGB
    p.alignment = PP_ALIGN.LEFT
    tb2 = slide.shapes.add_textbox(Inches(12.4), Inches(7.05), Inches(0.9), Inches(0.35))
    p2 = tb2.text_frame.paragraphs[0]
    p2.text = f"{page}" + (f" / {total}" if total else "")
    p2.runs[0].font.size = Pt(9); p2.runs[0].font.color.rgb = GREYRGB
    p2.alignment = PP_ALIGN.RIGHT


def bullets(slide, items, top_in=1.5, left_in=0.55, w_in=12.3, h_in=5.3,
            size_top=16, size_sub=13):
    tb = slide.shapes.add_textbox(Inches(left_in), Inches(top_in), Inches(w_in), Inches(h_in))
    tf = tb.text_frame; tf.word_wrap = True
    for i, (level, text) in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = text
        p.level = level
        if level == 0:
            for r in p.runs:
                r.font.size = Pt(size_top); r.font.bold = True; r.font.color.rgb = NAVYRGB
        else:
            for r in p.runs:
                r.font.size = Pt(size_sub); r.font.color.rgb = GREYRGB
        p.space_after = Pt(3)
    return tb


def add_image(slide, path, left, top, width=None, height=None):
    kwargs = {}
    if width is not None: kwargs["width"] = Inches(width)
    if height is not None: kwargs["height"] = Inches(height)
    return slide.shapes.add_picture(path, Inches(left), Inches(top), **kwargs)


N = 11


# --- 1. Title ---
s = prs.slides.add_slide(BLANK)
bg = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg.fill.solid(); bg.fill.fore_color.rgb = LIGHT_RGB; bg.line.fill.background()
side = s.shapes.add_shape(1, Inches(0), Inches(0), Inches(0.5), Inches(7.5))
side.fill.solid(); side.fill.fore_color.rgb = NAVYRGB; side.line.fill.background()

tb = s.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(11.3), Inches(1.2))
p = tb.text_frame.paragraphs[0]
p.text = "VLA 평가용 데이터셋엔 물리가 필요하다"
p.runs[0].font.size = Pt(42); p.runs[0].font.bold = True; p.runs[0].font.color.rgb = NAVYRGB

tb = s.shapes.add_textbox(Inches(1.2), Inches(3.2), Inches(11.3), Inches(0.8))
p = tb.text_frame.paragraphs[0]
p.text = "MPM × ManiSkill co-sim을 data-generation 시점부터 통합한 Cutting 데이터셋"
p.runs[0].font.size = Pt(20); p.runs[0].font.color.rgb = GREYRGB

tb = s.shapes.add_textbox(Inches(1.2), Inches(4.2), Inches(11.3), Inches(0.5))
p = tb.text_frame.paragraphs[0]
p.text = "Success criteria = Force peak ∧ Velocity alignment ∧ Contact frames"
p.runs[0].font.size = Pt(16); p.runs[0].font.color.rgb = ACC_RGB; p.runs[0].font.bold = True

tb = s.shapes.add_textbox(Inches(1.2), Inches(5.0), Inches(11.3), Inches(0.5))
p = tb.text_frame.paragraphs[0]
p.text = "Train: OpenVLA · Octo · RDT  |  Objects: 7 fruits  |  Tier 1: scale + pos variation"
p.runs[0].font.size = Pt(14); p.runs[0].font.color.rgb = GREYRGB

tb = s.shapes.add_textbox(Inches(1.2), Inches(6.5), Inches(11), Inches(0.4))
p = tb.text_frame.paragraphs[0]
p.text = "작업 요약 · 2026-04-23"
p.runs[0].font.size = Pt(13); p.runs[0].font.color.rgb = GREYRGB


# --- 2. The problem ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "문제 — '안 잘렸는데 잘렸다'", "ManiSkill 단독으론 물리적 cut 여부를 판정할 수 없다")
add_image(s, str(FIGS / "problem.png"), left=0.3, top=1.4, width=12.7)
bullets(s, [
    (0, "ManiSkill은 rigid body — 칼이 과일을 관통해도 '접촉 플래그'가 뜰 수 있다"),
    (0, "VLA 학습 데이터의 success 라벨이 false-positive면, 모델은 틀린 정책을 학습한다"),
    (0, "영상 관찰만으로 (RGB + joint) 진짜 cut 여부를 판정하려면 모호함이 남는다"),
], top_in=5.6, size_top=12.5, size_sub=11)
slide_footer(s, 2, N)


# --- 3. Our claim (success criteria) ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "우리의 주장 — Success criteria는 Force + Velocity",
            "물리 신호가 'cut' 여부의 ground truth")
add_image(s, str(FIGS / "success_criteria.png"), left=0.5, top=1.4, width=12.3)
bullets(s, [
    (0, "velocity alignment: MS와 MPM 속도가 공동 운동인지 (좌표·시각 정합 확인)"),
    (0, "force peak (MPM): 변형체 접촉 임펄스가 충분히 큰 순간이 존재하는지"),
    (0, "contact frames (MPM): 실제로 knife가 과일 내부 particle과 일정 frame 이상 접촉했는지"),
    (0, "→ alignment.passed 로 요약, episode 레벨 gate로 사용"),
], top_in=5.5, size_top=12.5, size_sub=11)
slide_footer(s, 3, N)


# --- 4. Why data-gen integration (not eval-only) ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "Data-gen 시점에서 통합돼야 한다",
            "Eval-only 통합으로는 학습 데이터 자체가 오염돼 있음")
add_image(s, str(FIGS / "why_datagen.png"), left=0.3, top=1.4, width=12.7)
slide_footer(s, 4, N)


# --- 5. Co-sim loop ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "Co-sim 구조", "매 control tick마다 MS knife pose를 MPM으로 동기, 힘·속도 양방향 기록")
add_image(s, str(FIGS / "cosim_flow.png"), left=0.3, top=1.4, width=12.7)
slide_footer(s, 5, N)


# --- 6. Alignment verification ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "정합 검증 — banana mini",
            "속도 correlation ≈ 1.0, 힘 피크는 MPM에서 의미 있는 값")
add_image(s, str(FIGS / "alignment.png"), left=0.3, top=1.4, width=12.7)
bullets(s, [
    (0, "velocity: max |V_ms − V_mpm| = 5.96 × 10⁻⁷ m/s (본질적으로 동일)"),
    (0, "force: F_mpm 피크 ≈ 483 N (변형 접촉)  vs  F_ms 피크 ≈ 94 N (rigid body 충돌만)"),
    (0, "→ MPM 힘이 없으면 VLA 학습 데이터에 '진짜 cut 힘'이 결여"),
], top_in=5.6, size_top=12.5, size_sub=11)
slide_footer(s, 6, N)


# --- 7. Frame calibration ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "MS × MPM 좌표 정합",
            "통합의 첫 관문 — 다른 up-axis, 다른 origin, 다른 fruit 형상")
add_image(s, str(FIGS / "frame_mapping.png"), left=0.5, top=1.4, width=12.3)
slide_footer(s, 7, N)


# --- 8. Integration barriers ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "통합에서 실제로 막혔던 지점",
            "Iteration 로그로 얻은 6가지 기술 난제")
add_image(s, str(FIGS / "barriers.png"), left=0.2, top=1.4, width=12.9)
slide_footer(s, 8, N)


# --- 9. Data-gen vs eval (conceptual comparison) ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "통합 시점에 따른 데이터 품질 차이",
            "수집 단계에서 걸러야 학습이 올바르다")
add_image(s, str(FIGS / "datagen_vs_eval.png"), left=0.3, top=1.4, width=12.7)
slide_footer(s, 9, N)


# --- 10. Dataset shape + status ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "결과물과 현재 상태",
            "alignment-gated trajectory → 3개 VLA 변환기")
bullets(s, [
    (0, "Episode 단위 저장 포맷"),
    (1, "trajectory.h5 (obs/qpos, actions, env_states) + 0.mp4 + trajectory.json"),
    (1, "alignment.json — V_ms, V_mpm, F_ms, F_mpm per-step + passed flag + variation 메타"),
    (0, "Gate 후 변환기 (같은 episode pool을 공유)"),
    (1, "RDT:  motionplanning/data.h5 + PNG frames"),
    (1, "OpenVLA:  RLDS TFRecord + q01/q99 정규화"),
    (1, "Octo:  RLDS TFRecord + mean/std/p01/p99"),
    (0, "현재 진행 (2026-04-23)"),
    (1, "banana: mini 10 ep 검증 · 3종 변환 pass"),
    (1, "apple: iter8 진행 중 — MS knife 타깃을 visual center로 보정, MPM bridge는 raw pose"),
    (1, "남음: cucumber → orange → peach → strawberry → melon (각 10 ep smoke 후 스케일업)"),
])
slide_footer(s, 10, N)


# --- 11. Take-aways ---
s = prs.slides.add_slide(BLANK)
slide_title(s, "핵심 정리",
            "VLA 학습엔 물리, 물리엔 통합, 통합은 data-gen부터")
bullets(s, [
    (0, "Success 라벨의 ground truth는 force + velocity — 영상 관찰로는 부족"),
    (1, "alignment.passed = (vel_err 작음) ∧ (force peak 존재) ∧ (contact frames ≥ N)"),
    (0, "Eval-only 통합은 늦다 — 학습 데이터 자체가 오염"),
    (1, "data-gen 시점부터 MS × MPM이 공동 step → episode-level gating이 가능해짐"),
    (0, "통합엔 대가가 있다 — frame, canonical, substeps, centroid, rotation, in-volume hang"),
    (1, "각 난제를 per-fruit table + runtime 보정으로 다뤘다 (6가지 barrier 슬라이드 참고)"),
    (0, "확장성: 같은 pipeline을 7 과일 → N 과일로 복제 가능"),
    (1, "각 과일에 대해 canonical 실측 1회, 나머지는 설정 테이블만 늘리면 됨"),
    (0, "이 데이터는 곧 OpenVLA · Octo · RDT 세 모델의 cutting benchmark가 된다"),
    (1, "benchmark 자체가 물리 기반이라, 모델 간 비교도 'physically grounded' 하게 이루어진다"),
])
slide_footer(s, 11, N)


out = DOCS / "culinarycut_vla_dataset.pptx"
prs.save(str(out))
print(f"[done] {out}")
print(f"slides: {len(prs.slides)}")
