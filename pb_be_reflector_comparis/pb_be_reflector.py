"""
Author: Emil Mammadzada

GitHub:
https://github.com/emilmammadzada99

Contact:
emilmemmedzade23@outlook.com
egoemil32@gmail.com
"""
from ace_xs import AceLibrary, MaterialXS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random as rn
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from matplotlib import cm

# =========================================================
# DARK THEME
# =========================================================
plt.style.use("dark_background")

# =========================================================
# AUTO ZAID FINDER
# =========================================================
def find_ace_zaid(filepath):
    with open(filepath, "r", errors="replace") as f:
        return f.readline().strip().split()[0]


# =========================================================
# LOAD ACE LIBRARY
# =========================================================
print("Loading ACE cross section libraries...")

lib = AceLibrary()

# =========================================================
# FUEL → U235
# =========================================================
fuel_file = "endfb7/acedata/92235ENDF7.ace"
fuel_zaid = find_ace_zaid(fuel_file)

print(f"Reading Fuel ACE: {fuel_file} -> {fuel_zaid}")

lib.read_ace_file(filepath=fuel_file, target_zaid=fuel_zaid)

fuel_mat = MaterialXS(lib, name="U235 Fuel")
fuel_mat.add_isotope(zaid=fuel_zaid, atom_density=0.02)
fuel_mat.build()

print("Fuel loaded.\n")


# =========================================================
# REFLECTOR → Be
# =========================================================
be_file = "endfb7/acedata/4009ENDF7.ace"
be_zaid = find_ace_zaid(be_file)

print(f"Reading Be Reflector ACE: {be_file} -> {be_zaid}")

lib.read_ace_file(filepath=be_file, target_zaid=be_zaid)

be_mat = MaterialXS(lib, name="Be Reflector")
be_mat.add_isotope(zaid=be_zaid, atom_density=0.12)
be_mat.build()

print("Be reflector loaded.\n")


# =========================================================
# REFLECTOR → Pb
# =========================================================
pb_file = "endfb7/acedata/82208ENDF7.ace"
pb_zaid = find_ace_zaid(pb_file)

print(f"Reading Pb Reflector ACE: {pb_file} -> {pb_zaid}")

lib.read_ace_file(filepath=pb_file, target_zaid=pb_zaid)

pb_mat = MaterialXS(lib, name="Pb Reflector")
pb_mat.add_isotope(zaid=pb_zaid, atom_density=0.033)
pb_mat.build()

print("Pb reflector loaded.\n")


# =========================================================
# GEOMETRY
# =========================================================
R0      = 6.5
R_steps = np.linspace(7.0, 8.6, 60)
E_ref   = 0.0253  # eV

colors = ["#00ffff", "#ff6600", "#39ff14", "#ff00ff"]


# =========================================================
# CORE XS → U235
# =========================================================
sigma_t_core = fuel_mat.get_total_xs(E_ref)
sigma_f_core = fuel_mat.get_fission_xs(E_ref)
sigma_c_core = fuel_mat.get_capture_xs(E_ref)

sigma_s_core = max(sigma_t_core - sigma_f_core - sigma_c_core, 1e-6)
pF = sigma_f_core / sigma_t_core


# =========================================================
# Be XS
# =========================================================
sigma_t_be = be_mat.get_total_xs(E_ref)
sigma_c_be = be_mat.get_capture_xs(E_ref)
sigma_s_be = max(sigma_t_be - sigma_c_be, 1e-6)
pA_be      = sigma_c_be / sigma_t_be


# =========================================================
# Pb XS
# =========================================================
sigma_t_pb = pb_mat.get_total_xs(E_ref)
sigma_c_pb = pb_mat.get_capture_xs(E_ref)
sigma_s_pb = max(sigma_t_pb - sigma_c_pb, 1e-6)
pA_pb      = sigma_c_pb / sigma_t_pb


print("======================================")
print("PHYSICS PARAMETERS")
print("======================================")
print(f"U235 total     = {sigma_t_core:.5f}")
print(f"U235 fission   = {sigma_f_core:.5f}")
print(f"U235 capture   = {sigma_c_core:.5f}")
print(f"U235 scatter   = {sigma_s_core:.5f}")
print(f"pF             = {pF:.5f}")
print()
print(f"Be total       = {sigma_t_be:.5f}")
print(f"Be capture     = {sigma_c_be:.5f}")
print(f"Be scatter     = {sigma_s_be:.5f}")
print(f"pA(Be)         = {pA_be:.5f}")
print()
print(f"Pb total       = {sigma_t_pb:.5f}")
print(f"Pb capture     = {sigma_c_pb:.5f}")
print(f"Pb scatter     = {sigma_s_pb:.5f}")
print(f"pA(Pb)         = {pA_pb:.5f}")
print("======================================\n")


# =========================================================
# SIGMA VECTORS
# =========================================================
sigma_vector_be = [
    sigma_s_be * 0.7,
    sigma_s_be * 0.9,
    sigma_s_be * 1.1,
    sigma_s_be * 1.3,
]

sigma_vector_pb = [
    sigma_s_pb * 0.7,
    sigma_s_pb * 0.9,
    sigma_s_pb * 1.1,
    sigma_s_pb * 1.3,
]


# =========================================================
# HEATMAP COLORMAPS
# =========================================================
cmap_be = mcolors.LinearSegmentedColormap.from_list(
    "be_heat",
    ["#050510", "#001840", "#0044cc", "#00ccff", "#ffffff"], N=256
)
cmap_pb = mcolors.LinearSegmentedColormap.from_list(
    "pb_heat",
    ["#050510", "#1a0500", "#bb2200", "#ff7700", "#ffffff"], N=256
)


# =========================================================
# GAUSSIAN BLUR (FFT, no scipy)
# =========================================================
def gauss_blur(arr, sigma_px=5):
    rows, cols = arr.shape
    fy = np.fft.fftfreq(rows)
    fx = np.fft.fftfreq(cols)
    FX, FY = np.meshgrid(fx, fy)
    s = sigma_px / max(rows, cols)
    kernel = np.exp(-2 * np.pi**2 * (FX**2 + FY**2) / (s**-2))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * kernel)).clip(0)


# =========================================================
# FIGURE LAYOUT — 1080×1080  ( 2×2 grid)
# =========================================================
DPI = 100
fig = plt.figure(figsize=(10.8, 10.8), dpi=DPI, facecolor="#050510")

gs = fig.add_gridspec(
    2, 2,
    height_ratios=[1, 1],
    hspace=0.30,
    wspace=0.28,
    left=0.06, right=0.97,
    top=0.93, bottom=0.04
)

# Satır 0 → Be
ax_be_geom = fig.add_subplot(gs[0, 0])
ax_be_plot = fig.add_subplot(gs[0, 1])

# Satır 1 → Pb
ax_pb_geom = fig.add_subplot(gs[1, 0])
ax_pb_plot = fig.add_subplot(gs[1, 1])

for ax in [ax_be_geom, ax_be_plot, ax_pb_geom, ax_pb_plot]:
    ax.set_facecolor("#08081a")
    for sp in ax.spines.values():
        sp.set_edgecolor("#1a1a3e")


# =========================================================
# MONTE CARLO FUNCTION 
# =========================================================
def run_case(
    ax_geom,
    ax_plot,
    sigma_vector,
    sigma_abs_prob,
    sigma_label,
    R_current,
    heat_cmap,
    core_color,
    refl_color
):
    GRID = 150
    heat = np.zeros((GRID, GRID))

    theta = np.linspace(0, 2*np.pi, 300)

    # Core gradient layers
    for a, scale in zip([0.08, 0.03, 0.01], [1.0, 0.95, 0.90]):
        ax_geom.fill(
            R0*scale*np.cos(theta),
            R0*scale*np.sin(theta),
            color=core_color, alpha=a
        )

    ax_geom.plot(
        R0*np.cos(theta), R0*np.sin(theta),
        color=core_color, lw=2
    )

    # Reflector gradient layers
    for a, scale in zip([0.08, 0.03, 0.01], [1.0, 0.95, 0.90]):
        ax_geom.fill(
            R_current*scale*np.cos(theta),
            R_current*scale*np.sin(theta),
            color=refl_color, alpha=a
        )

    ax_geom.plot(
        R_current*np.cos(theta), R_current*np.sin(theta),
        "--", color=refl_color, alpha=0.7, lw=1
    )

    table_rows = []

    for h, sigmaS in enumerate(sigma_vector):

        escaped     = 0
        fissions    = 0
        absorptions = 0
        sample_avg  = []
        csi         = []

        for i in range(1, 7000):

            start_angle = 2 * np.pi * i / 120
            x0 = R0 * np.cos(start_angle)
            y0 = R0 * np.sin(start_angle)

            mu    = -1 + 2 * rn.random()
            phi   = 2 * np.pi * rn.random()
            sigma = sigma_s_core
            alive = True

            while alive:

                free_path = -np.log(rn.random() + 1e-12) / max(sigma, 1e-8)

                x = x0 + np.sqrt(1 - mu**2) * np.cos(phi) * free_path
                y = y0 + np.sqrt(1 - mu**2) * np.sin(phi) * free_path

                ix = int((x + 10) / 20 * (GRID - 1))
                iy = int((y + 10) / 20 * (GRID - 1))
                if 0 <= ix < GRID and 0 <= iy < GRID:
                    heat[iy, ix] += 1

                dist = np.sqrt(x*x + y*y)

                if dist >= R_current:
                    escaped += 1
                    csi.append(1)
                    alive = False

                elif R0 <= dist < R_current:
                    if rn.random() < sigma_abs_prob:
                        absorptions += 1
                        csi.append(0)
                        alive = False
                    else:
                        x0, y0 = x, y
                        mu  = -1 + 2 * rn.random()
                        phi = 2 * np.pi * rn.random()
                        sigma = sigmaS

                else:
                    if rn.random() < pF:
                        fissions += 1
                        csi.append(0)
                        alive = False
                    else:
                        x0, y0 = x, y
                        mu  = -1 + 2 * rn.random()
                        phi = 2 * np.pi * rn.random()
                        sigma = sigma_s_core

            sample_avg.append(sum(csi) / len(csi))

        ax_plot.plot(
            sample_avg,
            lw=1.5,
            color=colors[h],
            label=f"Σs={sigmaS:.2f}"
        )

        total = escaped + fissions + absorptions

        table_rows.append([
            f"{sigmaS:.2f}",
            f"{escaped} / {total}",
            f"{fissions} / {total}",
            f"{absorptions} / {total}"
        ])

    # ── Heatmap render ──────────────────────────────────
    h_sm = gauss_blur(heat, sigma_px=5)
    vmax = h_sm.max() if h_sm.max() > 0 else 1
    ax_geom.imshow(
        h_sm,
        origin="lower",
        extent=[-10, 10, -10, 10],
        cmap=heat_cmap,
        vmin=0, vmax=vmax,
        aspect="auto",
        interpolation="bilinear",
        alpha=0.85,
        zorder=2
    )

    ax_geom.plot(R0*np.cos(theta), R0*np.sin(theta),
                 color=core_color, lw=2.0, alpha=0.95, zorder=5)
    ax_geom.plot(R_current*np.cos(theta), R_current*np.sin(theta),
                 "--", color=refl_color, lw=1.4, alpha=0.85, zorder=5)

    ax_plot.set_title(f"{sigma_label} Leakage Probability",
                      fontsize=10, color="#ccccff")
    ax_plot.set_ylim(0, 1.1)
    ax_plot.grid(alpha=0.1, color="#333366")
    ax_plot.tick_params(colors="#888899", labelsize=8)
    ax_plot.legend(fontsize=9, ncol=2,
                   facecolor="#0a0a2e", edgecolor="#333366", labelcolor="white")

    ax_geom.set_xlim(-10, 10)
    ax_geom.set_ylim(-10, 10)
    ax_geom.tick_params(colors="#888899", labelsize=7)

    print(f"\n===== {sigma_label} Summary | R = {R_current:.2f} =====")
    print(f"{'Reflector Σs':<15} {'Leakage':<15} {'Fission':<15} {'Absorption':<15}")
    for row in table_rows:
        print(f"{row[0]:<15} {row[1]:<15} {row[2]:<15} {row[3]:<15}")
    print("============================================\n")

    return table_rows


# =========================================================
# UPDATE FUNCTION
# =========================================================
def update(frame):

    ax_be_geom.clear()
    ax_be_plot.clear()
    ax_pb_geom.clear()
    ax_pb_plot.clear()

    for ax in [ax_be_geom, ax_be_plot, ax_pb_geom, ax_pb_plot]:
        ax.set_facecolor("#08081a")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1a1a3e")

    fig.patch.set_facecolor("#050510")

    R_current = R_steps[frame]
    thickness = R_current - R0

    fig.suptitle(
        "Core  ·  Be / Pb Reflector  ·  Monte Carlo Neutron Transport",
        fontsize=13, fontweight="bold", color="#e0e0ff", y=0.985
    )

    # =========================
    # Be case
    # =========================
    run_case(
        ax_be_geom, ax_be_plot,
        sigma_vector_be, pA_be,
        "Be", R_current,
        heat_cmap=cmap_be,
        core_color="#00cfff",
        refl_color="#0066ff"
    )

    ax_be_geom.set_title(
        f"Be Reflector  |  Thickness = {thickness:.2f} m",
        fontsize=10, color="#00cfff"
    )

    # =========================
    # Pb case
    # =========================
    run_case(
        ax_pb_geom, ax_pb_plot,
        sigma_vector_pb, pA_pb,
        "Pb", R_current,
        heat_cmap=cmap_pb,
        core_color="#ff8800",
        refl_color="#883300"
    )

    ax_pb_geom.set_title(
        f"Pb Reflector  |  Thickness = {thickness:.2f} m",
        fontsize=10, color="#ff9955"
    )


# =========================================================
# SAVE — MP4 1080×1080
# =========================================================
plt.tight_layout(rect=[0, 0, 1, 0.97])

print("Generating 1080×1080 MP4...")

ani = FuncAnimation(fig, update, frames=len(R_steps), interval=400)

try:
    writer = FFMpegWriter(fps=3, bitrate=3000)
    ani.save("u235_be_pb_reflector.mp4", writer=writer, dpi=DPI)
    print("Done -> u235_be_pb_reflector.mp4")
except Exception as e:
    print(f"FFmpeg unavailable ({e}), saving GIF...")
    ani.save("u235_be_pb_reflector.gif", writer=PillowWriter(fps=3), dpi=DPI)
    print("Done -> u235_be_pb_reflector.gif")

plt.close()
