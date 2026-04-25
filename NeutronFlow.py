import numpy as np
import matplotlib.pyplot as plt
import random as rn
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib import cm

# -----------------------------
# Constant Parameters
# -----------------------------
pA = 0.05          # Absorption probability in reflector
pF = 0.85          # Fission probability in core
R0 = 6.5           # Core radius
sigmaS_core = 0.3  # Core scattering cross section
toll = 0.08        # Tolerance for faster visualization

sigma_vector = [0.3, 0.5, 0.7, 0.9]  # Scattering cross sections
colors = cm.plasma(np.linspace(0.1, 0.9, len(sigma_vector)))

# -----------------------------
# Animation geometry settings
# Reflector thickness evolution (from 7.0 m to 8.6 m)
# -----------------------------
R_steps = np.linspace(7.0, 8.6, 15)

# -----------------------------
# Figure and layout setup
# -----------------------------
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])

ax1 = fig.add_subplot(gs[0, 0])   # Geometry plot
ax2 = fig.add_subplot(gs[0, 1])   # Statistics plot
ax_table = fig.add_subplot(gs[1, :])  # Table area
ax_table.axis('off')

# -----------------------------
# Animation update function
# -----------------------------
def update(frame):
    
    ax1.clear()
    ax2.clear()
    ax_table.clear()
    ax_table.axis('off')
    
    R_current = R_steps[frame]
    thickness = R_current - R0

    # -------------------------
    # 1. Geometry visualization
    # -------------------------
    theta = np.linspace(0, 2*np.pi, 100)

    # Reflector region
    ax1.fill_between(theta, R0, R_current, color='gray', alpha=0.2, label="Reflector")
    ax1.plot(R_current*np.cos(theta), R_current*np.sin(theta), 'k--', lw=1)

    # Core region
    ax1.fill(R0*np.cos(theta), R0*np.sin(theta), color='navy', alpha=0.15, label="Core")
    ax1.plot(R0*np.cos(theta), R0*np.sin(theta), 'navy', lw=2)

    table_rows = []

    # -------------------------
    # 2. Monte Carlo simulation
    # -------------------------
    for h, sigmaS in enumerate(sigma_vector):
        
        i = 0
        escaped = 0
        fissions = 0
        absorptions = 0
        
        sample_avg = []
        csi = []

        # Simulate fixed number of neutrons per frame
        while i < 120:
            i += 1

            # Initial neutron source position
            x0, y0 = R0 * np.cos(np.pi/4), R0 * np.sin(np.pi/4)

            life = 1

            # Random direction sampling
            mu = -1 + 2*rn.random()
            phi = 2*np.pi*rn.random()

            sigma = sigmaS

            # Neutron transport loop
            while life == 1:

                free_path = -1/sigma * np.log(rn.random() + 1e-9)

                x = x0 + np.sqrt(1-mu**2)*np.cos(phi)*free_path
                y = y0 + np.sqrt(1-mu**2)*np.sin(phi)*free_path

                dist = np.sqrt(x**2 + y**2)

                # -------------------------
                # Escape condition
                # -------------------------
                if dist >= R_current:
                    life = 0
                    escaped += 1
                    csi.append(1)
                    ax1.plot(x, y, 'x', color=colors[h], markersize=3, alpha=0.5)

                # -------------------------
                # Reflector region
                # -------------------------
                elif R0 <= dist < R_current:
                    if rn.random() < pA:
                        life = 0
                        absorptions += 1
                        csi.append(0)
                    else:
                        x0, y0 = x, y
                        mu, phi = -1 + 2*rn.random(), 2*np.pi*rn.random()

                # -------------------------
                # Core region
                # -------------------------
                else:
                    if rn.random() < pF:
                        life = 0
                        fissions += 1
                        csi.append(0)
                        ax1.plot(x, y, '.', color=colors[h], markersize=4)
                    else:
                        x0, y0 = x, y
                        mu, phi = -1 + 2*rn.random(), 2*np.pi*rn.random()
                        sigma = sigmaS_core

            sample_avg.append(sum(csi)/i)

        # -------------------------
        # Plot results
        # -------------------------
        ax2.plot(sample_avg, color=colors[h], lw=1.5, label=f"$\\Sigma_s$: {sigmaS}")

        total = escaped + fissions + absorptions

        table_rows.append([
            f"{sigmaS}",
            f"% {(escaped/total)*100:.1f}",
            f"% {(fissions/total)*100:.1f}",
            f"% {(absorptions/total)*100:.1f}"
        ])

    # -------------------------
    # Styling
    # -------------------------
    ax1.set_title(f"Reflector Thickness: {thickness:.2f} m", fontsize=12, fontweight='bold')
    ax1.axis('equal')
    ax1.set_xlim(-10, 10)
    ax1.set_ylim(-10, 10)

    ax2.set_title("Leakage Probability (Sample Average)", fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.legend(loc='upper right', fontsize='8', ncol=2)
    ax2.grid(alpha=0.2)

    # -------------------------
    # Table (bottom panel)
    # -------------------------
    col_labels = [
        'Scattering Cross Section ($\\Sigma_s$)',
        'Leakage',
        'Extra Fission',
        'Absorption'
    ]

    the_table = ax_table.table(
        cellText=table_rows,
        colLabels=col_labels,
        loc='center',
        cellLoc='center'
    )

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(10)
    the_table.scale(1, 1.8)

# -----------------------------
# Run animation and save GIF
# -----------------------------
plt.tight_layout()

print("Generating GIF...")

ani = FuncAnimation(fig, update, frames=len(R_steps), interval=400)

ani.save("neutron_physics.gif", writer=PillowWriter(fps=3))

plt.close()

print("Done! Check 'neutron_physics.gif'.")