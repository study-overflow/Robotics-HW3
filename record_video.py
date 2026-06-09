"""
Record MuJoCo simulation video for 3-DOF planar manipulator (Case 1)
Shows: arm motion + true force (red) vs estimated force (blue dashed)
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import mujoco
import os

np.random.seed(42)

L1, L2, L3 = 1.0, 1.0, 1.0
model_path = os.path.join(os.path.dirname(__file__), "manipulator.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

dt = model.opt.timestep
T_video = 5.0
video_steps = int(T_video / dt)
skip = max(1, int(video_steps // (30 * T_video)))

def compute_jacobian(q):
    q1, q2, q3 = q[0], q[1], q[2]
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)
    return np.array([[-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123],
                     [ L1*c1 + L2*c12 + L3*c123,  L2*c12 + L3*c123,  L3*c123]], dtype=np.float64)

# Precompute
q_log, F_true_log, F_est_log = [], [], []

for step in range(video_steps):
    t = step * dt
    q_des = np.array([np.pi/6 + 0.1*np.sin(0.5*t),
                      np.pi/4 + 0.1*np.cos(0.6*t),
                      np.pi/6 + 0.08*np.sin(0.4*t)])
    dq_des = np.array([0.05*np.cos(0.5*t), -0.06*np.sin(0.6*t), 0.032*np.cos(0.4*t)])

    F_ext = np.array([5.0*np.sin(2.0*t), 3.0*np.cos(1.5*t)])

    Kp, Kd = 60.0, 8.0
    tau_control = Kp*(q_des - data.qpos[:]) + Kd*(dq_des - data.qvel[:])

    J = compute_jacobian(data.qpos)
    tau_ext = J.T @ F_ext
    data.ctrl[:] = tau_control + tau_ext

    mujoco.mj_step(model, data)

    if step % skip == 0:
        tau_ext_est = data.ctrl - tau_control
        J = compute_jacobian(data.qpos)
        try:
            F_est = np.linalg.lstsq(J.T, tau_ext_est, rcond=None)[0]
        except np.linalg.LinAlgError:
            F_est = np.zeros(2)

        q_log.append(np.copy(data.qpos))
        F_true_log.append(F_ext.copy())
        F_est_log.append(F_est.copy())

q_log = np.array(q_log)
F_true_log = np.array(F_true_log)
F_est_log = np.array(F_est_log)

print(f"Recording video with {len(q_log)} frames...")

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim([-3.5, 3.5])
ax.set_ylim([-3.5, 3.5])
ax.set_aspect('equal')
ax.grid(True)
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('Case 1: True Force (red) vs Estimated Force (blue dashed)', fontsize=14)

# Legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='red', lw=2, label='True Force'),
                   Line2D([0], [0], color='blue', lw=2, linestyle='--', label='Estimated Force')]
ax.legend(handles=legend_elements, loc='upper right')

text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

(line_arm,) = ax.plot([], [], 'o-', linewidth=3, markersize=8, color='blue')

arrow_true = None
arrow_est = None

def init():
    line_arm.set_data([], [])
    text.set_text('')
    return line_arm, text

def update(frame):
    global arrow_true, arrow_est
    if arrow_true is not None:
        arrow_true.remove()
    if arrow_est is not None:
        arrow_est.remove()

    q = q_log[frame]
    F_true = F_true_log[frame]
    F_est = F_est_log[frame]

    j1 = np.array([L1*np.cos(q[0]), L1*np.sin(q[0])])
    j2 = j1 + np.array([L2*np.cos(q[0]+q[1]), L2*np.sin(q[0]+q[1])])
    ee = j2 + np.array([L3*np.cos(q[0]+q[1]+q[2]), L3*np.sin(q[0]+q[1]+q[2])])

    x_coords = [0, j1[0], j2[0], ee[0]]
    y_coords = [0, j1[1], j2[1], ee[1]]
    line_arm.set_data(x_coords, y_coords)

    scale = 0.3
    # True force (red)
    arrow_true = FancyArrowPatch((ee[0], ee[1]),
                                  (ee[0] + scale*F_true[0], ee[1] + scale*F_true[1]),
                                  arrowstyle='->', mutation_scale=25, linewidth=2.5,
                                  color='red')
    ax.add_patch(arrow_true)

    # Estimated force (blue dashed)
    arrow_est = FancyArrowPatch((ee[0], ee[1]),
                                 (ee[0] + scale*F_est[0], ee[1] + scale*F_est[1]),
                                 arrowstyle='->', mutation_scale=25, linewidth=2,
                                 color='blue', linestyle='--')
    ax.add_patch(arrow_est)

    text.set_text(f'Frame: {frame}\nTrue:  Fx={F_true[0]:.2f}, Fy={F_true[1]:.2f}\nEst:   Fx={F_est[0]:.2f}, Fy={F_est[1]:.2f}')

    return line_arm, text

ani = animation.FuncAnimation(fig, update, frames=len(q_log), init_func=init,
                              interval=1000/30, blit=False)

output_video = "/data2/zzhangg/Robotics-HW-3/simulation_video.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=5000)
ani.save(output_video, writer=writer)
plt.close()
print(f"Video saved to: {output_video}")
