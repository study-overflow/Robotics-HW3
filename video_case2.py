"""
Case 2 Visualization Video: True vs Estimated Force and Location
Shows the 3-DOF arm with force applied at unknown position on link 3
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch, Circle
import mujoco
import os

np.random.seed(42)

L1, L2, L3 = 1.0, 1.0, 1.0
model_path = os.path.join(os.path.dirname(__file__), "manipulator.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

dt = model.opt.timestep
T_sim = 6.0
n_steps = int(T_sim / dt)


def compute_jacobian_at_point(q, r):
    q1, q2, q3 = q[0], q[1], q[2]
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)
    return np.array([[-L1*s1 - L2*s12 - r*s123, -L2*s12 - r*s123, -r*s123],
                     [ L1*c1 + L2*c12 + r*c123,  L2*c12 + r*c123,  r*c123]], dtype=np.float64)


def compute_jacobian_ee(q):
    q1, q2, q3 = q[0], q[1], q[2]
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)
    return np.array([[-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123],
                     [ L1*c1 + L2*c12 + L3*c123,  L2*c12 + L3*c123,  L3*c123]], dtype=np.float64)


# Precompute simulation data
print("Precomputing simulation data for Case 2 video...")
data.qpos[:] = [0.0, 0.0, 0.0]
data.qvel[:] = [0.0, 0.0, 0.0]
mujoco.mj_forward(model, data)

q_log, F_true_log, F_est_log, r_est_log = [], [], [], []
r_contact = 0.6

for step in range(n_steps):
    t = step * dt
    q_des = np.array([0.3*np.sin(0.5*t), 0.2*np.cos(0.8*t), 0.1*np.sin(0.3*t)])
    dq_des = np.array([0.15*np.cos(0.5*t), -0.16*np.sin(0.8*t), 0.03*np.cos(0.3*t)])

    F_ext = np.array([3.0*np.sin(1.0*t), 2.0*np.cos(0.7*t)])

    Kp, Kd = 80.0, 10.0
    tau_control = Kp*(q_des - data.qpos[:]) + Kd*(dq_des - data.qvel[:])

    J_c = compute_jacobian_at_point(data.qpos, r_contact)
    tau_ext = J_c.T @ F_ext
    data.ctrl[:] = tau_control + tau_ext

    mujoco.mj_step(model, data)

    # Estimate
    tau_ext_est = data.ctrl - tau_control
    tau1, tau2, tau3 = tau_ext_est[0], tau_ext_est[1], tau_ext_est[2]

    q = data.qpos
    q1, q2, q3 = q[0], q[1], q[2]
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)

    A = np.array([[-L1*s1,  L1*c1 ],
                  [-L2*s12, L2*c12]], dtype=np.float64)
    b = np.array([tau1 - tau2, tau2 - tau3], dtype=np.float64)

    try:
        if abs(np.linalg.det(A)) > 1e-6:
            sol = np.linalg.solve(A, b)
            Fx_est, Fy_est = sol[0], sol[1]
            denom = -s123*Fx_est + c123*Fy_est
            if abs(denom) > 1e-6:
                r_est = tau3 / denom
            else:
                Fx_est, Fy_est, r_est = 0.0, 0.0, 0.0
        else:
            Fx_est, Fy_est, r_est = 0.0, 0.0, 0.0
    except:
        Fx_est, Fy_est, r_est = 0.0, 0.0, 0.0

    q_log.append(np.copy(data.qpos))
    F_true_log.append(F_ext.copy())
    F_est_log.append([Fx_est, Fy_est])
    r_est_log.append(r_est)

q_log = np.array(q_log)
F_true_log = np.array(F_true_log)
F_est_log = np.array(F_est_log)
r_est_log = np.array(r_est_log)

skip = 2  # Reduce frame count
q_log = q_log[::skip]
F_true_log = F_true_log[::skip]
F_est_log = F_est_log[::skip]
r_est_log = r_est_log[::skip]

print(f"Total frames: {len(q_log)}")

# Create animation
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Case 2: External Force Estimation at Unknown Position (3-DOF)', fontsize=14, fontweight='bold')

# Top-left: Arm animation
ax_arm = axes[0, 0]
ax_arm.set_xlim([-3.5, 3.5])
ax_arm.set_ylim([-3.5, 3.5])
ax_arm.set_aspect('equal')
ax_arm.grid(True)
ax_arm.set_xlabel('x (m)')
ax_arm.set_ylabel('y (m)')
ax_arm.set_title('Arm Configuration & Force Vectors')

# Top-right: Fx comparison
ax_fx = axes[0, 1]
ax_fx.set_xlim([0, len(q_log)])
ax_fx.set_ylim([-6, 6])
ax_fx.grid(True)
ax_fx.set_xlabel('Frame')
ax_fx.set_ylabel('$F_x$ (N)')
ax_fx.set_title('$F_x$: True vs Estimated')

# Bottom-left: Fy comparison
ax_fy = axes[1, 0]
ax_fy.set_xlim([0, len(q_log)])
ax_fy.set_ylim([-4, 4])
ax_fy.grid(True)
ax_fy.set_xlabel('Frame')
ax_fy.set_ylabel('$F_y$ (N)')
ax_fy.set_title('$F_y$: True vs Estimated')

# Bottom-right: r comparison
ax_r = axes[1, 1]
ax_r.set_xlim([0, len(q_log)])
ax_r.set_ylim([0, 1.2])
ax_r.grid(True)
ax_r.set_xlabel('Frame')
ax_r.set_ylabel('r (m)')
ax_r.set_title('Contact Position: True vs Estimated')

(line_arm,) = ax_arm.plot([], [], 'o-', linewidth=3, markersize=8, color='blue')
(contact_point,) = ax_arm.plot([], [], 'go', markersize=10)
text_info = ax_arm.text(0.02, 0.98, '', transform=ax_arm.transAxes, fontsize=9,
                             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

(fx_true_line,) = ax_fx.plot([], [], 'b-', linewidth=2, label='True')
(fx_est_line,) = ax_fx.plot([], [], 'r--', linewidth=1.5, label='Estimated')
ax_fx.legend(loc='upper right')

(fy_true_line,) = ax_fy.plot([], [], 'b-', linewidth=2, label='True')
(fy_est_line,) = ax_fy.plot([], [], 'r--', linewidth=1.5, label='Estimated')
ax_fy.legend(loc='upper right')

(r_true_line,) = ax_r.plot([], [], 'b-', linewidth=2, label='True')
(r_est_line,) = ax_r.plot([], [], 'r--', linewidth=1.5, label='Estimated')
ax_r.legend(loc='upper right')

# Force arrows
arrow_true = None
arrow_est = None


def init():
    line_arm.set_data([], [])
    contact_point.set_data([], [])
    text_info.set_text('')
    fx_true_line.set_data([], [])
    fx_est_line.set_data([], [])
    fy_true_line.set_data([], [])
    fy_est_line.set_data([], [])
    r_true_line.set_data([], [])
    r_est_line.set_data([], [])
    return line_arm, contact_point, text_info, fx_true_line, fx_est_line, fy_true_line, fy_est_line, r_true_line, r_est_line


def update(frame):
    global arrow_true, arrow_est

    # Clear previous arrows
    if arrow_true is not None:
        arrow_true.remove()
    if arrow_est is not None:
        arrow_est.remove()

    q = q_log[frame]
    F_true = F_true_log[frame]
    F_est = F_est_log[frame]
    r_est = r_est_log[frame]

    # Forward kinematics
    j1 = np.array([L1*np.cos(q[0]), L1*np.sin(q[0])])
    j2 = j1 + np.array([L2*np.cos(q[0]+q[1]), L2*np.sin(q[0]+q[1])])
    ee = j2 + np.array([L3*np.cos(q[0]+q[1]+q[2]), L3*np.sin(q[0]+q[1]+q[2])])

    # Contact point on link 3
    theta = q[0] + q[1] + q[2]
    cp = j2 + np.array([r_contact*np.cos(theta), r_contact*np.sin(theta)])
    cp_est = j2 + np.array([max(0, min(r_est, L3))*np.cos(theta), max(0, min(r_est, L3))*np.sin(theta)])

    # Draw arm
    x_coords = [0, j1[0], j2[0], ee[0]]
    y_coords = [0, j1[1], j2[1], ee[1]]
    line_arm.set_data(x_coords, y_coords)
    contact_point.set_data([cp[0]], [cp[1]])

    # True force arrow (red)
    scale = 0.2
    arrow_true = FancyArrowPatch((cp[0], cp[1]),
                                  (cp[0] + scale*F_true[0], cp[1] + scale*F_true[1]),
                                  arrowstyle='->', mutation_scale=25, linewidth=2.5,
                                  color='red', label='True Force')
    ax_arm.add_patch(arrow_true)

    # Estimated force arrow (blue dashed)
    arrow_est = FancyArrowPatch((cp_est[0], cp_est[1]),
                                 (cp_est[0] + scale*F_est[0], cp_est[1] + scale*F_est[1]),
                                 arrowstyle='->', mutation_scale=25, linewidth=2,
                                 color='blue', linestyle='--', label='Est. Force')
    ax_arm.add_patch(arrow_est)

    # Info text
    text_info.set_text(f'Frame: {frame}\nTrue:  Fx={F_true[0]:.2f}, Fy={F_true[1]:.2f}, r={r_contact:.2f}\nEst:   Fx={F_est[0]:.2f}, Fy={F_est[1]:.2f}, r={r_est:.2f}')

    # Update plots
    x = np.arange(frame + 1)
    fx_true_line.set_data(x, F_true_log[:frame+1, 0])
    fx_est_line.set_data(x, F_est_log[:frame+1, 0])
    fy_true_line.set_data(x, F_true_log[:frame+1, 1])
    fy_est_line.set_data(x, F_est_log[:frame+1, 1])
    r_true_line.set_data(x, [r_contact]*(frame+1))
    r_est_line.set_data(x, r_est_log[:frame+1])

    return line_arm, contact_point, text_info, fx_true_line, fx_est_line, fy_true_line, fy_est_line, r_true_line, r_est_line


ani = animation.FuncAnimation(fig, update, frames=len(q_log), init_func=init,
                              interval=1000/30, blit=False)

output_video = "/data2/zzhangg/Robotics-HW-3/case2_estimation.mp4"
writer = animation.FFMpegWriter(fps=30, bitrate=5000)
ani.save(output_video, writer=writer)
plt.close()
print(f"Video saved to: {output_video}")
