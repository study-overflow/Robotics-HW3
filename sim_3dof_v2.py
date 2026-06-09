"""
3-DOF Planar Manipulator: External Force Estimation v2
======================================================
Case 1: Force at end-effector (known position).
Case 2: Force at unknown position on link 3 -> FULLY OBSERVABLE for 3-DOF!
For 3-DOF: tau = [tau1, tau2, tau3]^T, unknowns = [Fx, Fy, r]
Using the analytical decoupling:
  tau_diff1 = tau1 - tau2 = L1 * (-s1*Fx + c1*Fy)
  tau_diff2 = tau2 - tau3 = L2 * (-s12*Fx + c12*Fy)
  tau3      = r * (-s123*Fx + c123*Fy)
This gives a 2x2 system for Fx, Fy, then r is computed directly.
Author: Robotics Control Assignment 3
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import os

np.random.seed(42)

L1, L2, L3 = 1.0, 1.0, 1.0
m1, m2, m3 = 1.0, 0.8, 0.6
g_val = 9.81

model_path = os.path.join(os.path.dirname(__file__), "manipulator.xml")
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

dt = model.opt.timestep
T_sim = 6.0
n_steps = int(T_sim / dt)


def compute_jacobian_ee(q):
    """End-effector Jacobian J (2x3) for 3-DOF planar arm."""
    q1, q2, q3 = q[0], q[1], q[2]
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)
    J = np.array([[-L1*s1 - L2*s12 - L3*s123, -L2*s12 - L3*s123, -L3*s123],
                  [ L1*c1 + L2*c12 + L3*c123,  L2*c12 + L3*c123,  L3*c123]], dtype=np.float64)
    return J


def compute_jacobian_at_point(q, r):
    """Jacobian at point on link 3, distance r from joint 3."""
    q1, q2, q3 = q[0], q[1], q[2]
    s1, c1 = np.sin(q1), np.cos(q1)
    s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
    s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)
    J = np.array([[-L1*s1 - L2*s12 - r*s123, -L2*s12 - r*s123, -r*s123],
                  [ L1*c1 + L2*c12 + r*c123,  L2*c12 + r*c123,  r*c123]], dtype=np.float64)
    return J


# ========================================================================
# Case 1: Force at end-effector (known location)
# ========================================================================
def run_case1():
    print("Case 1: Simulating external force at end-effector (3-DOF)...")
    data.qpos[:] = [np.pi/6, np.pi/4, np.pi/6]
    data.qvel[:] = [0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    time_log, F_true_log, F_est_log = [], [], []

    for step in range(n_steps):
        t = step * dt
        q_des = np.array([np.pi/6 + 0.1*np.sin(0.5*t),
                          np.pi/4 + 0.1*np.cos(0.6*t),
                          np.pi/6 + 0.08*np.sin(0.4*t)])
        dq_des = np.array([0.05*np.cos(0.5*t), -0.06*np.sin(0.6*t), 0.032*np.cos(0.4*t)])

        F_ext = np.array([5.0*np.sin(2.0*t), 3.0*np.cos(1.5*t)])

        Kp, Kd = 60.0, 8.0
        tau_control = Kp*(q_des - data.qpos[:]) + Kd*(dq_des - data.qvel[:])

        J = compute_jacobian_ee(data.qpos)
        tau_ext = J.T @ F_ext
        data.ctrl[:] = tau_control + tau_ext

        mujoco.mj_step(model, data)

        tau_ext_est = data.ctrl - tau_control
        J = compute_jacobian_ee(data.qpos)
        try:
            F_est = np.linalg.lstsq(J.T, tau_ext_est, rcond=None)[0]
        except np.linalg.LinAlgError:
            F_est = np.zeros(2)

        time_log.append(t)
        F_true_log.append(F_ext.copy())
        F_est_log.append(F_est.copy())

    time_log = np.array(time_log)
    F_true = np.array(F_true_log)
    F_est = np.array(F_est_log)

    skip = 5
    time_log = time_log[skip:]
    F_true = F_true[skip:]
    F_est = F_est[skip:]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].plot(time_log, F_true[:, 0], 'b-', label='True', linewidth=2)
    axes[0, 0].plot(time_log, F_est[:, 0], 'r--', label='Estimated', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Force (N)')
    axes[0, 0].set_title('Case 1: $F_x$ Estimation (3-DOF)')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(time_log, F_true[:, 1], 'b-', label='True', linewidth=2)
    axes[0, 1].plot(time_log, F_est[:, 1], 'r--', label='Estimated', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Force (N)')
    axes[0, 1].set_title('Case 1: $F_y$ Estimation (3-DOF)')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    err_x = F_true[:, 0] - F_est[:, 0]
    err_y = F_true[:, 1] - F_est[:, 1]
    axes[1, 0].plot(time_log, err_x, 'g-', label='$F_x$ error', linewidth=1.5)
    axes[1, 0].plot(time_log, err_y, 'm-', label='$F_y$ error', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('Error (N)')
    axes[1, 0].set_title('Case 1: Estimation Error (3-DOF)')
    axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(F_true[:, 0], F_true[:, 1], 'b-', label='True', linewidth=2)
    axes[1, 1].plot(F_est[:, 0], F_est[:, 1], 'r--', label='Estimated', linewidth=1.5)
    axes[1, 1].set_xlabel('$F_x$ (N)'); axes[1, 1].set_ylabel('$F_y$ (N)')
    axes[1, 1].set_title('Case 1: Force Trajectory (3-DOF)')
    axes[1, 1].legend(); axes[1, 1].grid(True); axes[1, 1].axis('equal')

    plt.tight_layout()
    plt.savefig('fig_case1_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Case 1 done. Mean abs error: Fx={np.mean(np.abs(err_x)):.4f}N, Fy={np.mean(np.abs(err_y)):.4f}N")
    return time_log, F_true, F_est


# ========================================================================
# Case 2: Force at unknown position on link 3
# Key insight for 3-DOF:
#   tau_ext = [tau1, tau2, tau3]^T = J(q, r)^T @ [Fx, Fy]
#   tau1 - tau2 = L1 * (-s1*Fx + c1*Fy)
#   tau2 - tau3 = L2 * (-s12*Fx + c12*Fy)
#   tau3 = r * (-s123*Fx + c123*Fy)
# This gives 3 eqns for 3 unknowns (Fx, Fy, r) -> fully observable!
# ========================================================================
def run_case2():
    print("Case 2: Simulating external force at unknown location (3-DOF)...")
    data.qpos[:] = [0.0, 0.0, 0.0]
    data.qvel[:] = [0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    time_log, F_true_log, r_true_log = [], [], []
    F_est_log, r_est_log = [], []

    r_contact = 0.6  # Unknown to estimator

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

        # Extract tau_ext from simulation
        tau_ext_est = data.ctrl - tau_control
        tau1, tau2, tau3 = tau_ext_est[0], tau_ext_est[1], tau_ext_est[2]

        # Current joint angles
        q = data.qpos
        q1, q2, q3 = q[0], q[1], q[2]
        s1, c1 = np.sin(q1), np.cos(q1)
        s12, c12 = np.sin(q1+q2), np.cos(q1+q2)
        s123, c123 = np.sin(q1+q2+q3), np.cos(q1+q2+q3)

        # Solve for Fx, Fy using the decoupled equations:
        # tau1 - tau2 = L1 * (-s1*Fx + c1*Fy)
        # tau2 - tau3 = L2 * (-s12*Fx + c12*Fy)
        A = np.array([[-L1*s1,  L1*c1 ],
                      [-L2*s12, L2*c12]], dtype=np.float64)
        b = np.array([tau1 - tau2, tau2 - tau3], dtype=np.float64)

        det_A = (-L1*s1)*(L2*c12) - (L1*c1)*(-L2*s12)
        det_A = -L1*L2*(s1*c12 - c1*s12)
        # s1*c12 - c1*s12 = sin(q1)*cos(q1+q2) - cos(q1)*sin(q1+q2) = -sin(q2)
        # So det_A = L1*L2*sin(q2)

        try:
            # Solve 2x2 linear system for Fx, Fy
            if abs(det_A) > 1e-6:  # Non-singular
                Fx_est = ( b[0]*L2*c12 - b[1]*L1*c1 ) / det_A
                # Actually let's just use np.linalg.solve
                sol = np.linalg.solve(A, b)
                Fx_est, Fy_est = sol[0], sol[1]

                # Then solve for r: tau3 = r * (-s123*Fx + c123*Fy)
                denom = -s123*Fx_est + c123*Fy_est
                if abs(denom) > 1e-6:
                    r_est = tau3 / denom
                else:
                    r_est = 0.0
            else:
                Fx_est, Fy_est, r_est = 0.0, 0.0, 0.0
        except np.linalg.LinAlgError:
            Fx_est, Fy_est, r_est = 0.0, 0.0, 0.0

        time_log.append(t)
        F_true_log.append(F_ext.copy())
        r_true_log.append(r_contact)
        F_est_log.append([Fx_est, Fy_est])
        r_est_log.append(r_est)

    time_log = np.array(time_log)
    F_true = np.array(F_true_log)
    F_est = np.array(F_est_log)
    r_true = np.array(r_true_log)
    r_est = np.array(r_est_log)

    # Skip initial transient
    skip = 20
    time_log = time_log[skip:]
    F_true = F_true[skip:]
    F_est = F_est[skip:]
    r_true = r_true[skip:]
    r_est = r_est[skip:]

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(time_log, F_true[:, 0], 'b-', label='True $F_x$', linewidth=2)
    axes[0, 0].plot(time_log, F_est[:, 0], 'r--', label='Est. $F_x$', linewidth=1.5)
    axes[0, 0].set_xlabel('Time (s)'); axes[0, 0].set_ylabel('Force (N)')
    axes[0, 0].set_title('Case 2: $F_x$ Estimation (3-DOF)')
    axes[0, 0].legend(); axes[0, 0].grid(True)

    axes[0, 1].plot(time_log, F_true[:, 1], 'b-', label='True $F_y$', linewidth=2)
    axes[0, 1].plot(time_log, F_est[:, 1], 'r--', label='Est. $F_y$', linewidth=1.5)
    axes[0, 1].set_xlabel('Time (s)'); axes[0, 1].set_ylabel('Force (N)')
    axes[0, 1].set_title('Case 2: $F_y$ Estimation (3-DOF)')
    axes[0, 1].legend(); axes[0, 1].grid(True)

    err_x = F_true[:, 0] - F_est[:, 0]
    err_y = F_true[:, 1] - F_est[:, 1]
    axes[1, 0].plot(time_log, err_x, 'g-', label='$F_x$ error', linewidth=1.5)
    axes[1, 0].plot(time_log, err_y, 'm-', label='$F_y$ error', linewidth=1.5)
    axes[1, 0].set_xlabel('Time (s)'); axes[1, 0].set_ylabel('Error (N)')
    axes[1, 0].set_title('Case 2: Estimation Error (3-DOF)')
    axes[1, 0].legend(); axes[1, 0].grid(True)

    axes[1, 1].plot(time_log, r_true, 'b-', label='True $r$', linewidth=2)
    axes[1, 1].plot(time_log, r_est, 'r--', label='Est. $r$', linewidth=1.5)
    axes[1, 1].set_xlabel('Time (s)'); axes[1, 1].set_ylabel('Distance from Joint 3 (m)')
    axes[1, 1].set_title('Case 2: Location Estimation (3-DOF)')
    axes[1, 1].legend(); axes[1, 1].grid(True)
    axes[1, 1].set_ylim([0, L3])

    plt.tight_layout()
    plt.savefig('fig_case2_results.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Case 2 (3-DOF) done.")
    print(f"Mean abs error: Fx={np.mean(np.abs(err_x)):.4f}N, Fy={np.mean(np.abs(err_y)):.4f}N, r={np.mean(np.abs(r_true - r_est)):.4f}m")
    print(f"Final: True r={r_true[-1]:.3f}, Est r={r_est[-1]:.3f}")
    return time_log, F_true, F_est, r_true, r_est


# ========================================================================
# Visualization
# ========================================================================
def plot_workspace():
    fig, ax = plt.subplots(figsize=(8, 8))
    theta = np.linspace(0, 2*np.pi, 200)
    r_max = L1 + L2 + L3
    ax.plot(r_max * np.cos(theta), r_max * np.sin(theta), 'b--', label='Workspace boundary')

    for q in [[0, 0, 0], [np.pi/6, np.pi/4, np.pi/6], [np.pi/3, -np.pi/4, np.pi/6]]:
        j1 = np.array([L1*np.cos(q[0]), L1*np.sin(q[0])])
        j2 = j1 + np.array([L2*np.cos(q[0]+q[1]), L2*np.sin(q[0]+q[1])])
        ee = j2 + np.array([L3*np.cos(q[0]+q[1]+q[2]), L3*np.sin(q[0]+q[1]+q[2])])
        ax.plot([0, j1[0], j2[0], ee[0]], [0, j1[1], j2[1], ee[1]], 'o-', linewidth=2, markersize=6)

    ax.set_xlim([-3.5, 3.5]); ax.set_ylim([-3.5, 3.5])
    ax.set_aspect('equal'); ax.grid(True)
    ax.set_title('3-DOF Planar Manipulator Workspace')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(['Workspace boundary'], loc='upper right')
    plt.tight_layout()
    plt.savefig('fig_workspace.png', dpi=300, bbox_inches='tight')
    plt.close()


# ========================================================================
# Main
# ========================================================================
if __name__ == "__main__":
    print("Generating figures for 3-DOF manipulator...")
    plot_workspace()
    time1, F_true1, F_est1 = run_case1()
    data = mujoco.MjData(model)
    time2, F_true2, F_est2, r_true, r_est = run_case2()
    print("\nAll figures for 3-DOF generated successfully!")
