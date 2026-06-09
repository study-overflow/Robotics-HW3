# 三自由度平面机械臂外力估计

## 依赖

```bash
pip install mujoco numpy matplotlib scipy
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `main.pdf` | 报告（Case 1/2 建模、算法、仿真结果与尖峰分析） |
| `sim_3dof_v2.py` | **主仿真代码**：运行 Case 1 和 Case 2，生成结果图 |
| `manipulator.xml` | MuJoCo 机械臂模型（3-DOF 平面臂） |
| `record_video.py` | 生成 Case 1 机械臂运动 + 外力矢量视频 |
| `video_case2.py` | 生成 Case 2 估计可视化视频（四格画面） |
| `simulation_video.mp4` | Case 1 视频：机械臂运动 + 外力矢量 |
| `case2_estimation.mp4` | Case 2 视频：四格（机械臂 + Fx/Fy/r 估计曲线） |

## 运行

```bash
# 主仿真：生成 fig_case1_results.png, fig_case2_results.png, fig_workspace.png
python sim_3dof_v2.py

# 生成 Case 1 运动视频
python record_video.py

# 生成 Case 2 估计可视化视频
python video_case2.py
```
