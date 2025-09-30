# Sensor Coverage Comparison

## Comparison to all sensors

### Present in the **Sensor Overview** but **Missing from Topic/Key List (2024-10-01-11-29-55)**

- **Velodyne VLP-16 (ANYmal LiDAR)**  
  - Appears in the overview as a LiDAR on the ANYmal robot.  
  - No `velodyne`-related topics in the list.  
  - Only Hesai + Livox LiDARs are represented.  

- **ZED2i Integrated IMU**  
  - Overview lists ZED2i IMU (45 Hz).  
  - Topic list has ZED2i images, depth, and VIO, but no IMU topic.  

- **Alphasense MONO Cameras**  
  - Overview says 2× Alphasense 1.6 MONO.  
  - Topic list only has `alphasense_front_left/right/center`, `alphasense_left/right` → all RGB or not clearly marked MONO.  
  - Likely MONO cameras are not logged separately (or collapsed into left/right naming).  

- **Leica MS60 + AP20 Ground Truth**  
  - Overview explicitly lists this as ground truth.  
  - Topic list has `ap20_imu` and `prism_position`, but not a combined Leica ground truth topic.  

---

### Present in **Topic/Key List (2024-10-01-11-29-55)** but **Not in Sensor Overview**

- **ANYmal Battery (`anymal_state_battery`)**  
  - Present in topic list but not in overview.  
  - Likely added for robot health monitoring, not perception.  

- **Various Derived Odometry Topics**  
  - Examples: `dlio_map_odometry`, `cpt7_ie_tc_odometry`, `anymal_state_odometry`.  
  - These are fused state estimators or derived trajectories, not raw sensors → excluded from high-level overview.  

- **STIM320 Temperature Sensors**  
  - `stim320_accelerometer_temperature` and `stim320_gyroscope_temperature`.  
  - Overview only lists “STIM320 IMU 500 Hz,” but doesn’t mention these sub-measurements.  

## ISAAC ↔ Grand-Tour (2024-10-01-11-29-55) Observation Matching

### 1 Direct one-to-one matches

| ISAAC observation | Grand-Tour signal(s) | Notes | Source |
|-------------------|-----------------------|-------|--------|
| Base linear velocity | `anymal_state_odometry.twist_lin`, `anymal_state_state_estimator.twist_lin`, CPT7 IE odometry `*_odometry.twist_lin` | Multiple sources of body-frame linear velocity. | 2024-10-01-11-29-55_data |
| Base angular velocity | `anymal_state_odometry.twist_ang`, `anymal_state_state_estimator.twist_ang`, IMUs (`anymal_imu.ang_vel`, `adis_imu.ang_vel`, `alphasense_imu.ang_vel`) | Use estimator twist directly or IMU gyro if preferred. | 2024-10-01-11-29-55_data |
| Joint positions | `anymal_state_actuator.*_state_joint_position (0–11)` | 12 joints with per-joint streams. | 2024-10-01-11-29-55_data |
| Joint velocities | `anymal_state_actuator.*_state_joint_velocity` | Matches joint velocity requirement. | 2024-10-01-11-29-55_data |
| Previous actions (last joint commands) | `anymal_state_actuator.*_command_position`, `*_command_velocity`, `*_command_joint_torque`, `*_command_current`, `*_command_mode`, `*_command_pid_gains_*` | More detailed than ISAAC’s “previous action.” Pick your controller’s action channel(s). | 2024-10-01-11-29-55_data |
| Commanded base linear velocity (target x,y) | `anymal_command_twist.linear` | Robot-level velocity command topic. | 2024-10-01-11-29-55_data |
| Commanded base angular velocity (target yaw rate) | `anymal_command_twist.angular (use z)` | Yaw-rate command available. | 2024-10-01-11-29-55_data |

---

### 2 Derivable with light processing

| ISAAC observation | Grand-Tour signal(s) | How to derive | Source |
|-------------------|-----------------------|---------------|--------|
| Gravity vector measurement | IMU accelerometers: `anymal_imu.lin_acc`, `adis_imu.lin_acc`, `alphasense_imu.lin_acc` | Low-pass the accelerometer in quasi-static intervals or estimate gravity via orientation + 1g magnitude; remove dynamic acceleration if needed. | 2024-10-01-11-29-55_data |
| Base linear / angular velocity (alternative) | IMU + pose sources | If you prefer pure proprioception: integrate/fuse IMU with kinematics or use estimator `twist_*` directly (Section 1). | 2024-10-01-11-29-55_data |

---

### 3 Likely missing (not directly logged)

| ISAAC observation | Status in Grand-Tour | Possible workaround | Source |
|-------------------|-----------------------|---------------------|--------|
| 108 terrain height measurements around base | Not found as pre-sampled heights | Reconstruct heights from LiDAR point clouds (`hesai_points`, `livox_points`, and their undistorted variants) using a local 2.5D height map around the base, then sample on the same 108-cell grid to emulate the ISAAC observation. | 2024-10-01-11-29-55_data |
| Explicit sim-to-real noise channels (joints ±0.01 rad, joint vel ±1.5 rad/s, base lin vel ±0.01 m/s, base ang vel ±0.2 rad/s, projected gravity ±0.05 rad/s², terrain heights ±0.1 m) | Not logged as separate noisy streams | Calibrate noise offline: compute empirical residuals between overlapping sensors/estimators (e.g., IMU vs state estimator vs CPT7 IE) and fit noise magnitudes to match ISAAC’s schedule before injecting into sim. | 2024-10-01-11-29-55_data |

## ISAAC ↔ Grand-Tour (2024-10-01-11-47-44) Observation Matching
---

### 1 Direct one-to-one matches

| ISAAC observation | Grand-Tour signal(s) | Notes |
|---|---|---|
| **Base linear velocity** | `anymal_state_odometry.twist_lin`, `anymal_state_state_estimator.twist_lin`, CPT7 IE `*_odometry.twist_lin` | Multiple sources of body-frame linear velocity. |
| **Base angular velocity** | `anymal_state_odometry.twist_ang`, `anymal_state_state_estimator.twist_ang`; IMUs: `anymal_imu.ang_vel`, `adis_imu.ang_vel`, `alphasense_imu.ang_vel` | Use estimator twist directly or IMU gyro. |
| **Joint positions** | `anymal_state_actuator.*_state_joint_position` (0–11) | 12 joints with per-joint streams. |
| **Joint velocities** | `anymal_state_actuator.*_state_joint_velocity` | Matches ISAAC joint-vel requirement. |
| **Previous actions (last joint commands)** | `anymal_state_actuator.*_command_position`, `*_command_velocity`, `*_command_joint_torque`, `*_command_current`, `*_command_mode`, `*_command_pid_gains_*` | Richer than ISAAC’s “previous action”; choose your action channel(s). |
| **Commanded base linear velocity (target x,y)** | `anymal_command_twist.linear` | Robot-level velocity command. |
| **Commanded base angular velocity (target yaw rate)** | `anymal_command_twist.angular.z` | Yaw-rate command available. |

---

### 2 Derivable with light processing

| ISAAC observation | Grand-Tour signal(s) | How to derive |
|---|---|---|
| **Gravity vector measurement** | IMU accels: `anymal_imu.lin_acc`, `adis_imu.lin_acc`, `alphasense_imu.lin_acc` | Low-pass accel during quasi-static motion; or use orientation to project a 1g vector; subtract dynamic accel when needed. |
| **(Alt) Base linear / angular velocity** | IMUs + pose/estimator | If you prefer proprioception-only: fuse IMU with kinematics; or just use estimator `twist_*` above. |

---

### 3 Likely missing (not directly logged)

| ISAAC observation | Status in Grand-Tour | Practical workaround |
|---|---|---|
| **108 terrain height measurements around the base** | Not pre-sampled | Reconstruct local 2.5D height map from LiDAR (`hesai_points`, `livox_points`, and their **undistorted** variants), in the base frame per timestamp; sample the same 108-cell grid to emulate ISAAC’s vector. |
| **Explicit sim-to-real noise streams**<br/>(joints ±0.01 rad, joint vel ±1.5 rad/s, base lin vel ±0.01 m/s, base ang vel ±0.2 rad/s, projected gravity ±0.05 rad/s², terrain heights ±0.1 m) | Not logged as separate channels | Calibrate σ offline from residuals across overlapping sensors/estimators (e.g., IMU vs odom vs CPT7 IE) and inject during sim training to mirror ISAAC noise. |

---

### 4 Suitability verdict & to-dos

**Verdict:** **Suitable** for ISAAC-style offline training. You have direct coverage for base twists, joint states, previous actions, and commanded twists; gravity is recoverable from IMUs; the only notable gap is the **pre-sampled 108-height vector**, which is straightforward to compute from the available LiDAR. Noise must be **added in sim** using dataset-fit parameters.

**To-dos to match ISAAC exactly:**
1. **Terrain sampler:** LiDAR → base-frame point cloud → rolling 2.5D height map → sample 108 fixed offsets around the base each step.  
2. **Gravity estimator:** Smooth IMU accel and/or use orientation to extract gravity vector; validate against estimator pose if available.  
3. **Noise calibration:** Estimate per-channel σ from empirical residuals (encoders vs actuator state, IMU vs odom, odom vs CPT7 IE) and configure ISAAC sim-to-real noise to match.  


# Mission Comparison: Unique Data Fields

This report highlights the **topics/fields present in one mission but absent in the other**.

---

## ✅ Present in Mission A (2024-10-01-11-29-55) but Missing in Mission B (2024-10-01-11-47-44)

- **cpt7_ie_rt_odometry**
- **cpt7_ie_rt_tf**
- **cpt7_ie_tc_odometry**
- **cpt7_ie_tc_tf**
- **gnss_raw_cpt7_ie_rt**
- **gnss_raw_cpt7_ie_tc**
- **navsatfix_cpt7_ie_tc**
- **hesai_points_undistorted_filtered**
- **livox_points_undistorted_filtered**

These are mostly **GNSS / tightly coupled odometry topics** and some **filtered LiDAR variants**, available only in Mission A.

---

## ✅ Present in Mission B (2024-10-01-11-47-44) but Missing in Mission A (2024-10-01-11-29-55)

- *(No additional GNSS/odometry streams found)*  
- The datasets are otherwise nearly identical, except that Mission B **does not include** the GNSS-derived and filtered LiDAR fields above.

---

## 📝 Summary

- **Mission A (11-29-55)** includes **extra GNSS and filtered LiDAR data**: tightly/loosely coupled CPT7 IE odometry, GNSS raw/RT, and navsatfix fields.  
- **Mission B (11-47-44)** is missing those, but otherwise aligns closely in core sensors (IMUs, cameras, depth, LiDAR raw, actuator states, etc.).
