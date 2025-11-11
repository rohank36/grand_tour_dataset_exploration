| Sensor                       | In Sync? | Offset (approx) | Notes                  |
| :--------------------------- | :------: | :-------------- | :--------------------- |
| anymal_state_odometry        |     ✅    | +40 ms          | Slight delay after IMU |
| anymal_state_state_estimator |     ✅    | −30 ms          | Slightly earlier       |
| anymal_imu                   |     ✅    | baseline        | reference clock        |
| anymal_state_actuator        |     ✅    | +1 ms           | same domain            |
| hdr_front                    |     ✅    | +10 ms          | after IMU              |
| alphasense_front_center      |     ✅    | −50 → +150 ms   | camera domain offset   |
| anymal_command_twist         |     ❌    | +5 s            | later command loop     |
