# Demo videos

Live traffic-light detection in CARLA. Ego on autopilot in Town10HD_Opt (the
held-out test town), YOLOv8n at 960 px, boxes drawn per frame. The HUD shows the
weather condition, detection count and inference latency.

| File | Condition |
|---|---|
| `demo_clear.mp4` | clear — reference |
| `demo_haze.mp4` | haze — mild |
| `demo_fog_dense.mp4` | dense fog — severe |
| `demo_night.mp4` | night — severe |
| `demo_monsoon_night.mp4` | monsoon at night — extreme |
| `demo_blizzard.mp4` | blizzard — extreme |

Encoded at 1280x720 H.264. Full-resolution 1600x900 originals were not committed.

Watch `demo_clear.mp4` and `demo_night.mp4` back to back: the camera degrades
sharply in darkness while LiDAR retains 98.6% of its clear-weather returns.
