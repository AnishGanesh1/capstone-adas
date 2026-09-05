# LiDAR degradation by weather condition

Baseline: `clear` = 33577 points per sweep (n=100).

| Condition | Tier | Frames | Mean points | Retained | Attenuation | Dropoff | Noise |
|---|---|---|---|---|---|---|---|
| `clear` | reference | 100 | 33577 | 100.0% | 0.004 | 0.45 | 0.00 |
| `midnight` | severe | 450 | 33119 | 98.6% | 0.004 | 0.45 | 0.00 |
| `night` | mild | 300 | 33116 | 98.6% | 0.004 | 0.45 | 0.00 |
| `glare_dawn` | severe | 450 | 32955 | 98.1% | 0.006 | 0.45 | 0.00 |
| `haze` | mild | 300 | 32819 | 97.7% | 0.010 | 0.45 | 0.01 |
| `wet_night` | mild | 300 | 32416 | 96.5% | 0.006 | 0.45 | 0.01 |
| `glare_wet` | severe | 450 | 29279 | 87.2% | 0.015 | 0.50 | 0.03 |
| `rain` | mild | 300 | 26085 | 77.7% | 0.020 | 0.55 | 0.04 |
| `fog` | mild | 300 | 25318 | 75.4% | 0.030 | 0.55 | 0.03 |
| `glare_fog` | extreme | 300 | 23930 | 71.3% | 0.025 | 0.58 | 0.03 |
| `hail` | mild | 300 | 23083 | 68.7% | 0.025 | 0.60 | 0.06 |
| `storm_day` | severe | 450 | 21014 | 62.6% | 0.030 | 0.62 | 0.06 |
| `fog_night` | severe | 450 | 20267 | 60.4% | 0.040 | 0.62 | 0.04 |
| `storm_night` | severe | 450 | 19031 | 56.7% | 0.032 | 0.65 | 0.07 |
| `fog_dense` | severe | 450 | 18713 | 55.7% | 0.045 | 0.65 | 0.05 |
| `deep_night_fog` | extreme | 300 | 18346 | 54.6% | 0.038 | 0.66 | 0.04 |
| `storm_dusk` | extreme | 300 | 17396 | 51.8% | 0.035 | 0.68 | 0.08 |
| `hail_severe` | severe | 450 | 17291 | 51.5% | 0.035 | 0.68 | 0.08 |
| `monsoon` | extreme | 300 | 16025 | 47.7% | 0.040 | 0.70 | 0.09 |
| `fog_night_extreme` | extreme | 300 | 15409 | 45.9% | 0.055 | 0.70 | 0.05 |
| `monsoon_night` | extreme | 300 | 15193 | 45.2% | 0.042 | 0.72 | 0.10 |
| `fog_whiteout` | extreme | 300 | 13919 | 41.5% | 0.060 | 0.72 | 0.06 |
| `blizzard` | extreme | 300 | 13637 | 40.6% | 0.050 | 0.74 | 0.11 |
