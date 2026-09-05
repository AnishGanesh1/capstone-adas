"""Single source of truth. Every script imports from here. Do not duplicate values."""

# ---------------------------------------------------------------- splits
TOWNS_TRAIN = ['Town01', 'Town03']
TOWNS_VAL   = ['Town02']
TOWNS_TEST  = ['Town10HD_Opt']
ALL_TOWNS   = TOWNS_TRAIN + TOWNS_VAL + TOWNS_TEST

# ---------------------------------------------------------------- classes
TL_CLASSES = ['tl_red', 'tl_yellow', 'tl_green', 'tl_off']
TL_CLASS_ID = {'Red': 0, 'Yellow': 1, 'Green': 2, 'Off': 3}

BEV_CLASSES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle', 'pedestrian']

# ---------------------------------------------------------------- sensors
IMAGE_W, IMAGE_H, FOV = 1280, 720, 90.0
FIXED_DELTA = 0.05                       # 20 Hz
CAM_X, CAM_Z = 1.5, 2.4                  # windscreen mount
LIDAR_X, LIDAR_Z = 0.0, 2.4              # roof mount

LIDAR_BASE = {
    'channels': '64',
    'range': '100.0',
    'points_per_second': '1300000',
    'rotation_frequency': str(1.0 / FIXED_DELTA),   # MUST equal 1/fixed_delta
    'upper_fov': '10.0',
    'lower_fov': '-30.0',
    'horizontal_fov': '360.0',
}

# ---------------------------------------------------------------- labelling
MIN_BOX_PX = 8            # discard traffic lights shorter than this
MIN_VISIBLE_FRAC = 0.35   # discard if less of the box is genuinely the lamp
MAX_TL_DIST = 90.0        # metres
SEM_TAG_TRAFFIC_LIGHT = 7   # VERIFIED on this build - do not overwrite   # VERIFY with detect_tl_tag.py before a full run

# ---------------------------------------------------------------- capture
SETTLE_TICKS = 20         # after a weather change, before capturing
TICKS_BETWEEN_SAVES = 10  # ~0.5 s of driving
MIN_MOVE_M = 1.0          # don't save near-duplicate frames while stopped
MAX_SKIPS = 40            # escape hatch if the ego is stuck at a red light
BACKGROUND_KEEP = 0.20    # fraction of no-traffic-light frames to keep
SEED = 42

# ---------------------------------------------------------------- weather
# camera weather + the matching LiDAR degradation.
# CARLA does NOT couple these itself - if you skip the lidar dict your
# "adverse weather LiDAR" is identical to clear-weather LiDAR.
CONDITIONS = {
    'clear': dict(
        weather=dict(cloudiness=0, precipitation=0, precipitation_deposits=0,
                     wind_intensity=0, sun_azimuth_angle=0, sun_altitude_angle=70,
                     fog_density=0, fog_distance=0, fog_falloff=0.1, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.004, dropoff_general_rate=0.45,
                   noise_stddev=0.0)),
    'rain': dict(
        weather=dict(cloudiness=80, precipitation=80, precipitation_deposits=60,
                     wind_intensity=60, sun_azimuth_angle=0, sun_altitude_angle=45,
                     fog_density=10, fog_distance=60, fog_falloff=0.5, wetness=60),
        lidar=dict(atmosphere_attenuation_rate=0.020, dropoff_general_rate=0.55,
                   noise_stddev=0.04)),
    'fog': dict(
        weather=dict(cloudiness=60, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_azimuth_angle=0, sun_altitude_angle=45,
                     fog_density=70, fog_distance=10, fog_falloff=1.5, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.030, dropoff_general_rate=0.55,
                   noise_stddev=0.03)),
    'haze': dict(
        weather=dict(cloudiness=40, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_azimuth_angle=0, sun_altitude_angle=60,
                     fog_density=30, fog_distance=60, fog_falloff=0.8, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.010, dropoff_general_rate=0.45,
                   noise_stddev=0.01)),
    'hail': dict(
        weather=dict(cloudiness=90, precipitation=90, precipitation_deposits=80,
                     wind_intensity=90, sun_azimuth_angle=0, sun_altitude_angle=40,
                     fog_density=20, fog_distance=40, fog_falloff=0.8, wetness=70),
        lidar=dict(atmosphere_attenuation_rate=0.025, dropoff_general_rate=0.60,
                   noise_stddev=0.06)),
    'night': dict(
        weather=dict(cloudiness=20, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_azimuth_angle=0, sun_altitude_angle=-20,
                     fog_density=5, fog_distance=60, fog_falloff=0.5, wetness=0),
        # LiDAR is an ACTIVE sensor - darkness does not degrade it. This is a result.
        lidar=dict(atmosphere_attenuation_rate=0.004, dropoff_general_rate=0.45,
                   noise_stddev=0.0)),
    'wet_night': dict(
        weather=dict(cloudiness=30, precipitation=0, precipitation_deposits=70,
                     wind_intensity=20, sun_azimuth_angle=0, sun_altitude_angle=-20,
                     fog_density=10, fog_distance=50, fog_falloff=0.5, wetness=80),
        lidar=dict(atmosphere_attenuation_rate=0.006, dropoff_general_rate=0.45,
                   noise_stddev=0.01)),

    # ------------------------------------------------------------------
    # SEVERE SET - deliberately harder than anything above.
    # Use: --conditions fog_dense,fog_night,storm_day,storm_night,
    #                   hail_severe,glare_dawn,glare_wet,midnight
    # ------------------------------------------------------------------
    'fog_dense': dict(           # fog begins 5 m ahead - near whiteout
        weather=dict(cloudiness=90, precipitation=0, precipitation_deposits=0,
                     wind_intensity=20, sun_azimuth_angle=0, sun_altitude_angle=35,
                     fog_density=95, fog_distance=5, fog_falloff=2.0, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.045, dropoff_general_rate=0.65,
                   noise_stddev=0.05)),
    'fog_night': dict(           # worst case: no light AND no visibility
        weather=dict(cloudiness=80, precipitation=0, precipitation_deposits=0,
                     wind_intensity=20, sun_azimuth_angle=0, sun_altitude_angle=-25,
                     fog_density=90, fog_distance=8, fog_falloff=1.8, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.040, dropoff_general_rate=0.62,
                   noise_stddev=0.04)),
    'storm_day': dict(           # maximum rain + wind + standing water
        weather=dict(cloudiness=100, precipitation=100, precipitation_deposits=90,
                     wind_intensity=100, sun_azimuth_angle=0, sun_altitude_angle=30,
                     fog_density=25, fog_distance=30, fog_falloff=0.8, wetness=90),
        lidar=dict(atmosphere_attenuation_rate=0.030, dropoff_general_rate=0.62,
                   noise_stddev=0.06)),
    'storm_night': dict(         # rain at night: headlight bloom on wet asphalt
        weather=dict(cloudiness=100, precipitation=100, precipitation_deposits=95,
                     wind_intensity=100, sun_azimuth_angle=0, sun_altitude_angle=-25,
                     fog_density=30, fog_distance=25, fog_falloff=0.8, wetness=100),
        lidar=dict(atmosphere_attenuation_rate=0.032, dropoff_general_rate=0.65,
                   noise_stddev=0.07)),
    'hail_severe': dict(         # approximated - CARLA has no hail model
        weather=dict(cloudiness=100, precipitation=100, precipitation_deposits=100,
                     wind_intensity=100, sun_azimuth_angle=0, sun_altitude_angle=25,
                     fog_density=35, fog_distance=25, fog_falloff=0.9, wetness=90),
        lidar=dict(atmosphere_attenuation_rate=0.035, dropoff_general_rate=0.68,
                   noise_stddev=0.08)),
    'glare_dawn': dict(          # low sun straight into the lens - washes out lamps
        weather=dict(cloudiness=10, precipitation=0, precipitation_deposits=0,
                     wind_intensity=5, sun_azimuth_angle=180, sun_altitude_angle=6,
                     fog_density=15, fog_distance=40, fog_falloff=0.4, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.006, dropoff_general_rate=0.45,
                   noise_stddev=0.0)),
    'glare_wet': dict(           # low sun + wet road = specular chaos
        weather=dict(cloudiness=20, precipitation=20, precipitation_deposits=80,
                     wind_intensity=20, sun_azimuth_angle=0, sun_altitude_angle=8,
                     fog_density=20, fog_distance=35, fog_falloff=0.5, wetness=80),
        lidar=dict(atmosphere_attenuation_rate=0.015, dropoff_general_rate=0.50,
                   noise_stddev=0.03)),
    'midnight': dict(            # true darkness, street lighting only
        weather=dict(cloudiness=60, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_azimuth_angle=0, sun_altitude_angle=-60,
                     fog_density=20, fog_distance=40, fog_falloff=0.6, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.004, dropoff_general_rate=0.45,
                   noise_stddev=0.0)),

    # ------------------------------------------------------------------
    # EXTREME SET - beyond severe. Use when scaling the dataset up.
    # ------------------------------------------------------------------
    'fog_whiteout': dict(
        weather=dict(cloudiness=100, precipitation=0, precipitation_deposits=0,
                     wind_intensity=25, sun_azimuth_angle=0, sun_altitude_angle=30,
                     fog_density=100, fog_distance=2, fog_falloff=2.5, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.060, dropoff_general_rate=0.72,
                   noise_stddev=0.06)),
    'fog_night_extreme': dict(
        weather=dict(cloudiness=100, precipitation=0, precipitation_deposits=0,
                     wind_intensity=25, sun_azimuth_angle=0, sun_altitude_angle=-35,
                     fog_density=100, fog_distance=3, fog_falloff=2.3, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.055, dropoff_general_rate=0.70,
                   noise_stddev=0.05)),
    'monsoon': dict(
        weather=dict(cloudiness=100, precipitation=100, precipitation_deposits=100,
                     wind_intensity=100, sun_azimuth_angle=0, sun_altitude_angle=25,
                     fog_density=45, fog_distance=15, fog_falloff=1.0, wetness=100),
        lidar=dict(atmosphere_attenuation_rate=0.040, dropoff_general_rate=0.70,
                   noise_stddev=0.09)),
    'monsoon_night': dict(
        weather=dict(cloudiness=100, precipitation=100, precipitation_deposits=100,
                     wind_intensity=100, sun_azimuth_angle=0, sun_altitude_angle=-30,
                     fog_density=50, fog_distance=12, fog_falloff=1.0, wetness=100),
        lidar=dict(atmosphere_attenuation_rate=0.042, dropoff_general_rate=0.72,
                   noise_stddev=0.10)),
    'blizzard': dict(
        weather=dict(cloudiness=100, precipitation=100, precipitation_deposits=100,
                     wind_intensity=100, sun_azimuth_angle=0, sun_altitude_angle=20,
                     fog_density=60, fog_distance=12, fog_falloff=1.2, wetness=95),
        lidar=dict(atmosphere_attenuation_rate=0.050, dropoff_general_rate=0.74,
                   noise_stddev=0.11)),
    'glare_fog': dict(
        weather=dict(cloudiness=30, precipitation=0, precipitation_deposits=0,
                     wind_intensity=10, sun_azimuth_angle=180, sun_altitude_angle=4,
                     fog_density=55, fog_distance=20, fog_falloff=0.7, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.025, dropoff_general_rate=0.58,
                   noise_stddev=0.03)),
    'storm_dusk': dict(
        weather=dict(cloudiness=100, precipitation=95, precipitation_deposits=95,
                     wind_intensity=95, sun_azimuth_angle=200, sun_altitude_angle=2,
                     fog_density=40, fog_distance=18, fog_falloff=0.9, wetness=100),
        lidar=dict(atmosphere_attenuation_rate=0.035, dropoff_general_rate=0.68,
                   noise_stddev=0.08)),
    'deep_night_fog': dict(
        weather=dict(cloudiness=80, precipitation=0, precipitation_deposits=0,
                     wind_intensity=15, sun_azimuth_angle=0, sun_altitude_angle=-70,
                     fog_density=70, fog_distance=10, fog_falloff=1.5, wetness=0),
        lidar=dict(atmosphere_attenuation_rate=0.038, dropoff_general_rate=0.66,
                   noise_stddev=0.04)),
}

# convenience groups
CONDITIONS_BASE = ['clear', 'rain', 'fog', 'haze', 'hail', 'night', 'wet_night']
CONDITIONS_MILD = ['rain', 'fog', 'haze', 'hail', 'night', 'wet_night']
CONDITIONS_SEVERE = ['fog_dense', 'fog_night', 'storm_day', 'storm_night',
                     'hail_severe', 'glare_dawn', 'glare_wet', 'midnight']
CONDITIONS_EXTREME = ['fog_whiteout', 'fog_night_extreme', 'monsoon', 'monsoon_night',
                      'blizzard', 'glare_fog', 'storm_dusk', 'deep_night_fog']

# EVERY adverse condition. This is the default for all capture scripts -
# 'clear' is deliberately excluded, it is reference data, not training data.
CONDITIONS_ADVERSE = CONDITIONS_MILD + CONDITIONS_SEVERE + CONDITIONS_EXTREME

TIERS = {
    'mild':    CONDITIONS_MILD,
    'severe':  CONDITIONS_SEVERE,
    'extreme': CONDITIONS_EXTREME,
    'adverse': CONDITIONS_ADVERSE,     # default
    'all':     list(CONDITIONS),
    'ref':     ['clear'],              # measurement reference only
}


def intensify(cond, factor):
    """Scale a preset's severity. factor 1.0 = unchanged, 1.4 = 40% harsher.

    Lets you make the dataset harder without inventing new presets: raises
    precipitation / fog / wind toward their caps, pulls fog_distance in,
    pushes night further below the horizon, and scales the LiDAR degradation
    to match (CARLA will not do that second part for you).
    """
    w = dict(cond['weather'])
    l = dict(cond['lidar'])
    if factor == 1.0:
        return dict(weather=w, lidar=l)

    for k in ('cloudiness', 'precipitation', 'precipitation_deposits',
              'wind_intensity', 'fog_density', 'wetness'):
        if w.get(k, 0) > 0:
            w[k] = round(min(100.0, w[k] * factor), 1)

    if w.get('fog_distance', 0) > 0:
        w['fog_distance'] = round(max(1.0, w['fog_distance'] / factor), 1)
    if 'fog_falloff' in w:
        w['fog_falloff'] = round(min(5.0, w['fog_falloff'] * factor), 2)
    if w.get('sun_altitude_angle', 0) < 0:                 # night gets darker
        w['sun_altitude_angle'] = round(max(-90.0, w['sun_altitude_angle'] * factor), 1)

    caps = dict(atmosphere_attenuation_rate=0.5, dropoff_general_rate=0.9,
                noise_stddev=0.25)
    for k, cap in caps.items():
        if k in l and l[k] > 0:
            l[k] = round(min(cap, l[k] * factor), 4)
    return dict(weather=w, lidar=l)
