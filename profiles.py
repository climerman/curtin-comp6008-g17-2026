"""
profiles.py - Synthetic hourly data for the home energy scheduling environment.

Everything here is a 24-value array (one value per hour, index 0 = midnight).
All numbers are illustrative assumptions for a typical Perth household;
state them as assumptions in the report.

Usage:
    from profiles import sample_day
    day = sample_day(rng)          # rng = np.random.default_rng(seed)
    day["solar"], day["load"], day["price"], ...
"""

import numpy as np

HOURS = np.arange(24)

# ---------------------------------------------------------------------------
# Solar generation (kW) - bell curve between sunrise and sunset, scaled by weather
# ---------------------------------------------------------------------------
SOLAR_PEAK_KW = 5.0            # roughly a 6.6 kW rooftop system on a clear day
SUNRISE, SUNSET = 6, 18

WEATHER_TYPES = ["sunny", "cloudy", "rainy"]
WEATHER_PROBS = [0.6, 0.3, 0.1]                          # Perth is mostly sunny
WEATHER_SCALE = {"sunny": 1.0, "cloudy": 0.5, "rainy": 0.2}


def solar_profile(weather, rng=None, noise=0.1):
    """Hourly solar output (kW) for a given weather type."""
    curve = np.zeros(24)
    daylight = (HOURS >= SUNRISE) & (HOURS <= SUNSET)
    # half-sine shape: 0 at sunrise/sunset, peak at midday
    curve[daylight] = np.sin(np.pi * (HOURS[daylight] - SUNRISE) / (SUNSET - SUNRISE))
    solar = SOLAR_PEAK_KW * WEATHER_SCALE[weather] * curve
    if rng is not None:                                   # small random variation
        solar *= 1 + rng.normal(0, noise, 24)
    return np.clip(solar, 0, None)


# ---------------------------------------------------------------------------
# Household base load (kW) - things we do NOT control (fridge, lights, cooking)
# 4x6 hours to baseline.
# ---------------------------------------------------------------------------
BASE_LOAD = np.array([
    0.4, 0.3, 0.3, 0.3, 0.3, 0.4,    # 00-05 night
    0.8, 1.2, 1.0, 0.6, 0.5, 0.5,    # 06-11 morning peak
    0.6, 0.5, 0.5, 0.6, 0.9, 1.5,    # 12-17 afternoon
    2.0, 1.8, 1.4, 1.0, 0.7, 0.5,    # 18-23 evening peak
])


def load_profile(rng=None, noise=0.1):
    load = BASE_LOAD.copy()
    if rng is not None:
        load *= 1 + rng.normal(0, noise, 24)
    return np.clip(load, 0.1, None)


# ---------------------------------------------------------------------------
# Time-of-use tariff ($/kWh) - cheap midday, expensive evening peak
# ---------------------------------------------------------------------------
PRICE_OFFPEAK, PRICE_MIDDAY, PRICE_PEAK = 0.25, 0.10, 0.55
FEED_IN_PRICE = 0.03            # paid for exporting solar to the grid


def price_profile():
    price = np.full(24, PRICE_OFFPEAK)
    price[9:15] = PRICE_MIDDAY       # 9am-3pm
    price[15:21] = PRICE_PEAK        # 3pm-9pm
    return price


# ---------------------------------------------------------------------------
# Home battery and EV
# ---------------------------------------------------------------------------
BATTERY = {"capacity_kwh": 10.0, "max_rate_kw": 5.0, "efficiency": 0.95}

EV = {
    "capacity_kwh": 60.0,
    "max_rate_kw": 7.0,
    "arrive_hour": 18,          # plugged in from 6pm ...
    "depart_hour": 7,           # ... until 7am next morning
    "arrive_soc": 0.4,          # 40% charge when it gets home
    "target_soc": 0.8,          # needs 80% by departure (penalty if not)
}


def ev_plugged_in():
    """Boolean array: True for hours the EV is at home and can charge."""
    return (HOURS >= EV["arrive_hour"]) | (HOURS < EV["depart_hour"])


# ---------------------------------------------------------------------------
# Shiftable appliances - must run once per day, agent chooses when
# ---------------------------------------------------------------------------
APPLIANCES = {
    "washing_machine": {"power_kw": 1.0, "duration_h": 2, "earliest": 7,  "deadline": 22},
    "dishwasher":      {"power_kw": 1.2, "duration_h": 2, "earliest": 8,  "deadline": 23},
}


# ---------------------------------------------------------------------------
# One episode = one simulated day
# ---------------------------------------------------------------------------
def sample_day(rng, weather=None):
    """Return all profiles for one day. Weather is random unless given."""
    if weather is None:
        weather = rng.choice(WEATHER_TYPES, p=WEATHER_PROBS)
    return {
        "weather": weather,
        "solar": solar_profile(weather, rng),
        "load": load_profile(rng),
        "price": price_profile(),
        "feed_in": FEED_IN_PRICE,
        "ev_plugged_in": ev_plugged_in(),
    }
