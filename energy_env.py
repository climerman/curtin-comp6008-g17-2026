"""
energy_env.py - Home energy scheduling environment (Gymnasium).

One episode = one day, 24 hourly steps, from 07:00 to 07:00 the next morning.
Starting at 07:00 means the EV's deadline (charged by 07:00 departure) falls
exactly at the end of the episode.

Observation (15 values, fixed order - do not change without telling the group):
    0 hour_sin, 1 hour_cos, 2 battery_soc, 3 ev_soc, 4 ev_plugged_in,
    5 solar_now, 6 solar_next, 7 load_now, 8 price_now, 9 price_next,
    10 sunny, 11 cloudy, 12 rainy, 13 washer_pending, 14 dishwasher_pending

Action (Discrete 12):  action = battery * 4 + ev * 2 + appliance
    battery:   0 = charge, 1 = idle, 2 = discharge
    ev:        0 = wait,   1 = charge
    appliance: 0 = defer,  1 = start next pending appliance
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import profiles as P

START_HOUR = 7
EV_PENALTY_PER_KWH = 1.0      # $ per kWh the EV is short at departure
APPLIANCE_PENALTY = 2.0       # $ if an appliance misses its deadline
SOLAR_MAX = 7.5               # observation-space bound on solar (kW)


def decode_action(action):
    """Split the single action number into its three parts."""
    action = int(action)
    return action // 4, (action // 2) % 2, action % 2


class EnergyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, weather=None):
        super().__init__()
        if weather not in (None, *P.WEATHER_TYPES):
            raise ValueError("weather must be sunny, cloudy or rainy")
        self.fixed_weather = weather          # None = random weather each episode
        self.appliance_names = list(P.APPLIANCES)

        low = np.array([-1, -1] + [0.0] * 15, dtype=np.float32)
        high = np.array([1, 1, 1, 1, 1,
                         SOLAR_MAX, SOLAR_MAX, 5.0,
                         P.PRICE_PEAK, P.PRICE_PEAK,
                         1, 1, 1, 1, 1,
                         2, 2], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(12)
        self.t = 24

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                    # sets self.np_random
        self.day = P.sample_day(self.np_random, self.fixed_weather)
        self.t = 0                                   # step counter 0..23
        self.battery_soc = 0.5
        self.ev_soc = P.EV["arrive_soc"]
        # per appliance: "pending" (not started), "running", "done", "missed"
        self.app_status = {n: "pending" for n in self.appliance_names}
        self.app_hours_left = {n: 0 for n in self.appliance_names}
        return self._obs(), {}

    # ------------------------------------------------------------------- step
    def step(self, action):
        if self.t >= 24:
            raise RuntimeError("Call reset() before starting another day")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer from 0 to 11")
        h = self.hour
        b_act, ev_act, app_act = decode_action(int(action))
        solar = self.day["solar"][h]
        load = self.day["load"][h]
        price = self.day["price"][h]
        penalty = 0.0

        # 1. Appliances: start the next pending one if allowed, then run them
        if app_act == 1:
            for n in self.appliance_names:
                spec = P.APPLIANCES[n]
                latest_start = spec["deadline"] - spec["duration_h"]
                if self.app_status[n] == "pending" and spec["earliest"] <= h <= latest_start:
                    self.app_status[n] = "running"
                    self.app_hours_left[n] = spec["duration_h"]
                    break
        appliance_kwh = 0.0
        for n in self.appliance_names:
            if self.app_status[n] == "running":
                appliance_kwh += P.APPLIANCES[n]["power_kw"]
                self.app_hours_left[n] -= 1
                if self.app_hours_left[n] == 0:
                    self.app_status[n] = "done"

        # 2. EV arrives home at 18:00 with a fixed charge level
        if h == P.EV["arrive_hour"]:
            self.ev_soc = P.EV["arrive_soc"]
        ev_kwh = 0.0
        if ev_act == 1 and self.day["ev_plugged_in"][h]:
            ev_kwh = min(P.EV["max_rate_kw"], (1 - self.ev_soc) * P.EV["capacity_kwh"])
            self.ev_soc += ev_kwh / P.EV["capacity_kwh"]

        # 3. Home battery (efficiency loss on the way in and out)
        cap, rate, eff = (P.BATTERY[k] for k in ("capacity_kwh", "max_rate_kw", "efficiency"))
        battery_before = self.battery_soc * cap
        batt_in = batt_out = 0.0
        if b_act == 0:                                   # charge
            batt_in = min(rate, (1 - self.battery_soc) * cap / eff)
            self.battery_soc += batt_in * eff / cap
        elif b_act == 2:                                 # discharge
            drawn = min(rate, self.battery_soc * cap)
            self.battery_soc -= drawn / cap
            batt_out = drawn * eff
        self.battery_soc = float(np.clip(self.battery_soc, 0, 1))

        # 4. Energy balance: positive = buy from grid, negative = sell
        net = load + appliance_kwh + ev_kwh + batt_in - solar - batt_out
        grid_import, grid_export = max(net, 0.0), max(-net, 0.0)
        cost = grid_import * price - grid_export * self.day["feed_in"]

        # 5. Penalties: appliance past its latest start time
        for n in self.appliance_names:
            spec = P.APPLIANCES[n]
            latest_start = spec["deadline"] - spec["duration_h"]
            if self.app_status[n] == "pending" and h >= latest_start:
                self.app_status[n] = "missed"
                penalty += APPLIANCE_PENALTY

        # 6. Advance time; at 07:00 the EV leaves - penalise any shortfall
        self.t += 1
        terminated = self.t == 24
        shortfall = 0.0
        if terminated:
            shortfall = max(P.EV["target_soc"] - self.ev_soc, 0) * P.EV["capacity_kwh"]
            penalty += EV_PENALTY_PER_KWH * shortfall

        reward = -(cost + penalty)
        info = {"cost": cost, "penalty": penalty, "grid_import": grid_import,
                "grid_export": grid_export, "hour": h, "weather": self.day["weather"],
                "solar": solar, "load": load, "appliance_kwh": appliance_kwh,
                "ev_kwh": ev_kwh, "battery_in": batt_in, "battery_out": batt_out,
                "battery_before": battery_before,
                "battery_after": self.battery_soc * cap,
                "ev_soc": self.ev_soc, "ev_shortfall": shortfall,
                "missed_jobs": sum(1 for n in self.appliance_names
                                   if self.app_status[n] == "missed")}
        return self._obs(), reward, terminated, False, info

    # --------------------------------------------------------------- helpers
    @property
    def hour(self):
        return (START_HOUR + self.t) % 24

    def _obs(self):
        h = self.hour
        nxt = (h + 1) % 24
        weather_onehot = [float(self.day["weather"] == w) for w in P.WEATHER_TYPES]
        pending = [float(self.app_status[n] == "pending") for n in self.appliance_names]
        hours_left = [float(self.app_hours_left[n]) for n in self.appliance_names]
        obs = [
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            self.battery_soc, self.ev_soc, float(self.day["ev_plugged_in"][h]),
            self.day["solar"][h], self.day["solar"][nxt], self.day["load"][h],
            self.day["price"][h], self.day["price"][nxt],
            *weather_onehot, *pending, *hours_left,
        ]
        return np.clip(np.array(obs, dtype=np.float32),
                       self.observation_space.low, self.observation_space.high)
