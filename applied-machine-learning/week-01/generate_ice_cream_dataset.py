"""
generate_ice_cream_dataset.py
─────────────────────────────────────────────────────────────────────────────
Generates a synthetic `ice_cream_sales.csv` for the PyTorch assignment.

Ground-truth sales model (per flavour, per day)
────────────────────────────────────────────────
Total daily sales are driven by four non-linear components:

  1. Temperature effect    — flat below 10 °C, then a quadratic ramp above 15 °C.
                             (This mirrors a ReLU + acceleration, which is the
                             pedagogical point of the assignment.)

  2. Precipitation penalty — exponential dampening: heavy rain kills sales fast.

  3. Weekend boost         — multiplicative +30 % on Saturdays / Sundays.

  4. Seasonal baseline     — asymmetric sine: summer peak is much higher than the
                             symmetric trough in winter. Achieved by squashing the
                             negative half of sin(2π·d/365) toward zero.

Per-flavour statistics
──────────────────────
Each flavour gets its own market-share function that varies across the year.
This makes the assignment's closing question ("can we use the aggregate model
for ordering?") clearly answerable as *no*, because the flavour mix shifts
throughout the year in ways the aggregate model never sees.

  vanilla    — dominant in summer, small in winter (strong seasonal signal)
  chocolate  — relatively flat year-round (weak seasonal signal)
  strawberry — spring/early-summer peak, drops off sharply in autumn
  mango      — strictly a summer flavour, near-zero in winter

Gaussian noise is added at the flavour level so that the aggregate still has
realistic variance without individual flavours being perfectly correlated.

Output schema
─────────────
date | flavor | temperature | precipitation | units_sold

The file is sorted by (date, flavor) with a fixed, consistent flavor order.
This is required so that students can safely use tensor.reshape() in the
assignment notebook.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────
RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

# ── Calendar ───────────────────────────────────────────────────────────────
START_DATE = "2022-01-01"
END_DATE   = "2024-12-31"           # three full years → 1096 days
FLAVORS    = ["vanilla", "chocolate", "strawberry", "mango"]
N_FLAVORS  = len(FLAVORS)

dates = pd.date_range(START_DATE, END_DATE, freq="D")
N_DAYS = len(dates)

# ══════════════════════════════════════════════════════════════════════════
# 1.  Weather simulation
# ══════════════════════════════════════════════════════════════════════════

def simulate_temperature(doy: np.ndarray, rng) -> np.ndarray:
    """
    Seasonal temperature for a mid-latitude location (e.g. central Europe).
    Annual mean ≈ 11 °C, amplitude ≈ 13 °C, plus day-to-day noise.
    """
    # Sine is shifted to the left by 80 days. 
    # Zero-intercept is at doy = 80, such that the peak is at doy = 80 + 365/4 = 171.25 (corresponding to June 20th in normal years).
    seasonal = 11 + 13 * np.sin(2 * np.pi * (doy - 80) / 365)                  # peaks ≈ late June
    noise    = rng.normal(0, 2.5, size=N_DAYS)
    # Clip the temperature values in the range [-2, 40].
    return np.clip(seasonal + noise, -8, 40).round(1)


def simulate_precipitation(doy: np.ndarray, rng) -> np.ndarray:
    """
    Most days are dry; wet days follow an exponential distribution.
    Slightly higher precipitation probability in winter/spring.
    """
    # Cosine is highest in winter where doy / 365 is close to either 0 or 1.
    wet_prob = 0.25 + 0.08 * np.cos(2 * np.pi * doy / 365)                     # ~33 % in winter
    is_wet   = rng.random(N_DAYS) < wet_prob
    
    # Draw samples from an exponential distribution for wet days. 
    # The expected value (scale) is 6 mm, which gives a realistic mix of light and heavy rain.
    amount   = rng.exponential(scale=6.0, size=N_DAYS)

    # Select precipitation amounts for wet days, and set dry days to 0. Round to 1 decimal place for realism.
    return np.where(is_wet, amount, 0.0).round(1)


# ══════════════════════════════════════════════════════════════════════════
# 2.  Non-linear sales components and seasonal baseline
# ══════════════════════════════════════════════════════════════════════════

def temperature_effect_on_sales(temp: np.ndarray) -> np.ndarray:
    """
    Piecewise non-linear response — intentionally resembles a soft ReLU:

      temp < 10 °C  →  almost no effect (people don't buy ice cream)
      10–15 °C      →  linear ramp (shoulder season)
      > 15 °C       →  quadratic acceleration (summer buying behaviour)

    This is the key non-linearity the assignment asks students to think about.
    A single linear model cannot capture the flat-then-steep shape; a network
    with ReLU activations can, because ReLU is itself a hinge function.
    """
    effect = np.zeros_like(temp)

    # Linear Shoulder Ramp: 10–15 °C [values range between 0 and 10]
    mask_ramp  = (temp >= 10) & (temp < 15)
    effect[mask_ramp] = 2.0 * (temp[mask_ramp] - 10)

    # Quadratic Zone: ≥ 15 °C [values start at 10, then accelerate up to ~110 at 35 °C].
    mask_hot   = temp >= 15
    effect[mask_hot]  = 10 + 4.0 * (temp[mask_hot] - 15) + 0.3 * (temp[mask_hot] - 15) ** 2

    return effect


def precipitation_penalty(precip: np.ndarray) -> np.ndarray:
    """Exponential dampening: even moderate rain cuts sales noticeably."""
    return np.exp(-0.12 * precip)                                              # 1.0 at 0 mm → ~0.30 at 10 mm


def seasonal_baseline(doy: np.ndarray) -> np.ndarray:
    """
    Asymmetric seasonal baseline.

    A symmetric sine would give winter/summer equal (but opposite) deviations.
    Instead we squash the negative half toward zero — winter is just "low",
    not a symmetric mirror of summer. Summer gets a pronounced spike.
    """
    # Sine is shifted to the left by 80 days. 
    # Zero-intercept is at doy = 80, peak is at doy = 80 + 365/4 = 171.25; trough is at doy = 80 + 3*365/4 = 353.75 (corresponding to June 20th and December 20th, respectively in normal years).
    raw = np.sin(2 * np.pi * (doy - 80) / 365)   # peaks ≈ late June
    # Squash the negative half: Multiply the negative winter values by 0.15 to make them only slightly negative, not deep negative
    # [`np.where()` yields `raw`, where `raw >= 0` and otherwise yields `0.15 * raw` (cf. https://numpy.org/devdocs/reference/generated/numpy.where.html)]
    squashed = np.where(raw >= 0, raw, 0.15 * raw)                             # values range from -0.15 to 1.0, with a pronounced summer peak and a mild winter dip
    
    # Scale the baseline to a realistic range [24 ... 70] of expected daily sales (before other effects).
    return 30 + 40 * squashed


def weekend_multiplier(is_weekend: np.ndarray) -> np.ndarray:
    return 1.0 + 0.30 * is_weekend                                             # +30 % on weekends


# ══════════════════════════════════════════════════════════════════════════
# 3.  Per-flavour contribution to sales
# ══════════════════════════════════════════════════════════════════════════
# Each flavour contributes to the total sales based on a smooth function of day-of-year.
# Contributions sum to 1.0 by construction (softmax-like normalisation).
#
# Pedagogical purpose: the flavour mix changes throughout the year, so the
# aggregate model — which only sees total units_sold — cannot infer how many
# vanilla vs. chocolate units to order on a given day.

def flavour_contribution_per_day(doy: np.ndarray) -> np.ndarray:
    """
    Returns an (N_days, N_flavors) array of contributions to the total sales that sum to 1.

    vanilla    — strong summer peak         (warm-weather flavour)           [sine is shifted to the left by 80 days; peak is at doy 171.25 (corresponding to June 20th in normal years)]
    chocolate  — weak seasonal variation    (comfort flavour, steady demand) [cosine is not shifted and peaks at doy 0 and 365 (corresponding to December 31st and January 1st in normal years), but is scaled down to be relatively flat]
    strawberry — spring & early-summer peak (fresh-fruit association)        [sine is shifted to the left by 60 days; peak is at doy 151.25 (corresponding to May 31st in normal years)]
    mango      — concentrated summer spike, near-zero otherwise              [sine is shifted to the left by 90 days; peak is at doy 181.25 (corresponding to June 30th in normal years)]
    """
    # Raw (un-normalised) share signals
    vanilla    = 1.0 + 1.8 * np.clip(np.sin(2 * np.pi * (doy - 80) / 365), 0, 1)
    chocolate  = 1.2 + 0.2 * np.cos(2 * np.pi * doy / 365)           # almost flat
    strawberry = 0.8 + 1.2 * np.clip(np.sin(2 * np.pi * (doy - 60) / 365), 0, 1)
    mango      = 0.1 + 2.5 * np.clip(np.sin(2 * np.pi * (doy - 95) / 365), 0, 1) ** 2

    raw = np.stack([vanilla, chocolate, strawberry, mango], axis=1) 
    # Normalize the raw contributions so that they sum to 1 for each day (softmax-like).
    return raw / raw.sum(axis=1, keepdims=True)

# ══════════════════════════════════════════════════════════════════════════
# 4.  Simulation of total sales, flavour contributions and noise
# ══════════════════════════════════════════════════════════════════════════

doy     = dates.day_of_year.to_numpy()                                         # day of year (1 … 365/366)
doy_wkd = dates.dayofweek.to_numpy()                                           # 0=Mon … 6=Sun  (used for is_weekend below)

simulated_temperature   = simulate_temperature(doy, rng)
simulated_precipitation = simulate_precipitation(doy, rng)
is_weekend              = (doy_wkd >= 5).astype(float)                         # 0 = weekday, 1 = weekend

temperature_effect = temperature_effect_on_sales(simulated_temperature)
precipitation_pen  = precipitation_penalty(simulated_precipitation)
seasonal_baseline  = seasonal_baseline(doy)
weekend_multiplier = weekend_multiplier(is_weekend)

# Total expected daily sales (before flavour split and noise).
expected_total_sales = (seasonal_baseline + temperature_effect) * precipitation_pen * weekend_multiplier

# ── sanity print (suppressed in student-facing output) ──
print(f"Expected daily total  — min: {expected_total_sales.min():.0f}, "
      f"mean: {expected_total_sales.mean():.0f}, max: {expected_total_sales.max():.0f}")

flavour_contributions = flavour_contribution_per_day(doy)   # (N_days, 4)

# Expected units per flavour per day
expected_sales_per_flavor = expected_total_sales[:, np.newaxis] * flavour_contributions          # (N_days, 4)

# Add realistic noise at the flavour level (Poisson-like via Normal approximation).
noise_scale = np.sqrt(np.maximum(expected_sales_per_flavor, 1)) * 1.2
raw_units   = expected_sales_per_flavor + rng.normal(0, noise_scale)
units_sold  = np.clip(np.round(raw_units), 0, None).astype(int)       # (N_days, 4)

print(f"Flavour totals (3 years) — "
      + ", ".join(f"{f}: {units_sold[:, i].sum():,}" for i, f in enumerate(FLAVORS)))


# ══════════════════════════════════════════════════════════════════════════
# 5.  Creation of final DataFrame
# ══════════════════════════════════════════════════════════════════════════
# Layout: one row per (date, flavor), sorted by (date, flavor_index).
# This fixed ordering is what allows students to reshape safely.

records = []
for day_idx in range(N_DAYS):
    for flav_idx, flavor in enumerate(FLAVORS):
        records.append({
            "date"            : dates[day_idx].date(),
            "flavor"          : flavor,
            "temperature"     : simulated_temperature[day_idx],
            "precipitation"   : simulated_precipitation[day_idx],
            "units_sold"      : units_sold[day_idx, flav_idx],
        })

df = pd.DataFrame(records)

# ── Verification ────────────────────────────────────────────────────────────
assert len(df) == N_DAYS * N_FLAVORS, "Row count mismatch"
assert (df.groupby("date")["flavor"].count() == N_FLAVORS).all(), \
    "Not every date has exactly N_FLAVORS rows"
# Confirm the flavor order within each date is consistent (required for reshape)
flavor_order = df.groupby("date")["flavor"].apply(list)
assert all(fl == FLAVORS for fl in flavor_order), \
    "Flavor order is not consistent across dates"

print(f"\nDataset shape : {df.shape}")
print(f"Date range    : {df['date'].min()} → {df['date'].max()}")
print(f"Flavour order : {FLAVORS}  ← fixed across all dates")
print(f"\nSample rows:")
print(df.head(N_FLAVORS * 2).to_string(index=False))

# ── Save ────────────────────────────────────────────────────────────────────
out_path = Path(__file__).parent / "ice_cream_sales.csv"
df.to_csv(out_path, index=False)
print(f"\n✓  Saved to {out_path}  ({len(df):,} rows)")


# ══════════════════════════════════════════════════════════════════════════
# 5.  Ground-truth summary (for instructor reference)
# ══════════════════════════════════════════════════════════════════════════
print("\n── Instructor reference ─────────────────────────────────────────────")
print("Non-linear components baked into the data:")
print("  1. Temperature  : flat below 10 °C, linear ramp 10–15 °C,")
print("                    quadratic above 15 °C  (intentional ReLU analogy)")
print("  2. Precipitation: exponential dampening  exp(-0.12 · mm)")
print("  3. Weekend      : multiplicative +30 %")
print("  4. Seasonality  : asymmetric sine — summer >> winter")
print()
print("Flavour mix (% of daily total, approximate annual average):")
annual_share = units_sold.sum(axis=0) / units_sold.sum()
for f, s in zip(FLAVORS, annual_share):
    print(f"  {f:<12} {100*s:.1f} %")
print()
print("Key teaching point:")
print("  The aggregate model sees  Σ flavours → total units.")
print("  It cannot recover per-flavour shares because those shares")
print("  are not in its targets. Students need a multi-output model")
print("  (or four separate models) trained on individual flavour data.")
