"""Deterministic fixture for the calibration chamber.

Data model (see README "数据口径"):
    V_ab(t, f) = g_a(t) * conj(g_b(t)) * S(t, f) + n
with a four-antenna array, 6 times and 8 channels.

Embedded scenarios:
  * four antennas forming closure triangles;
  * antenna A3 fully flagged (usable as an "offline reference");
  * source phase slope 0.95 rad/channel so visibility phases wrap across
    the negative real-axis branch between channels;
  * a phase jump on A1 at t_idx >= 4 (local phase glitch);
  * narrow-band RFI on baseline A0-A2 at t=2, channels 5-6;
  * a quality-bit-only bad sample without an amplitude excursion.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

ANTENNAS = ["A0", "A1", "A2", "A3"]
ANTENNA_NAMES = {
    "A0": "东-A0",
    "A1": "西-A1",
    "A2": "南-A2",
    "A3": "北-A3(离线)",
}
NTIME = 6
NCHAN = 8
FREQ0_GHZ = 1.0
DF_GHZ = 0.1
T0_ISO = "2026-09-21T00:00:00+00:00"
DT_SECONDS = 30
SIGMA = 0.003
SOURCE_PHASE_SLOPE = 0.95
PHASE_JUMP_RAD = math.radians(70.0)
SEED = 20260921

# true complex gains (amplitude, phase radians) at the reference epoch
TRUE_GAINS = {
    "A0": (1.00, math.radians(0.0)),
    "A1": (1.12, math.radians(35.0)),
    "A2": (0.90, math.radians(-60.0)),
    "A3": (1.05, math.radians(120.0)),
}

# additive RFI bursts keyed by (antenna_a_idx, antenna_b_idx, t_idx, ch_idx)
RFI_CELLS = {
    (0, 2, 2, 5): 5.0 + 0.0j,
    (0, 2, 2, 6): 5.0 + 0.0j,
}

# quality bit set without an amplitude/phase excursion (duty-cycle warning)
QUALITY_ONLY_BAD = {
    (1, 2, 1, 7),
}

FIXTURE_VERSION = "fixture-v1"


def timestamps() -> list[str]:
    from datetime import datetime, timedelta, timezone

    t0 = datetime.fromisoformat(T0_ISO)
    return [
        (t0 + timedelta(seconds=DT_SECONDS * i)).astimezone(timezone.utc).isoformat()
        for i in range(NTIME)
    ]


def frequencies_ghz() -> list[float]:
    return [round(FREQ0_GHZ + DF_GHZ * c, 6) for c in range(NCHAN)]


def true_gain(antenna_idx: int, t_idx: int) -> complex:
    amp, phase = TRUE_GAINS[ANTENNAS[antenna_idx]]
    if antenna_idx == 1 and t_idx >= 4:
        phase = phase + PHASE_JUMP_RAD
    return amp * complex(math.cos(phase), math.sin(phase))


def source_vis(t_idx: int, ch_idx: int) -> complex:
    amp = 2.0 + 0.05 * ch_idx
    phase = SOURCE_PHASE_SLOPE * ch_idx
    return amp * complex(math.cos(phase), math.sin(phase))


def build_fixture() -> dict:
    """Build the canonical fixture payload deterministically."""
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for ia in range(len(ANTENNAS)):
        for ib in range(ia + 1, len(ANTENNAS)):
            for t in range(NTIME):
                for ch in range(NCHAN):
                    ga = true_gain(ia, t)
                    gb = true_gain(ib, t)
                    sm = source_vis(t, ch)
                    model = sm
                    obs = model + complex(
                        rng.normal(0.0, SIGMA), rng.normal(0.0, SIGMA)
                    )
                    observed = np.conj(ga) * gb * obs
                    key = (ia, ib, t, ch)
                    if key in RFI_CELLS:
                        observed = observed + RFI_CELLS[key]
                    quality = 1 if (key in RFI_CELLS or key in QUALITY_ONLY_BAD) else 0
                    rows.append(
                        {
                            "antenna_a": ANTENNAS[ia],
                            "antenna_b": ANTENNAS[ib],
                            "t_idx": t,
                            "channel": ch,
                            "re": float(observed.real),
                            "im": float(observed.imag),
                            "model_re": float(model.real),
                            "model_im": float(model.imag),
                            "quality": quality,
                        }
                    )

    flags = [
        {
            "source": "manual",
            "reason": "参考天线完全离线（fixture 固定旗标）",
            "antenna_a": "A3",
            "antenna_b": None,
            "t_start": None,
            "t_end": None,
            "ch_start": None,
            "ch_end": None,
        }
    ]
    return {
        "meta": {
            "version": FIXTURE_VERSION,
            "ntime": NTIME,
            "nchan": NCHAN,
            "seed": SEED,
            "sigma": SIGMA,
            "timestamps": timestamps(),
            "frequencies_ghz": frequencies_ghz(),
            "source_phase_slope": SOURCE_PHASE_SLOPE,
            "true_gains": {
                name: {"amp": amp, "phase": phase}
                for name, (amp, phase) in TRUE_GAINS.items()
            },
        },
        "antennas": [
            {"name": name, "label": ANTENNA_NAMES[name]} for name in ANTENNAS
        ],
        "visibilities": rows,
        "flags": flags,
    }


def fixture_path() -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "visibility.json"


def write_fixture_json(path: Path | None = None) -> Path:
    path = path or fixture_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_fixture(), ensure_ascii=False, indent=2))
    return path


def load_fixture() -> dict:
    path = fixture_path()
    if path.exists():
        return json.loads(path.read_text())
    payload = build_fixture()
    write_fixture_json(path)
    return payload
