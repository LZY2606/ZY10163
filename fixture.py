"""固定测试夹具: 四天线小阵列。

包含验收要求的场景:
- A1/A2/A3 构成闭合三角形, 闭相位应接近 0 (点源模型)。
- A4 的所有样本带质量位 (完全被旗标的参考天线候选)。
- A3 的相位随频道线性变化并跨过负实轴 (±π 绕回)。
- A2 在 t>=2 有 +0.5 rad 相位突跳 (供手工旗标)。
- 基线 A2-A3 在 t=1, 频道 4..6 有局部干扰 (供自动建议)。
"""
from __future__ import annotations

import cmath
import random

ANTENNAS = ["A1", "A2", "A3", "A4"]
TIMES = [0.0, 1.0, 2.0, 3.0]
CHANNELS = list(range(8))

QUALITY_BAD = 1  # 质量位: 采样已坏 (如 A4 全程)

_AMP = {"A1": 1.00, "A2": 1.20, "A3": 0.90, "A4": 1.10}
_PHASE = {"A1": 0.0, "A2": 0.30, "A3": -0.20, "A4": 0.10}


def _antenna_phase(ant, t, f):
    ph = _PHASE[ant]
    if ant == "A2" and t >= 2.0:
        ph += 0.5  # 相位突跳
    if ant == "A3":
        ph += 2.6 + 0.4 * f  # 跨频道绕回: f=1 -> 2.8, f=2 -> 3.2 (越过 +π)
    return ph


def _gain(ant, t, f):
    return cmath.rect(_AMP[ant], _antenna_phase(ant, t, f))


def make_samples(order_seed=None):
    """生成样本列表。order_seed 非空时打乱返回顺序 (导入顺序无关性验收)。"""
    rng = random.Random(20260921)
    samples = []
    for i, a1 in enumerate(ANTENNAS):
        for a2 in ANTENNAS[i + 1:]:
            for t in TIMES:
                for f in CHANNELS:
                    vis = _gain(a1, t, f) * _gain(a2, t, f).conjugate()
                    vis *= complex(1 + rng.gauss(0, 0.002), rng.gauss(0, 0.002))
                    quality = 0
                    if a1 == "A4" or a2 == "A4":
                        quality |= QUALITY_BAD  # A4 全程被质量位旗标
                    if {a1, a2} == {"A2", "A3"} and t == 1.0 and 4 <= f <= 6:
                        vis += complex(rng.gauss(0, 3.0), rng.gauss(0, 3.0))  # 局部干扰
                    samples.append({
                        "ant1": a1, "ant2": a2, "time": t, "channel": f,
                        "real": vis.real, "imag": vis.imag, "quality": quality,
                    })
    if order_seed is not None:
        random.Random(order_seed).shuffle(samples)
    return samples
