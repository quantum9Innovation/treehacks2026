from __future__ import annotations

import json
from typing import TypedDict, cast

TrajectoryPoint = tuple[float, float, float, float]


class Data(TypedDict):
    trajectory: list[TrajectoryPoint]
    delay: float


class JSONData(TypedDict):
    trajectory: list[list[float]]
    delay: float


def load(path: str) -> Data:
    with open(path, encoding="utf-8") as f:
        raw = cast(JSONData, json.load(f))

    trajectory: list[TrajectoryPoint] = [
        (float(x), float(y), float(z), float(t)) for x, y, z, t in raw["trajectory"]
    ]

    return {"trajectory": trajectory, "delay": raw["delay"]}
