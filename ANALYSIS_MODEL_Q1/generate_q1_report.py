#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from array import array
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "model_eval_q1_full"
OUT_DIR = ROOT / "ANALYSIS_MODEL_Q1"
FIG_DIR = OUT_DIR / "figures"
REPORT_PATH = OUT_DIR / "REPORT_Q1_MODEL_EVAL.md"
STATS_PATH = OUT_DIR / "report_stats.json"


@dataclass
class DatasetStats:
    station_names: List[str]
    timestamps: List[datetime]
    split_ranges: Dict[str, Tuple[datetime, datetime]]
    split_rows: Dict[str, int]
    station_mean: Dict[str, float]
    station_std: Dict[str, float]
    station_min: Dict[str, float]
    station_max: Dict[str, float]
    station_cv: Dict[str, float]
    daily_totals: List[Tuple[datetime, float]]
    hourly_means: Dict[int, float]
    dow_means: Dict[int, float]
    month_means: Dict[int, float]
    network_mean_15m: float
    network_std_15m: float
    value_zero_pct: float
    value_min: float
    value_max: float
    corr: Dict[str, Dict[str, float]]
    hour_dow_heatmap: List[List[float]]


class SvgCanvas:
    def __init__(self, width: int, height: int, background: str = "#fffdf8") -> None:
        self.width = width
        self.height = height
        self.background = background
        self.elements: List[str] = []

    def add(self, element: str) -> None:
        self.elements.append(element)

    def rect(self, x: float, y: float, w: float, h: float, fill: str, stroke: str = "none", stroke_width: float = 1.0, rx: float = 0.0, opacity: float = 1.0) -> None:
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}" '
            f'stroke="{stroke}" stroke-width="{stroke_width:.2f}" rx="{rx:.2f}" opacity="{opacity:.3f}" />'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#243447", stroke_width: float = 1.5, opacity: float = 1.0, dash: str | None = None) -> None:
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}"{extra} />'
        )

    def polyline(self, points: Sequence[Tuple[float, float]], stroke: str = "#243447", stroke_width: float = 2.0, fill: str = "none", opacity: float = 1.0) -> None:
        path = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        self.add(
            f'<polyline points="{path}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" '
            f'opacity="{opacity:.3f}" stroke-linejoin="round" stroke-linecap="round" />'
        )

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "none", stroke_width: float = 1.0, opacity: float = 1.0) -> None:
        self.add(
            f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" '
            f'stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" />'
        )

    def text(self, x: float, y: float, value: str, size: int = 14, fill: str = "#1f2933", anchor: str = "start", weight: str = "400", rotate: float | None = None) -> None:
        transform = f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size}" font-family="Arial, Helvetica, sans-serif" '
            f'font-weight="{weight}" text-anchor="{anchor}"{transform}>{escape(value)}</text>'
        )

    def path(self, d: str, fill: str = "none", stroke: str = "#243447", stroke_width: float = 1.5, opacity: float = 1.0) -> None:
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width:.2f}" opacity="{opacity:.3f}" '
            'stroke-linejoin="round" stroke-linecap="round" />'
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float, stroke: str = "#425466", stroke_width: float = 2.0) -> None:
        self.line(x1, y1, x2, y2, stroke=stroke, stroke_width=stroke_width)
        ang = math.atan2(y2 - y1, x2 - x1)
        size = 8.0
        left = (x2 - size * math.cos(ang - math.pi / 6), y2 - size * math.sin(ang - math.pi / 6))
        right = (x2 - size * math.cos(ang + math.pi / 6), y2 - size * math.sin(ang + math.pi / 6))
        self.polyline([left, (x2, y2), right], stroke=stroke, stroke_width=stroke_width, fill="none")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">',
            f'<rect x="0" y="0" width="{self.width}" height="{self.height}" fill="{self.background}" />',
            *self.elements,
            "</svg>",
        ]
        path.write_text("\n".join(content), encoding="utf-8")


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % rgb


def blend(c1: str, c2: str, t: float) -> str:
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    return rgb_to_hex(
        (
            int(round(lerp(r1, r2, t))),
            int(round(lerp(g1, g2, t))),
            int(round(lerp(b1, b2, t))),
        )
    )


def fmt_num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def rolling_mean(values: Sequence[float], window: int) -> List[float]:
    out: List[float] = []
    acc = 0.0
    for i, val in enumerate(values):
        acc += val
        if i >= window:
            acc -= values[i - window]
        denom = min(i + 1, window)
        out.append(acc / denom)
    return out


def axis_ticks(vmin: float, vmax: float, steps: int = 5) -> List[float]:
    if vmax <= vmin:
        return [vmin]
    span = vmax - vmin
    raw = span / max(1, steps)
    mag = 10 ** math.floor(math.log10(raw))
    norm = raw / mag
    if norm < 1.5:
        step = 1 * mag
    elif norm < 3:
        step = 2 * mag
    elif norm < 7:
        step = 5 * mag
    else:
        step = 10 * mag
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks = []
    current = start
    safety = 0
    while current <= end + 1e-9 and safety < 100:
        ticks.append(round(current, 10))
        current += step
        safety += 1
    return ticks


def map_x(index: int, n: int, left: float, width: float) -> float:
    if n <= 1:
        return left
    return left + width * index / (n - 1)


def map_y(value: float, vmin: float, vmax: float, top: float, height: float) -> float:
    if vmax <= vmin:
        return top + height / 2
    return top + height - ((value - vmin) / (vmax - vmin)) * height


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_split_csv(path: Path) -> Tuple[List[datetime], List[str], List[array]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        station_names = header[1:]
        timestamps: List[datetime] = []
        series = [array("f") for _ in station_names]
        for row in reader:
            timestamps.append(datetime.fromisoformat(row[0]))
            for i, value in enumerate(row[1:]):
                series[i].append(float(value))
    return timestamps, station_names, series


def compute_dataset_stats() -> DatasetStats:
    split_paths = {
        "train": DATA_DIR / "datasets" / "dataset_train.csv",
        "val": DATA_DIR / "datasets" / "dataset_val.csv",
        "test": DATA_DIR / "datasets" / "dataset_test.csv",
    }
    full_path = DATA_DIR / "datasets" / "dataset_full.csv"

    split_ranges: Dict[str, Tuple[datetime, datetime]] = {}
    split_rows: Dict[str, int] = {}
    for name, path in split_paths.items():
        timestamps, station_names, _ = read_split_csv(path)
        split_rows[name] = len(timestamps)
        split_ranges[name] = (timestamps[0], timestamps[-1])

    with full_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        station_names = header[1:]
        n = len(station_names)

        timestamps: List[datetime] = []
        sums = [0.0] * n
        sums2 = [0.0] * n
        mins = [float("inf")] * n
        maxs = [float("-inf")] * n
        cross = [[0.0] * n for _ in range(n)]

        daily = defaultdict(float)
        hourly_sum = defaultdict(float)
        hourly_count = defaultdict(int)
        dow_sum = defaultdict(float)
        dow_count = defaultdict(int)
        month_sum = defaultdict(float)
        month_count = defaultdict(int)
        heat_sum = [[0.0 for _ in range(24)] for _ in range(7)]
        heat_count = [[0 for _ in range(24)] for _ in range(7)]

        total_obs = 0
        total_sum = 0.0
        total_sum2 = 0.0
        zero_count = 0
        global_min = float("inf")
        global_max = float("-inf")

        for row in reader:
            ts = datetime.fromisoformat(row[0])
            values = [float(v) for v in row[1:]]
            timestamps.append(ts)

            total = sum(values)
            daily[ts.date()] += total
            hourly_sum[ts.hour] += total
            hourly_count[ts.hour] += 1
            dow_sum[ts.weekday()] += total
            dow_count[ts.weekday()] += 1
            month_sum[ts.month] += total
            month_count[ts.month] += 1
            heat_sum[ts.weekday()][ts.hour] += total
            heat_count[ts.weekday()][ts.hour] += 1

            total_sum += total
            total_sum2 += total * total

            for i, val in enumerate(values):
                sums[i] += val
                sums2[i] += val * val
                mins[i] = min(mins[i], val)
                maxs[i] = max(maxs[i], val)
                zero_count += 1 if val == 0.0 else 0
                global_min = min(global_min, val)
                global_max = max(global_max, val)
                total_obs += 1

            for i in range(n):
                vi = values[i]
                for j in range(i, n):
                    cross[i][j] += vi * values[j]

        for i in range(n):
            for j in range(i):
                cross[i][j] = cross[j][i]

    rows = len(timestamps)
    station_mean = {}
    station_std = {}
    station_min = {}
    station_max = {}
    station_cv = {}
    corr: Dict[str, Dict[str, float]] = {}
    for i, name in enumerate(station_names):
        mean = sums[i] / rows
        var = max(sums2[i] / rows - mean * mean, 0.0)
        std = math.sqrt(var)
        station_mean[name] = mean
        station_std[name] = std
        station_min[name] = mins[i]
        station_max[name] = maxs[i]
        station_cv[name] = std / mean if mean else 0.0

    for i, name_i in enumerate(station_names):
        corr[name_i] = {}
        for j, name_j in enumerate(station_names):
            mean_i = station_mean[name_i]
            mean_j = station_mean[name_j]
            std_i = station_std[name_i]
            std_j = station_std[name_j]
            cov = cross[i][j] / rows - mean_i * mean_j
            denom = max(std_i * std_j, 1e-12)
            corr[name_i][name_j] = max(min(cov / denom, 1.0), -1.0)

    daily_totals = [(datetime.combine(day, datetime.min.time()), total) for day, total in sorted(daily.items())]
    hourly_means = {k: hourly_sum[k] / hourly_count[k] for k in sorted(hourly_sum)}
    dow_means = {k: dow_sum[k] / dow_count[k] for k in sorted(dow_sum)}
    month_means = {k: month_sum[k] / month_count[k] for k in sorted(month_sum)}
    heatmap = [
        [
            heat_sum[dow][hour] / heat_count[dow][hour] if heat_count[dow][hour] else 0.0
            for hour in range(24)
        ]
        for dow in range(7)
    ]

    network_mean_15m = total_sum / rows
    network_var_15m = max(total_sum2 / rows - network_mean_15m * network_mean_15m, 0.0)

    return DatasetStats(
        station_names=station_names,
        timestamps=timestamps,
        split_ranges=split_ranges,
        split_rows=split_rows,
        station_mean=station_mean,
        station_std=station_std,
        station_min=station_min,
        station_max=station_max,
        station_cv=station_cv,
        daily_totals=daily_totals,
        hourly_means=hourly_means,
        dow_means=dow_means,
        month_means=month_means,
        network_mean_15m=network_mean_15m,
        network_std_15m=math.sqrt(network_var_15m),
        value_zero_pct=100.0 * zero_count / max(total_obs, 1),
        value_min=global_min,
        value_max=global_max,
        corr=corr,
        hour_dow_heatmap=heatmap,
    )


def build_architecture_figure(path: Path) -> None:
    svg = SvgCanvas(1400, 720, background="#fffdf8")
    svg.text(70, 70, "DTS-GSSF Experimental Flow", size=28, weight="700")
    svg.text(70, 102, "Paper-oriented summary of the evaluation pipeline used by model_eval_q1_full", size=14, fill="#52606d")

    boxes = [
        (70, 160, 250, 150, "#e8f1fb", "#1f5f8b", "1. Synthetic Astana-Like Data", ["28 stations, 9 lines, 4 districts", "15-min counts, 50,016 timestamps", "rush-hour, weather, events, disruptions, drift"]),
        (380, 160, 250, 150, "#eef8ef", "#2f6f3e", "2. Feature Construction", ["lag-1, lag-2, lag-4 counts", "calendar encodings", "weather + event/disruption/drift flags"]),
        (690, 160, 250, 150, "#fff4e6", "#a65b00", "3. Forecasting Models", ["DTS-GSSF backbone", "DCRNN, Transformer, LSTM, GRU", "Historical Avg, Seasonal Naive"]),
        (1000, 160, 250, 150, "#f3ecff", "#6b46c1", "4. Evaluation Outputs", ["validation/test metrics", "component ablations", "online correction study"]),
    ]
    for x, y, w, h, fill, stroke, title, lines in boxes:
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, stroke_width=2.0, rx=18)
        svg.text(x + 18, y + 36, title, size=20, weight="700", fill=stroke)
        for i, line in enumerate(lines):
            svg.text(x + 18, y + 72 + i * 26, line, size=14, fill="#243447")
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + boxes[i][2]
        y1 = boxes[i][1] + boxes[i][3] / 2
        x2 = boxes[i + 1][0]
        y2 = boxes[i + 1][1] + boxes[i + 1][3] / 2
        svg.arrow(x1 + 12, y1, x2 - 12, y2, stroke="#425466", stroke_width=2.5)

    lower = [
        (110, 420, 290, 180, "#fff7d6", "#8d6e00", "Offline DTS-GSSF", ["graph-structured state-space backbone", "adaptive adjacency + LoRA forecast heads", "bottom-level station forecasting, H = 12"]),
        (450, 420, 290, 180, "#dff7f2", "#0b7285", "Controlled Ablations", ["No LoRA", "No Adaptive Adjacency", "No Graph Structure"]),
        (790, 420, 290, 180, "#fde8e8", "#c53030", "Online Residual Study", ["base forecast", "Kalman correction", "reconciliation, with/without adaptation"]),
    ]
    for x, y, w, h, fill, stroke, title, lines in lower:
        svg.rect(x, y, w, h, fill=fill, stroke=stroke, stroke_width=2.0, rx=18)
        svg.text(x + 18, y + 36, title, size=20, weight="700", fill=stroke)
        for i, line in enumerate(lines):
            svg.text(x + 18, y + 74 + i * 28, line, size=14, fill="#243447")
    svg.arrow(215, 310, 255, 420, stroke="#425466", stroke_width=2.0)
    svg.arrow(810, 310, 595, 420, stroke="#425466", stroke_width=2.0)
    svg.arrow(1125, 310, 935, 420, stroke="#425466", stroke_width=2.0)
    svg.text(70, 660, "Note: the current artifacts support a strong methodological report, but the benchmark remains synthetic and should be framed as such in any submission.", size=14, fill="#7b341e")
    svg.save(path)


def build_dataset_timeline_figure(path: Path, stats: DatasetStats) -> None:
    svg = SvgCanvas(1500, 760, background="#fffdf8")
    svg.text(70, 70, "Dataset Timeline and Train/Validation/Test Partition", size=28, weight="700")
    svg.text(70, 102, "Daily aggregated network demand with 7-day moving average and exact split boundaries from the exported CSV files", size=14, fill="#52606d")

    left, top, width, height = 90, 150, 1300, 470
    values = [v for _, v in stats.daily_totals]
    smooth = rolling_mean(values, 7)
    ymin = min(values)
    ymax = max(values)
    ticks = axis_ticks(ymin, ymax, 5)

    date_to_index = {dt.date(): i for i, (dt, _) in enumerate(stats.daily_totals)}
    split_colors = {"train": "#cfe8ff", "val": "#ffe5b5", "test": "#ffd4cf"}
    split_labels = {"train": "Train", "val": "Validation", "test": "Test"}
    for name in ("train", "val", "test"):
        start, end = stats.split_ranges[name]
        x0 = map_x(date_to_index[start.date()], len(stats.daily_totals), left, width)
        x1 = map_x(date_to_index[end.date()], len(stats.daily_totals), left, width)
        svg.rect(x0, top, max(x1 - x0, 6.0), height, fill=split_colors[name], opacity=0.55)

    for tick in ticks:
        y = map_y(tick, ticks[0], ticks[-1], top, height)
        svg.line(left, y, left + width, y, stroke="#d9e2ec", stroke_width=1.0)
        svg.text(left - 14, y + 5, f"{int(round(tick)):,}", size=12, fill="#52606d", anchor="end")

    daily_points = [(map_x(i, len(values), left, width), map_y(val, ticks[0], ticks[-1], top, height)) for i, val in enumerate(values)]
    smooth_points = [(map_x(i, len(smooth), left, width), map_y(val, ticks[0], ticks[-1], top, height)) for i, val in enumerate(smooth)]
    svg.polyline(daily_points, stroke="#7c8ea3", stroke_width=1.25, opacity=0.55)
    svg.polyline(smooth_points, stroke="#c2410c", stroke_width=3.0, opacity=0.95)
    svg.line(left, top + height, left + width, top + height, stroke="#243447", stroke_width=1.5)
    svg.line(left, top, left, top + height, stroke="#243447", stroke_width=1.5)

    x_labels = [stats.daily_totals[0][0], stats.daily_totals[len(stats.daily_totals) // 3][0], stats.daily_totals[2 * len(stats.daily_totals) // 3][0], stats.daily_totals[-1][0]]
    for dt in x_labels:
        idx = date_to_index[dt.date()]
        x = map_x(idx, len(stats.daily_totals), left, width)
        svg.line(x, top + height, x, top + height + 6, stroke="#243447", stroke_width=1.5)
        svg.text(x, top + height + 28, dt.strftime("%Y-%m"), size=12, anchor="middle", fill="#52606d")

    legend_y = 660
    items = [("#7c8ea3", "Daily total demand"), ("#c2410c", "7-day moving average"), ("#cfe8ff", "Train"), ("#ffe5b5", "Validation"), ("#ffd4cf", "Test")]
    x = 90
    for color, label in items:
        svg.rect(x, legend_y - 14, 20, 12, fill=color)
        svg.text(x + 28, legend_y - 2, label, size=13, fill="#243447")
        x += 190

    svg.text(90, 705, f"Range: {stats.timestamps[0]} to {stats.timestamps[-1]} | Mean network load per 15 minutes: {fmt_num(stats.network_mean_15m, 2)} | Zero-rate: {fmt_num(stats.value_zero_pct, 2)}%", size=13, fill="#52606d")
    svg.save(path)


def build_temporal_heatmap_figure(path: Path, stats: DatasetStats) -> None:
    svg = SvgCanvas(1260, 760, background="#fffdf8")
    svg.text(70, 70, "Average Network Demand by Day of Week and Hour of Day", size=28, weight="700")
    svg.text(70, 102, "Heatmap computed from the station-count CSVs after summing across all 28 stations", size=14, fill="#52606d")

    left, top = 150, 150
    cell_w, cell_h = 40, 62
    flat = [v for row in stats.hour_dow_heatmap for v in row]
    vmin, vmax = min(flat), max(flat)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    for dow in range(7):
        svg.text(left - 18, top + dow * cell_h + cell_h / 2 + 5, days[dow], size=14, anchor="end", fill="#243447", weight="700")
        for hour in range(24):
            value = stats.hour_dow_heatmap[dow][hour]
            t = 0.0 if vmax <= vmin else (value - vmin) / (vmax - vmin)
            color = blend("#fef3c7", "#0f766e", t)
            x = left + hour * cell_w
            y = top + dow * cell_h
            svg.rect(x, y, cell_w - 2, cell_h - 2, fill=color, rx=4)
            if hour % 3 == 0:
                svg.text(x + cell_w / 2, top - 14, f"{hour:02d}", size=12, anchor="middle", fill="#52606d")
            svg.text(x + cell_w / 2, y + cell_h / 2 + 4, f"{int(round(value))}", size=11, anchor="middle", fill="#102a43", weight="700")

    svg.text(150, 630, "Peak periods occur during the weekday evening commute, with sustained weekday loads above the weekend baseline.", size=14, fill="#243447")
    svg.text(150, 660, f"Weekday mean: {fmt_num(sum(stats.dow_means[d] for d in range(5)) / 5, 1)} | Weekend mean: {fmt_num(sum(stats.dow_means[d] for d in (5, 6)) / 2, 1)} | Peak hour: 18:00", size=13, fill="#52606d")

    for i in range(6):
        color = blend("#fef3c7", "#0f766e", i / 5)
        svg.rect(780 + i * 65, 648, 48, 14, fill=color, rx=3)
    svg.text(765, 661, f"{int(round(vmin))}", size=12, anchor="end", fill="#52606d")
    svg.text(1115, 661, f"{int(round(vmax))}", size=12, fill="#52606d")
    svg.text(930, 685, "Average network passengers per 15-minute interval", size=12, anchor="middle", fill="#52606d")
    svg.save(path)


def build_station_load_figure(path: Path, stats: DatasetStats) -> None:
    svg = SvgCanvas(1560, 820, background="#fffdf8")
    svg.text(70, 70, "Station-Level Mean Demand Ranking", size=28, weight="700")
    svg.text(70, 102, "Average passenger counts per station, sorted descending, with the highest-load stations highlighted", size=14, fill="#52606d")

    ranked = sorted(stats.station_names, key=lambda name: stats.station_mean[name], reverse=True)
    values = [stats.station_mean[name] for name in ranked]
    left, top, width, height = 90, 150, 1380, 500
    ymax = max(values) * 1.12
    ticks = axis_ticks(0.0, ymax, 5)
    bar_w = width / len(values)

    for tick in ticks:
        y = map_y(tick, ticks[0], ticks[-1], top, height)
        svg.line(left, y, left + width, y, stroke="#d9e2ec", stroke_width=1.0)
        svg.text(left - 12, y + 5, f"{tick:.0f}", size=12, anchor="end", fill="#52606d")

    top_set = set(ranked[:5])
    for i, name in enumerate(ranked):
        x = left + i * bar_w + 4
        y = map_y(values[i], ticks[0], ticks[-1], top, height)
        h = top + height - y
        color = "#c2410c" if name in top_set else "#5b7c99"
        svg.rect(x, y, max(bar_w - 8, 6), h, fill=color, rx=4, opacity=0.92)
        svg.text(x + bar_w / 2 - 2, top + height + 18, name.replace("Station | ", ""), size=10, anchor="end", fill="#243447", rotate=-55)

    svg.line(left, top + height, left + width, top + height, stroke="#243447", stroke_width=1.5)
    svg.line(left, top, left, top + height, stroke="#243447", stroke_width=1.5)

    top_five = ", ".join(f"{name.replace('Station | ', '')} ({fmt_num(stats.station_mean[name], 1)})" for name in ranked[:5])
    bottom_five = ", ".join(f"{name.replace('Station | ', '')} ({fmt_num(stats.station_mean[name], 1)})" for name in ranked[-5:])
    svg.text(90, 715, f"Top 5 mean-load stations: {top_five}", size=13, fill="#243447")
    svg.text(90, 745, f"Bottom 5 mean-load stations: {bottom_five}", size=13, fill="#52606d")
    svg.save(path)


def build_model_comparison_figure(path: Path, results: Dict[str, object]) -> None:
    svg = SvgCanvas(1560, 930, background="#fffdf8")
    svg.text(70, 70, "Main Benchmark Comparison on the Test Set", size=28, weight="700")
    svg.text(70, 102, "Small multiples for the flattened 12-step test evaluation; DTS-GSSF is highlighted", size=14, fill="#52606d")

    order = sorted(results.keys(), key=lambda name: results[name]["test_full"]["mae"])
    metrics = [
        ("mae", "MAE", False),
        ("rmse", "RMSE", False),
        ("wape", "WAPE", False),
        ("r2", "R^2", True),
    ]
    panel_positions = [
        (80, 150), (800, 150),
        (80, 520), (800, 520),
    ]
    panel_w, panel_h = 620, 260
    for (metric_key, title, higher_is_better), (left, top) in zip(metrics, panel_positions):
        values = [results[name]["test_full"][metric_key] for name in order]
        vmin = min(values)
        vmax = max(values)
        pad = (vmax - vmin) * 0.12 if vmax > vmin else 1.0
        if metric_key == "r2":
            axis_min = max(0.0, vmin - pad)
            axis_max = min(1.0, vmax + pad)
        else:
            axis_min = 0.0
            axis_max = vmax + pad
        ticks = axis_ticks(axis_min, axis_max, 4)
        svg.text(left, top - 20, title, size=20, weight="700", fill="#243447")
        for tick in ticks:
            y = map_y(tick, ticks[0], ticks[-1], top, panel_h)
            svg.line(left, y, left + panel_w, y, stroke="#d9e2ec", stroke_width=1.0)
            svg.text(left - 10, y + 5, f"{tick:.2f}", size=11, anchor="end", fill="#52606d")
        bar_w = panel_w / len(order)
        for i, name in enumerate(order):
            x = left + i * bar_w + 6
            value = results[name]["test_full"][metric_key]
            y = map_y(value, ticks[0], ticks[-1], top, panel_h)
            h = top + panel_h - y
            color = "#c2410c" if name == "DTS-GSSF" else "#66788a"
            svg.rect(x, y, max(bar_w - 12, 10), h, fill=color, rx=4, opacity=0.93)
            svg.text(x + bar_w / 2 - 2, top + panel_h + 16, name, size=10, anchor="end", fill="#243447", rotate=-50)
        svg.line(left, top + panel_h, left + panel_w, top + panel_h, stroke="#243447", stroke_width=1.4)
        svg.line(left, top, left, top + panel_h, stroke="#243447", stroke_width=1.4)
        note = "higher is better" if higher_is_better else "lower is better"
        svg.text(left + panel_w - 4, top - 20, note, size=11, anchor="end", fill="#52606d")
    svg.save(path)


def build_efficiency_frontier_figure(path: Path, results: Dict[str, object]) -> None:
    svg = SvgCanvas(1360, 820, background="#fffdf8")
    svg.text(70, 70, "Accuracy–Efficiency Frontier", size=28, weight="700")
    svg.text(70, 102, "Test-set MAE versus training time (log-scaled x-axis). Lower and further left is better.", size=14, fill="#52606d")

    left, top, width, height = 110, 150, 1140, 520
    items = []
    for name, payload in results.items():
        train_time = float(payload["train_time_sec"])
        train_time = max(train_time, 0.003)
        items.append((name, train_time, float(payload["test_full"]["mae"])))
    xs = [math.log10(v[1]) for v in items]
    ys = [v[2] for v in items]
    x_ticks_raw = [0.01, 0.1, 1, 10, 100, 1000, 10000]
    x_ticks = [t for t in x_ticks_raw if min(x for _, x, _ in items) <= t <= max(x for _, x, _ in items) * 1.15]
    if not x_ticks:
        x_ticks = [min(x for _, x, _ in items), max(x for _, x, _ in items)]
    y_ticks = axis_ticks(min(ys) - 0.2, max(ys) + 0.5, 5)
    xmin = min(math.log10(t) for t in x_ticks)
    xmax = max(math.log10(t) for t in x_ticks)
    ymin, ymax = y_ticks[0], y_ticks[-1]

    for tick in x_ticks:
        x = left + (math.log10(tick) - xmin) / max(xmax - xmin, 1e-9) * width
        svg.line(x, top, x, top + height, stroke="#e4e7eb", stroke_width=1.0)
        svg.text(x, top + height + 28, f"{tick:g}s", size=12, anchor="middle", fill="#52606d")
    for tick in y_ticks:
        y = map_y(tick, ymin, ymax, top, height)
        svg.line(left, y, left + width, y, stroke="#d9e2ec", stroke_width=1.0)
        svg.text(left - 12, y + 5, f"{tick:.2f}", size=12, anchor="end", fill="#52606d")

    for name, train_time, mae in items:
        x = left + (math.log10(train_time) - xmin) / max(xmax - xmin, 1e-9) * width
        y = map_y(mae, ymin, ymax, top, height)
        color = "#c2410c" if name == "DTS-GSSF" else "#4c6f8c"
        radius = 10 if name == "DTS-GSSF" else 8
        svg.circle(x, y, radius, fill=color, stroke="#102a43", stroke_width=1.2, opacity=0.95)
        svg.text(x + 12, y - 10, name, size=12, fill="#243447", weight="700" if name == "DTS-GSSF" else "400")

    svg.line(left, top + height, left + width, top + height, stroke="#243447", stroke_width=1.5)
    svg.line(left, top, left, top + height, stroke="#243447", stroke_width=1.5)
    svg.text(110, 720, "DTS-GSSF occupies the best-MAE point, but only by a narrow margin and with substantially higher training cost than the simpler baselines.", size=13, fill="#243447")
    svg.save(path)


def build_ablation_figure(path: Path, full_metrics: Dict[str, float], ablations: Dict[str, object]) -> None:
    svg = SvgCanvas(1360, 820, background="#fffdf8")
    svg.text(70, 70, "Component Ablation on Test-Full Metrics", size=28, weight="700")
    svg.text(70, 102, "The graph module is indispensable; adaptive adjacency is beneficial; LoRA has only a marginal effect in the current run.", size=14, fill="#52606d")

    names = ["DTS-GSSF", "No LoRA", "No Adaptive Adj", "No Graph"]
    metric_key = "mae"
    values = [full_metrics[metric_key], ablations["No LoRA"]["test_full"][metric_key], ablations["No Adaptive Adj"]["test_full"][metric_key], ablations["No Graph"]["test_full"][metric_key]]
    left, top, width, height = 110, 170, 510, 470
    ticks = axis_ticks(0.0, max(values) * 1.12, 5)
    for tick in ticks:
        y = map_y(tick, ticks[0], ticks[-1], top, height)
        svg.line(left, y, left + width, y, stroke="#d9e2ec", stroke_width=1.0)
        svg.text(left - 12, y + 5, f"{tick:.1f}", size=12, anchor="end", fill="#52606d")
    bar_w = width / len(names)
    colors = ["#c2410c", "#5b7c99", "#5b7c99", "#5b7c99"]
    for i, (name, value, color) in enumerate(zip(names, values, colors)):
        x = left + i * bar_w + 10
        y = map_y(value, ticks[0], ticks[-1], top, height)
        svg.rect(x, y, bar_w - 20, top + height - y, fill=color, rx=5, opacity=0.94)
        svg.text(x + (bar_w - 20) / 2, y - 10, f"{value:.3f}", size=12, anchor="middle", fill="#243447", weight="700")
        svg.text(x + (bar_w - 20) / 2, top + height + 26, name, size=12, anchor="middle", fill="#243447")
    svg.line(left, top + height, left + width, top + height, stroke="#243447", stroke_width=1.5)
    svg.line(left, top, left, top + height, stroke="#243447", stroke_width=1.5)
    svg.text(left, top - 18, "MAE", size=18, weight="700", fill="#243447")

    right_left, right_top = 760, 180
    rows = [
        ("No LoRA", ablations["No LoRA"]["test_full"]),
        ("No Adaptive Adj", ablations["No Adaptive Adj"]["test_full"]),
        ("No Graph", ablations["No Graph"]["test_full"]),
    ]
    svg.text(right_left, right_top - 20, "Relative Change vs Full Model", size=20, weight="700", fill="#243447")
    svg.rect(right_left, right_top, 510, 370, fill="#ffffff", stroke="#d9e2ec", stroke_width=1.5, rx=12)
    headers = ["Variant", "Delta MAE", "WAPE", "R^2"]
    cols_x = [right_left + 18, right_left + 220, right_left + 350, right_left + 440]
    for x, header in zip(cols_x, headers):
        svg.text(x, right_top + 34, header, size=14, weight="700", fill="#243447")
    y = right_top + 80
    for name, payload in rows:
        delta = 100.0 * (payload["mae"] - full_metrics["mae"]) / full_metrics["mae"]
        svg.text(cols_x[0], y, name, size=14, fill="#243447")
        svg.text(cols_x[1], y, f"+{delta:.2f}%", size=14, fill="#c2410c" if delta > 1.0 else "#52606d", weight="700")
        svg.text(cols_x[2], y, f"{payload['wape']:.3f}", size=14, fill="#243447")
        svg.text(cols_x[3], y, f"{payload['r2']:.3f}", size=14, fill="#243447")
        y += 48
    svg.text(right_left, 610, "Interpretation: removing graph propagation nearly doubles MAE, while removing adaptive adjacency causes a modest but consistent degradation.", size=13, fill="#243447")
    svg.save(path)


def build_online_ablation_figure(path: Path, online: Dict[str, object]) -> None:
    svg = SvgCanvas(1300, 760, background="#fffdf8")
    svg.text(70, 70, "Online Residual Correction Study", size=28, weight="700")
    svg.text(70, 102, "Observed online MAE values from the released evaluation artifacts", size=14, fill="#52606d")

    order = ["base", "kalman", "kalman_recon", "no_adapt_kalman", "no_adapt_kalman_recon"]
    labels = {
        "base": "Base",
        "kalman": "Kalman",
        "kalman_recon": "Kalman + Reconciliation",
        "no_adapt_kalman": "No-Adapt Kalman",
        "no_adapt_kalman_recon": "No-Adapt Kalman + Reconciliation",
    }
    values = [online[key]["mae"] for key in order]
    left, top, width, height = 110, 170, 1080, 440
    ticks = axis_ticks(min(values) - 0.05, max(values) + 0.1, 5)
    for tick in ticks:
        y = map_y(tick, ticks[0], ticks[-1], top, height)
        svg.line(left, y, left + width, y, stroke="#d9e2ec", stroke_width=1.0)
        svg.text(left - 12, y + 5, f"{tick:.2f}", size=12, anchor="end", fill="#52606d")
    bar_w = width / len(order)
    base = online["base"]["mae"]
    for i, key in enumerate(order):
        value = online[key]["mae"]
        x = left + i * bar_w + 16
        y = map_y(value, ticks[0], ticks[-1], top, height)
        color = "#c2410c" if key == "base" else "#5b7c99"
        svg.rect(x, y, bar_w - 32, top + height - y, fill=color, rx=5, opacity=0.94)
        delta = 100.0 * (value - base) / base
        svg.text(x + (bar_w - 32) / 2, y - 10, f"{value:.3f}", size=12, anchor="middle", fill="#243447", weight="700")
        svg.text(x + (bar_w - 32) / 2, top + height + 20, labels[key], size=11, anchor="middle", fill="#243447")
        if key != "base":
            svg.text(x + (bar_w - 32) / 2, y - 28, f"{delta:+.2f}%", size=11, anchor="middle", fill="#7b341e")
    svg.line(left, top + height, left + width, top + height, stroke="#243447", stroke_width=1.5)
    svg.line(left, top, left, top + height, stroke="#243447", stroke_width=1.5)
    svg.text(110, 680, "The online corrector variants do not improve on the raw backbone in this release, although adaptation reduces the penalty relative to the no-adaptation settings.", size=13, fill="#243447")
    svg.save(path)


def table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([head, sep, *body])


def write_report(stats: DatasetStats, run_state: Dict[str, object], ablations: Dict[str, object], online: Dict[str, object]) -> None:
    results = run_state["results"]
    dts = results["DTS-GSSF"]
    ranked = sorted(results.keys(), key=lambda name: results[name]["test_full"]["mae"])
    best_station_rank = sorted(stats.station_names, key=lambda name: stats.station_mean[name], reverse=True)
    top5 = [(name.replace("Station | ", ""), stats.station_mean[name]) for name in best_station_rank[:5]]
    bottom5 = [(name.replace("Station | ", ""), stats.station_mean[name]) for name in best_station_rank[-5:]]

    lookback = 48
    horizon = 12
    freq_min = 15
    train_rows = stats.split_rows["train"]
    val_rows = stats.split_rows["val"]
    test_rows = stats.split_rows["test"]
    total_rows = train_rows + val_rows + test_rows
    window_rows = [
        ("Train", train_rows, max(train_rows - lookback - horizon, 0)),
        ("Validation", val_rows, max(val_rows - lookback - horizon, 0)),
        ("Test", test_rows, max(test_rows - lookback - horizon, 0)),
    ]

    split_table = table(
        ["Split", "Rows", "Date Range", "Approx. Days", "Windows"],
        [
            [
                name,
                f"{rows:,}",
                f"{stats.split_ranges[key][0].strftime('%Y-%m-%d %H:%M')} to {stats.split_ranges[key][1].strftime('%Y-%m-%d %H:%M')}",
                f"{rows / 96:.2f}",
                f"{windows:,}",
            ]
            for (name, rows, windows), key in zip(window_rows, ["train", "val", "test"])
        ],
    )

    feature_table = table(
        ["Feature Block", "Count", "Description"],
        [
            ["Lagged passenger counts", "3", "Lag-1, lag-2, lag-4 station counts"],
            ["Calendar encodings", "5", "hour-of-day sine/cosine, day-of-week sine/cosine, weekend flag"],
            ["Weather channels", "3", "synthetic temperature, precipitation flag, wind"],
            ["Operational flags", "3", "event, disruption, drift indicators"],
            ["Total input dimensions", "14", "per station, per 15-minute timestep"],
        ],
    )

    main_table = table(
        ["Rank", "Model", "MAE", "RMSE", "WAPE", "sMAPE", "R^2", "Train Time (s)"],
        [
            [
                str(i + 1),
                name,
                fmt_num(results[name]["test_full"]["mae"], 4),
                fmt_num(results[name]["test_full"]["rmse"], 4),
                fmt_num(results[name]["test_full"]["wape"], 4),
                fmt_num(results[name]["test_full"]["smape"], 4),
                fmt_num(results[name]["test_full"]["r2"], 4),
                fmt_num(results[name]["train_time_sec"], 3),
            ]
            for i, name in enumerate(ranked)
        ],
    )

    ablation_table = table(
        ["Variant", "MAE", "Delta vs Full", "RMSE", "WAPE", "R^2"],
        [
            [
                "DTS-GSSF (Full)",
                fmt_num(dts["test_full"]["mae"], 4),
                "0.00%",
                fmt_num(dts["test_full"]["rmse"], 4),
                fmt_num(dts["test_full"]["wape"], 4),
                fmt_num(dts["test_full"]["r2"], 4),
            ],
            *[
                [
                    name,
                    fmt_num(ablations[name]["test_full"]["mae"], 4),
                    f"+{100.0 * (ablations[name]['test_full']['mae'] - dts['test_full']['mae']) / dts['test_full']['mae']:.2f}%",
                    fmt_num(ablations[name]["test_full"]["rmse"], 4),
                    fmt_num(ablations[name]["test_full"]["wape"], 4),
                    fmt_num(ablations[name]["test_full"]["r2"], 4),
                ]
                for name in ["No LoRA", "No Adaptive Adj", "No Graph"]
            ],
        ],
    )

    online_table = table(
        ["Variant", "MAE", "Delta vs Base"],
        [
            [label, fmt_num(online[key]["mae"], 4), f"{100.0 * (online[key]['mae'] - online['base']['mae']) / online['base']['mae']:+.2f}%"]
            for key, label in [
                ("base", "Base"),
                ("kalman", "Kalman"),
                ("kalman_recon", "Kalman + Reconciliation"),
                ("no_adapt_kalman", "No-Adapt Kalman"),
                ("no_adapt_kalman_recon", "No-Adapt Kalman + Reconciliation"),
            ]
        ],
    )

    dts_mae = dts["test_full"]["mae"]
    transformer_gain = 100.0 * (results["Transformer"]["test_full"]["mae"] - dts_mae) / results["Transformer"]["test_full"]["mae"]
    dcrnn_gain = 100.0 * (results["DCRNN"]["test_full"]["mae"] - dts_mae) / results["DCRNN"]["test_full"]["mae"]
    seasonal_gain = 100.0 * (results["Seasonal Naive"]["test_full"]["mae"] - dts_mae) / results["Seasonal Naive"]["test_full"]["mae"]
    weekday_mean = sum(stats.dow_means[d] for d in range(5)) / 5
    weekend_mean = sum(stats.dow_means[d] for d in (5, 6)) / 2
    daily_network = stats.network_mean_15m * 96

    report = f"""# DTS-GSSF Q1-Style Evaluation Report

## Title
**DTS-GSSF for Multivariate Passenger Flow Forecasting on the `model_eval_q1_full` Astana-Like Synthetic Benchmark**

## Abstract
This report consolidates the full experimental evidence contained in [`model_eval_q1_full`](../model_eval_q1_full), including dataset exports, benchmark metrics, component ablations, and online correction results. The benchmark contains {total_rows:,} 15-minute observations across 28 stations, spanning {fmt_num((stats.timestamps[-1] - stats.timestamps[0]).total_seconds() / 86400, 2)} days. Using the released test metrics, DTS-GSSF achieves the best overall **flattened-horizon MAE** ({fmt_num(dts['test_full']['mae'], 4)}) and **WAPE** ({fmt_num(dts['test_full']['wape'], 4)}), outperforming Transformer by {fmt_num(transformer_gain, 2)}% on MAE, DCRNN by {fmt_num(dcrnn_gain, 2)}%, and Seasonal Naive by {fmt_num(seasonal_gain, 2)}%. However, the margin over strong deep baselines is narrow, and the method is materially more expensive to train. Component ablations show that **graph structure is the critical contributor** (+94.38% MAE without graph propagation), while adaptive adjacency yields moderate gains and LoRA contributes only marginally in the current run. The online residual-correction variants do not improve upon the raw backbone in the provided artifacts, indicating that the streaming adaptation stack remains an open optimization target. Because the dataset is generated by the repository’s seeded simulator rather than raw operational AFC logs, the present evidence is best framed as a controlled synthetic benchmark study rather than a fully operational deployment paper.

## Keywords
Passenger flow forecasting, spatio-temporal learning, graph state-space models, adaptive adjacency, LoRA, ablation study, synthetic transit benchmark

## 1. Executive Summary
- The benchmark is **methodologically rich but synthetic**. The repository code shows that `model_evaluation.py` generates the dataset via `main.py` before training and evaluation.
- DTS-GSSF is the best model on **test-full MAE** and **WAPE**, but not on every metric. DCRNN is slightly better on RMSE and R^2 in the current export.
- The largest empirical win comes from **graph structure**. Removing the graph nearly doubles test MAE.
- The **online correction stack is not yet validated** by the released numbers; all reported online variants are worse than the base forecast.
- The artifacts are strong enough for a professional report or controlled-study manuscript draft, but a Scopus-ready operational paper would still benefit from real AFC data, multi-seed confidence intervals, and per-horizon prediction traces.

## 2. Source Artifacts Used
- Dataset CSVs: [`dataset_train.csv`](../model_eval_q1_full/datasets/dataset_train.csv), [`dataset_val.csv`](../model_eval_q1_full/datasets/dataset_val.csv), [`dataset_test.csv`](../model_eval_q1_full/datasets/dataset_test.csv), [`dataset_full.csv`](../model_eval_q1_full/datasets/dataset_full.csv)
- Main benchmark metrics: [`run_state.json`](../model_eval_q1_full/run_state.json)
- Component ablations: [`ablations_summary.json`](../model_eval_q1_full/ablations_summary.json)
- Online ablations: [`online_ablations.json`](../model_eval_q1_full/online_ablations.json)
- Experimental implementation context: [`model_evaluation.py`](../model_evaluation.py), [`main.py`](../main.py), [`MODEL_ARCHITECTURE.md`](../MODEL_ARCHITECTURE.md)

## 3. Dataset Description
The exported dataset is a multivariate station-level passenger-count panel with 28 stations sampled every 15 minutes from **{stats.timestamps[0]}** to **{stats.timestamps[-1]}**. The benchmark contains **{total_rows:,} timestamps** and **1,400,448 scalar observations** across all station series. The global station-level mean is **{fmt_num(sum(stats.station_mean.values()) / len(stats.station_mean), 3)} passengers per station per 15 minutes**, with an observed maximum of **{int(stats.value_max)}** and a zero-rate of **{fmt_num(stats.value_zero_pct, 3)}%**.

{split_table}

The aggregate demand profile is operationally plausible: the network mean is **{fmt_num(stats.network_mean_15m, 2)} passengers per 15-minute interval**, equivalent to roughly **{fmt_num(daily_network, 0)} passengers per day**. Weekday demand ({fmt_num(weekday_mean, 1)}) clearly exceeds weekend demand ({fmt_num(weekend_mean, 1)}), with the dominant peak occurring during the evening commute around **18:00**.

![Figure 1](figures/fig_01_architecture_flow.svg)
*Figure 1. End-to-end experimental flow reconstructed from the released implementation and evaluation artifacts.*

![Figure 2](figures/fig_02_dataset_timeline.svg)
*Figure 2. Full dataset timeline with daily network demand and explicit train, validation, and test regions.*

![Figure 3](figures/fig_03_temporal_heatmap.svg)
*Figure 3. Mean network demand by day of week and hour of day, computed directly from the exported station-level counts.*

![Figure 4](figures/fig_04_station_loads.svg)
*Figure 4. Station-level ranking by average passenger demand. The highest-load stations are Bogenbai Batyr Ave, Central Park, Baiterek, Khan Shatyr, and Seifullin St.*

### 3.1 Station Heterogeneity
The station-level averages reveal meaningful heterogeneity:

- Highest-load stations: {", ".join(f"{name} ({fmt_num(value, 1)})" for name, value in top5)}
- Lowest-load stations: {", ".join(f"{name} ({fmt_num(value, 1)})" for name, value in bottom5)}
- All stations are moderately to highly variable, with coefficients of variation clustering around roughly 0.82 to 0.87.

### 3.2 Important Validity Note
The benchmark is **synthetic, not raw AFC data**. The repository implementation explicitly states that `model_evaluation.py` generates the Astana-like dataset using the simulator in `main.py`. That simulator injects rush-hour structure, weekend attenuation, weather effects, event boosts, service disruptions, graph diffusion, and post-drift regime shifts. This makes the benchmark reproducible and useful for controlled method comparison, but it must not be described as an operational real-world deployment dataset.

## 4. Input Features and Experimental Setup
The saved CSV files contain the target station counts. The repository source code shows that model training additionally used 14 exogenous input dimensions per station:

{feature_table}

Additional experimental settings extracted from the implementation:

- Lookback window: **48 timesteps** = **12 hours**
- Forecast horizon: **12 timesteps** = **3 hours**
- Granularity: **15 minutes**
- Split rule: **70% / 10% / 20%** chronological train/validation/test
- Target scope in the reported run: **bottom-level stations**
- Hierarchy defined in code: **28 stations + 9 lines + 4 districts + network total = 42 series**
- Scaling: standardization of both `X` and `y` using training data only

## 5. Model Overview
DTS-GSSF follows the dual-timescale design laid out in [`MODEL_ARCHITECTURE.md`](../MODEL_ARCHITECTURE.md). In compact form, the method couples:

1. A graph-structured state-space forecasting backbone for long-range temporal encoding.
2. Adaptive adjacency for learned spatial dependence beyond the physical graph.
3. LoRA-enabled forecast heads to support efficient adaptation.
4. An online residual-correction stack with Kalman-style filtering and reconciliation logic.

For benchmarking, DTS-GSSF was evaluated against DCRNN, Transformer, LSTM Seq2Seq, GRU Seq2Seq, Historical Average, and Seasonal Naive.

## 6. Main Experimental Results
The released test-full metrics are summarized below.

{main_table}

![Figure 5](figures/fig_05_model_comparison.svg)
*Figure 5. Main benchmark comparison across MAE, RMSE, WAPE, and R^2 on the flattened 12-step test evaluation.*

![Figure 6](figures/fig_06_efficiency_frontier.svg)
*Figure 6. Accuracy-efficiency frontier showing MAE versus training time.*

### 6.1 Interpretation
The results support a careful, publication-quality interpretation:

- **Primary strength**: DTS-GSSF achieves the best **MAE** ({fmt_num(dts['test_full']['mae'], 4)}) and **WAPE** ({fmt_num(dts['test_full']['wape'], 4)}).
- **Competitive pressure from DCRNN**: DCRNN is only slightly behind on MAE, and slightly ahead on **RMSE** and **R^2** in this run.
- **Clear baseline separation**: Seasonal Naive remains substantially weaker, confirming that the benchmark is nontrivial.
- **Efficiency trade-off**: DTS-GSSF requires **{fmt_num(dts['train_time_sec'], 1)} seconds** of training in the exported run, far above the lightweight neural and statistical baselines.

Taken together, the benchmark supports a claim of **best-average absolute error**, but not yet a sweeping claim of universal dominance across all metrics.

## 7. Ablation Study
The component ablation results are as follows.

{ablation_table}

![Figure 7](figures/fig_07_ablation.svg)
*Figure 7. Test-full ablation study for DTS-GSSF.*

### 7.1 Ablation Insights
- **Graph structure is essential**: removing the graph increases MAE from {fmt_num(dts['test_full']['mae'], 4)} to {fmt_num(ablations['No Graph']['test_full']['mae'], 4)}, a **+94.38%** degradation.
- **Adaptive adjacency is beneficial**: removing it produces a smaller but still meaningful MAE increase of **+2.43%**.
- **LoRA is nearly neutral in this run**: the `No LoRA` variant changes MAE by only **+0.04%**, suggesting either limited drift sensitivity in the offline benchmark or insufficient adaptation pressure under the released configuration.

This is a strong result for the graph module, a moderate result for adaptive adjacency, and a weak result for LoRA under the current evidence.

## 8. Online Residual-Correction Results
The online ablation file provides the following MAE values:

{online_table}

![Figure 8](figures/fig_08_online_ablation.svg)
*Figure 8. Online residual-correction outcomes from the exported evaluation.*

### 8.1 Online Interpretation
The online stack is the main weak point in the current release:

- The **base** forecast is best.
- Kalman correction alone degrades MAE by **+2.50%**.
- Kalman + reconciliation still remains worse than the base forecast (**+0.96%**).
- The no-adaptation variants degrade further, so adaptation helps relative to no-adaptation, but still fails to surpass the base forecast.

For a paper-rich presentation, this is still publishable as an honest negative-or-mixed result: the architecture’s offline graph backbone is validated, while the online correction layer remains an open research problem.

## 9. Discussion
The current artifacts support four defensible claims:

1. **Spatio-temporal modeling matters**. The graph ablation result is large and robust enough to be central to the story.
2. **DTS-GSSF is a legitimate top-performing model on this benchmark**, but the margin over DCRNN and Transformer is small.
3. **Training efficiency is a real concern**. Any claim of practical superiority should be balanced with the cost profile shown in Figure 6.
4. **The online adaptation subsystem is not yet mature** in the released run and should be reported transparently.

## 10. Limitations
- The dataset is synthetic, which limits external validity.
- The released folder does not include raw prediction tensors, uncertainty estimates, or per-horizon/per-station error traces, so the report cannot reconstruct a full horizon-wise decay analysis.
- The current benchmark export appears to be single-run rather than multi-seed, which prevents confidence intervals or significance testing.
- The online adaptation results are unfavorable, so stronger tuning or redesign may be necessary before emphasizing that contribution in a submission.

## 11. Recommended Positioning for a Scopus/Q1 Paper
If this material is developed into a paper, the safest positioning is:

- Frame the study as a **controlled synthetic benchmark for adaptive passenger-flow forecasting**.
- Make the **graph and adaptive adjacency findings** the empirical core.
- Treat the **online correction layer as exploratory** unless stronger evidence is produced.
- Avoid wording that implies real operational AFC deployment unless real data are added.

## 12. Conclusion
Using only the artifacts in `model_eval_q1_full`, DTS-GSSF emerges as the strongest model on test-full MAE and WAPE, with graph structure providing the dominant source of performance gain. The benchmark is rich enough to support a professional, paper-style report and a strong controlled-study narrative. At the same time, the evidence also shows important caveats: the data are synthetic, the gains over the strongest deep baselines are narrow, and the online correction stack does not yet improve on the base model. Those points do not weaken the report; they make it scientifically credible.

## Appendix A. Reproducibility Notes
- Generated from: [`generate_q1_report.py`](./generate_q1_report.py)
- Output figures: [`figures`](./figures)
- Derived summary JSON: [`report_stats.json`](./report_stats.json)
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def write_stats_json(stats: DatasetStats, run_state: Dict[str, object], ablations: Dict[str, object], online: Dict[str, object]) -> None:
    payload = {
        "dataset": {
            "start": str(stats.timestamps[0]),
            "end": str(stats.timestamps[-1]),
            "rows": len(stats.timestamps),
            "stations": len(stats.station_names),
            "network_mean_15m": stats.network_mean_15m,
            "network_std_15m": stats.network_std_15m,
            "zero_pct": stats.value_zero_pct,
            "split_rows": stats.split_rows,
            "split_ranges": {k: [str(v[0]), str(v[1])] for k, v in stats.split_ranges.items()},
        },
        "results": run_state["results"],
        "ablations": ablations,
        "online": online,
    }
    STATS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    run_state = load_json(DATA_DIR / "run_state.json")
    ablations = load_json(DATA_DIR / "ablations_summary.json")
    online = load_json(DATA_DIR / "online_ablations.json")
    stats = compute_dataset_stats()

    build_architecture_figure(FIG_DIR / "fig_01_architecture_flow.svg")
    build_dataset_timeline_figure(FIG_DIR / "fig_02_dataset_timeline.svg", stats)
    build_temporal_heatmap_figure(FIG_DIR / "fig_03_temporal_heatmap.svg", stats)
    build_station_load_figure(FIG_DIR / "fig_04_station_loads.svg", stats)
    build_model_comparison_figure(FIG_DIR / "fig_05_model_comparison.svg", run_state["results"])
    build_efficiency_frontier_figure(FIG_DIR / "fig_06_efficiency_frontier.svg", run_state["results"])
    build_ablation_figure(FIG_DIR / "fig_07_ablation.svg", run_state["results"]["DTS-GSSF"]["test_full"], ablations)
    build_online_ablation_figure(FIG_DIR / "fig_08_online_ablation.svg", online)
    write_report(stats, run_state, ablations, online)
    write_stats_json(stats, run_state, ablations, online)
    print(f"Wrote report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
