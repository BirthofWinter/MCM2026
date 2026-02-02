import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from dataclasses import asdict, replace

# ===== Path fix (same style as your current script) =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERALIZATION_DIR = os.path.join(CURRENT_DIR, "generalization")
sys.path.append(GENERALIZATION_DIR)

from optimization_framework import OptimizationWeights, MetricEvaluator, ParameterOptimizer
from calculate import BuildingConfig, ThermalSystem, OverhangStats, VerticalFins, NoShading


# -------------------------
# IO helpers (copied/compatible with your old ml_precompute.py)
# -------------------------
def load_weather_robust(filepath: str) -> pd.DataFrame:
    """
    Compatible with PVGIS csv. Header starts from 'time'.
    """
    header_row = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().startswith("time"):
                header_row = i
                break

    df = pd.read_csv(filepath, header=0, skiprows=header_row, engine="python", skipfooter=10)
    df.columns = [c.replace(":", "").strip() for c in df.columns]

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
        df = df.dropna(subset=["time"])
    else:
        df["time"] = pd.date_range("2023-01-01", periods=len(df), freq="H")

    for c in ["Gb(i)", "Gd(i)", "Gr(i)", "H_sun", "T2m", "WS10m", "Int"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["T2m"]) if "T2m" in df.columns else df
    if "Gb(i)" not in df.columns:
        df["Gb(i)"] = 0.0

    return df.reset_index(drop=True)


def load_weights(weights_json_path: str) -> OptimizationWeights:
    with open(weights_json_path, "r", encoding="utf-8") as f:
        w = json.load(f)

    return OptimizationWeights(
        w_comfort_dev=float(w.get("w_comfort_dev", 1.0)),
        w_temp_var=float(w.get("w_temp_var", 0.5)),
        w_energy=float(w.get("w_energy", 0.0001)),
        w_cost=float(w.get("w_cost", 0.01)),
        w_light_quality=float(w.get("w_light_quality", 0.1)),
        target_temp_min=float(w.get("target_temp_min", 20.0)),
        target_temp_max=float(w.get("target_temp_max", 26.0)),
        scale_temp=float(w.get("scale_temp", 1.0)),
        scale_energy=float(w.get("scale_energy", 1e6)),
        scale_cost=float(w.get("scale_cost", 1000.0)),
        scale_light=float(w.get("scale_light", 1e5)),
    )


def build_base_config() -> BuildingConfig:
    # keep exactly the same defaults as your old script
    return BuildingConfig(
        module_width=4.0,
        room_depth=8.0,
        room_height=3.5,

        Q_internal=500.0,
        window_ratio=0.45,

        layer_thickness=0.3,
        k_wall=0.8,
        rho_wall=1800.0,
        c_wall=1000.0,

        h_in=8.0,
        h_out=20.0,

        u_window=2.8,
        tau_window=0.75,
        tau_visible=0.6,

        ventilation_rate=2.0,

        c_room_light=0.5,

        night_cooling=False,
        night_vent_rate=5.0,

        k_const_absorb=0.6,
    )


# -------------------------
# Joint sampling core
# -------------------------
def parse_list_arg(s: str, cast_fn=str):
    items = [x.strip() for x in s.split(",") if x.strip()]
    return [cast_fn(x) for x in items]


def make_shading(shading_type: str, win_h: float, win_w: float, azimuth: float,
                overhang_depth: float, fin_depth: float):
    if shading_type == "Overhang":
        return OverhangStats(win_h, win_w, azimuth, depth_L=float(overhang_depth))
    if shading_type == "Fins":
        return VerticalFins(win_h, win_w, azimuth, depth_L=float(fin_depth))
    return NoShading(win_h, win_w, azimuth)


def estimate_joint_cost(po: ParameterOptimizer, shading_type: str, overhang_depth: float, fin_depth: float, facade_area: float):
    """
    Use the same cost model already inside ParameterOptimizer:
    - Overhang: cost(overhang_depth)
    - Fins: cost(fin_depth)
    - NoShading: 0
    """
    if shading_type == "Overhang":
        return float(po.estimate_cost("overhang_depth", float(overhang_depth), facade_area))
    if shading_type == "Fins":
        return float(po.estimate_cost("fin_depth", float(fin_depth), facade_area))
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", required=True, help="weather csv path")
    parser.add_argument("--lat", required=True, type=float, help="latitude")
    parser.add_argument("--weights", required=True, help="weights.json path")
    parser.add_argument("--out", default="dataset_joint.csv", help="output csv")

    # NEW: joint sampling controls
    parser.add_argument("--n_samples", type=int, default=5000, help="number of joint samples to generate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hours", type=int, default=720, help="simulate first N hours for speed (default 720=30 days)")

    # design space constraints (your confirmed choices)
    parser.add_argument("--shading_types", default="Overhang,Fins,NoShading", help="allowed shading types")
    parser.add_argument("--azimuths", default="0,90,180,270", help="allowed FacadeAzimuth values (deg)")
    parser.add_argument("--overhang_min", type=float, default=0.0)
    parser.add_argument("--overhang_max", type=float, default=3.0)
    parser.add_argument("--fin_min", type=float, default=0.0)
    parser.add_argument("--fin_max", type=float, default=3.0)

    args = parser.parse_args()

    weather_df = load_weather_robust(args.weather)
    if weather_df.empty:
        raise RuntimeError("Weather dataframe is empty. Check input file.")

    # subset for speed
    if args.hours is not None and args.hours > 0 and len(weather_df) > args.hours:
        weather_df = weather_df.iloc[:args.hours].copy()

    weights = load_weights(args.weights)
    base_cfg = build_base_config()

    # reuse evaluator + cost logic from your framework
    # ParameterOptimizer already constructs MetricEvaluator internally, but we want clean access.
    po = ParameterOptimizer(base_config=base_cfg, weather_df=weather_df, location_lat=args.lat, weights=weights)
    evaluator = MetricEvaluator(weights)

    rng = np.random.default_rng(args.seed)

    shading_types = parse_list_arg(args.shading_types, str)
    azimuths = parse_list_arg(args.azimuths, float)

    # Use the same window dimensions as your sweep code
    win_h, win_w = 2.0, 1.5

    rows = []
    facade_area_for_cost = 420.0  # keep consistent with your old run_sweep() cost call

    for i in range(args.n_samples):
        shading_type = rng.choice(shading_types)
        azimuth = float(rng.choice(azimuths))

        overhang_depth = float(rng.uniform(args.overhang_min, args.overhang_max))
        fin_depth = float(rng.uniform(args.fin_min, args.fin_max))

        # Optional: make unused depth = 0 for interpretability
        if shading_type == "Overhang":
            fin_depth = 0.0
        elif shading_type == "Fins":
            overhang_depth = 0.0
        else:  # NoShading
            overhang_depth = 0.0
            fin_depth = 0.0

        shade = make_shading(shading_type, win_h, win_w, azimuth, overhang_depth, fin_depth)
        system = ThermalSystem(base_cfg, shade, location_lat=args.lat)

        res_df = system.simulate(weather_df)

        cost = estimate_joint_cost(po, shading_type, overhang_depth, fin_depth, facade_area_for_cost)
        metrics = evaluator.evaluate(res_df, cost)

        # Build a single sample row
        row = {}
        row.update(metrics)

        # decision vars
        row["ShadingType"] = shading_type
        row["FacadeAzimuth"] = azimuth
        row["overhang_depth"] = overhang_depth
        row["fin_depth"] = fin_depth

        # scenario context
        row["Latitude"] = args.lat
        row["WeatherFile"] = os.path.basename(args.weather)

        # include building config as scenario columns (so ml_optimize_rf can condition on them)
        cfg_dict = asdict(base_cfg)
        for k, v in cfg_dict.items():
            row[f"Cfg_{k}"] = v

        # include weights for reproducibility
        for k, v in asdict(weights).items():
            row[f"W_{k}"] = v

        rows.append(row)

        if (i + 1) % max(1, args.n_samples // 10) == 0:
            print(f"[{i+1}/{args.n_samples}] samples generated...")

    out_df = pd.DataFrame(rows)

    # Recommend putting target first
    front_cols = ["Total_Score", "Discomfort_Avg", "Temp_Std", "Total_Energy_MJ", "Strategy_Cost", "Raw_Total_Solar_MJ"]
    cols = front_cols + [c for c in out_df.columns if c not in front_cols]
    out_df = out_df[cols]

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved joint dataset: {args.out} | rows={len(out_df)}")


if __name__ == "__main__":
    main()
