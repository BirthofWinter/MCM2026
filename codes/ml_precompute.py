import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
from dataclasses import asdict

# ===== Path fix for generalization =====
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
GENERALIZATION_DIR = os.path.join(CURRENT_DIR, "generalization")
sys.path.append(GENERALIZATION_DIR)

# ===== Now imports will work =====
from optimization_framework import OptimizationWeights, ParameterOptimizer
from calculate import BuildingConfig



def load_weather_robust(filepath: str) -> pd.DataFrame:
    """
    兼容 PVGIS 导出的 csv：前面可能有说明行，真正表头从 'time' 开始
    """
    header_row = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().startswith("time"):
                header_row = i
                break

    df = pd.read_csv(filepath, header=0, skiprows=header_row, engine="python", skipfooter=10)
    df.columns = [c.replace(":", "").strip() for c in df.columns]

    # time: 'yyyymmdd:HHMM'
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
        df = df.dropna(subset=["time"])
    else:
        # 如果没有 time，就造一个小时序列
        df["time"] = pd.date_range("2023-01-01", periods=len(df), freq="H")

    # 数值列尽量转成 float
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

    # 只允许你调权重（以及 target/scale），其他不让脚本变复杂
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
    return BuildingConfig(
        # Geometry
        module_width=4.0,
        room_depth=8.0,
        room_height=3.5,

        # Heat/internal
        Q_internal=500.0,
        window_ratio=0.45,

        # Wall
        layer_thickness=0.3,
        k_wall=0.8,
        rho_wall=1800.0,
        c_wall=1000.0,

        # Convection
        h_in=8.0,
        h_out=20.0,

        # Window / light
        u_window=2.8,
        tau_window=0.75,
        tau_visible=0.6,

        # Ventilation
        ventilation_rate=2.0,

        # Lighting model
        c_room_light=0.5,

        # Advanced
        night_cooling=False,
        night_vent_rate=5.0,

        # Absorptance
        k_const_absorb=0.6,
    )


def default_sweeps():
    return [
        # ===== Shading geometry (optimizer-controlled) =====
        dict(param_name="overhang_depth", shading_type="Overhang", facade_azimuth=0,
             values=[0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 2.0, 2.5,0.45,0.65,0.75,1.0,1.1,1.15,1.75,2.25]),
        dict(param_name="fin_depth", shading_type="Fins", facade_azimuth=90,
             values=[0.0, 0.4, 0.8, 1.2, 1.6, 2.0,0.1,0.2,0.3,0.5,0.6,0.7,0.9,1,1.1,1.3,1.4,1.7]),

        # ===== Window / glazing =====
        dict(param_name="window_ratio", shading_type="Overhang", facade_azimuth=0,
             values=[0.15, 0.25, 0.35, 0.45, 0.55]),
        dict(param_name="u_window", shading_type="Overhang", facade_azimuth=0,
             values=[1.0, 1.5, 2.0, 2.8, 3.5]),
        dict(param_name="tau_window", shading_type="Overhang", facade_azimuth=0,
             values=[0.25, 0.35, 0.45, 0.55, 0.75]),
        dict(param_name="tau_visible", shading_type="Overhang", facade_azimuth=0,
             values=[0.30, 0.45, 0.60, 0.70]),

        # ===== Wall / materials =====
        dict(param_name="k_const_absorb", shading_type="Overhang", facade_azimuth=0,
             values=[0.2, 0.3, 0.4, 0.6, 0.8]),
        dict(param_name="layer_thickness", shading_type="Overhang", facade_azimuth=0,
             values=[0.15, 0.2, 0.3, 0.4]),
        dict(param_name="k_wall", shading_type="Overhang", facade_azimuth=0,
             values=[0.04, 0.1, 0.3, 0.8, 1.8]),
        dict(param_name="rho_wall", shading_type="Overhang", facade_azimuth=0,
             values=[100.0, 800.0, 1800.0, 2400.0]),
        dict(param_name="c_wall", shading_type="Overhang", facade_azimuth=0,
             values=[800.0, 900.0, 1000.0, 1100.0]),

        # ===== Ventilation / advanced =====
        dict(param_name="ventilation_rate", shading_type="Overhang", facade_azimuth=0,
             values=[0.3, 0.5, 1.0, 2.0, 4.0,0.2,0.15,0.12,0.14,0.18,0.88,0.9,1.3,1.4,1.5,2.1,3.0,3.5,2.5]),
        dict(param_name="night_vent_rate", shading_type="Overhang", facade_azimuth=0,
             values=[3.0, 5.0, 8.0, 12.0,4,7,9]),
        dict(param_name="night_cooling", shading_type="Overhang", facade_azimuth=0,
             values=[0, 1]),

        # ===== Lighting model =====
        dict(param_name="c_room_light", shading_type="Overhang", facade_azimuth=0,
             values=[0.3, 0.5, 0.7,0.4,0.6,0.8]),

        # ===== Geometry (optional) =====
        dict(param_name="room_depth", shading_type="Overhang", facade_azimuth=0,
             values=[4.0, 5.0,6.0,7.0, 8.0,9.0, 10.0, 11.0,12.0]),
    ]



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weather", required=True, help="weather csv path")
    parser.add_argument("--lat", required=True, type=float, help="latitude")
    parser.add_argument("--weights", required=True, help="weights.json path")
    parser.add_argument("--out", default="dataset.csv", help="output csv")
    args = parser.parse_args()

    weather_df = load_weather_robust(args.weather)
    if weather_df.empty:
        raise RuntimeError("Weather dataframe is empty. Check input file.")

    weights = load_weights(args.weights)
    base_cfg = build_base_config()

    optimizer = ParameterOptimizer(
        base_config=base_cfg,
        weather_df=weather_df,
        location_lat=args.lat,
        weights=weights
    )

    all_rows = []
    sweeps = default_sweeps()

    for sweep in sweeps:
        param_name = sweep["param_name"]
        shading_type = sweep["shading_type"]
        facade_azimuth = sweep["facade_azimuth"]
        values = sweep["values"]

        df_res = optimizer.run_sweep(
            param_name=param_name,
            param_range=values,
            shading_type=shading_type,
            facade_azimuth=facade_azimuth
        )

        # 增加一些上下文列，方便 ML 训练/分组
        df_res["Latitude"] = args.lat
        df_res["WeatherFile"] = os.path.basename(args.weather)
        df_res["ShadingType"] = shading_type
        df_res["FacadeAzimuth"] = facade_azimuth

        # 记录权重（便于复现实验/不同纬度不同权重）
        for k, v in asdict(weights).items():
            df_res[f"W_{k}"] = v

        # 记录基准配置（可选）
        df_res["Base_window_ratio"] = base_cfg.window_ratio
        df_res["Base_wall_area"] = base_cfg.wall_area

        all_rows.append(df_res)

    out_df = pd.concat(all_rows, ignore_index=True)

    # 推荐：把 Total_Score 放最前面，方便筛选最优
    front_cols = ["Total_Score", "Discomfort_Avg", "Temp_Std", "Total_Energy_MJ", "Strategy_Cost", "Raw_Total_Solar_MJ"]
    cols = front_cols + [c for c in out_df.columns if c not in front_cols]
    out_df = out_df[cols]

    out_df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"[OK] Saved dataset: {args.out} | rows={len(out_df)}")


if __name__ == "__main__":
    main()
