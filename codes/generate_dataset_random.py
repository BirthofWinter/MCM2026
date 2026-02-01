import os, sys, json, argparse
import numpy as np
import pandas as pd
from dataclasses import asdict

# ===== paths =====
CUR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CUR)
sys.path.insert(0, os.path.join(CUR, "generalization"))

from calculate import BuildingConfig
from optimization_framework import OptimizationWeights, MetricEvaluator
from calculate import ThermalSystem, OverhangStats, VerticalFins, NoShading  # 你们已有


def load_weather_robust(filepath: str) -> pd.DataFrame:
    header_row = 0
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if line.strip().startswith("time"):
                header_row = i
                break
    df = pd.read_csv(filepath, header=0, skiprows=header_row, engine="python", skipfooter=10)
    df.columns = [c.replace(":", "").strip() for c in df.columns]
    df["time"] = pd.to_datetime(df["time"], format="%Y%m%d:%H%M", errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)
    for c in ["Gb(i)", "T2m"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "Gb(i)" not in df.columns:
        df["Gb(i)"] = 0.0
    df = df.dropna(subset=["T2m"]).reset_index(drop=True)
    return df


def load_weights(path: str) -> OptimizationWeights:
    with open(path, "r", encoding="utf-8") as f:
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


def sample_one(rng: np.random.Generator):
    """
    这里定义你要采样的“参数空间”
    连续用 uniform；离散用 choice
    """
    # 遮阳几何
    overhang_depth = rng.uniform(0.0, 2.5)
    fin_depth = rng.uniform(0.0, 2.0)

    # 建筑与窗
    window_ratio = rng.uniform(0.15, 0.55)
    u_window = rng.choice([1.0, 1.5, 2.0, 2.8, 3.5])
    tau_window = rng.uniform(0.25, 0.75)
    tau_visible = rng.uniform(0.3, 0.7)

    # 材料
    k_const_absorb = rng.uniform(0.2, 0.8)
    layer_thickness = rng.choice([0.2, 0.3, 0.4])
    k_wall = rng.choice([0.04, 0.1, 0.3, 0.8, 1.8])
    rho_wall = rng.choice([100.0, 800.0, 1800.0, 2400.0])
    c_wall = rng.choice([800.0, 900.0, 1000.0, 1100.0])

    # 通风/策略
    ventilation_rate = rng.uniform(0.3, 4.0)
    night_cooling = bool(rng.integers(0, 2))
    night_vent_rate = rng.uniform(3.0, 12.0)

    # 光照模型
    c_room_light = rng.uniform(0.3, 0.7)

    return dict(
        overhang_depth=overhang_depth,
        fin_depth=fin_depth,
        window_ratio=window_ratio,
        u_window=u_window,
        tau_window=tau_window,
        tau_visible=tau_visible,
        k_const_absorb=k_const_absorb,
        layer_thickness=layer_thickness,
        k_wall=k_wall,
        rho_wall=rho_wall,
        c_wall=c_wall,
        ventilation_rate=ventilation_rate,
        night_cooling=night_cooling,
        night_vent_rate=night_vent_rate,
        c_room_light=c_room_light,
    )


def build_shading(sample, facade_azimuth_deg):
    """
    你可以在这里切换：只做南向 overhang，东西向 fins 等
    为了先把数据量做起来，我给一个“组合策略”示例：
    """
    # 这里只做“南向”作为例子；你也可以把 facade_azimuth_deg 作为采样变量
    if facade_azimuth_deg == 0:
        return OverhangStats(window_height=2.0, window_width=2.0, wall_azimuth_deg=0, depth_L=sample["overhang_depth"])
    elif facade_azimuth_deg in (90, -90):
        return VerticalFins(window_height=2.0, window_width=2.0, wall_azimuth_deg=facade_azimuth_deg, depth_L=sample["fin_depth"])
    else:
        return NoShading(window_height=2.0, window_width=2.0, wall_azimuth_deg=facade_azimuth_deg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, default=103.8)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weather = load_weather_robust(args.weather).iloc[:720].copy()  # 先用 720 小时加速
    weights = load_weights(args.weights)
    evaluator = MetricEvaluator(weights)

    rng = np.random.default_rng(args.seed)
    rows = []

    for i in range(args.n):
        s = sample_one(rng)

        cfg = BuildingConfig(
            module_width=4.0, room_depth=8.0, room_height=3.5,
            Q_internal=500.0,
            window_ratio=float(s["window_ratio"]),
            layer_thickness=float(s["layer_thickness"]),
            k_wall=float(s["k_wall"]),
            rho_wall=float(s["rho_wall"]),
            c_wall=float(s["c_wall"]),
            h_in=8.0, h_out=20.0,
            u_window=float(s["u_window"]),
            tau_window=float(s["tau_window"]),
            tau_visible=float(s["tau_visible"]),
            ventilation_rate=float(s["ventilation_rate"]),
            c_room_light=float(s["c_room_light"]),
            night_cooling=bool(s["night_cooling"]),
            night_vent_rate=float(s["night_vent_rate"]),
            k_const_absorb=float(s["k_const_absorb"]),
        )
        
        # ===== cost model (consistent with optimization_framework.py) =====
        facade_area = 420.0   # 你可以先固定南立面 420m2（或以后作为参数）
        unit_price = 200.0
        glass_price = 300.0
        wall_price  = 100.0

        wwr = cfg.window_ratio

        cost_overhang = facade_area * wwr * s["overhang_depth"] * unit_price
        cost_fins     = facade_area * wwr * s["fin_depth"]     * unit_price
        cost_window   = (facade_area * wwr * glass_price) + (facade_area * (1 - wwr) * wall_price)

        cost = cost_overhang + cost_fins + cost_window


        # 这里先固定一个立面（南向）做数据集；想扩展就把 facade 也采样
        facade_az = 0
        shading = build_shading(s, facade_az)

        sim = ThermalSystem(cfg, shading, location_lat=args.lat, location_lon=args.lon)
        res = sim.simulate(weather)

        metrics = evaluator.evaluate(res, cost)

        row = {**s}
        row.update({
            "Latitude": args.lat,
            "Longitude": args.lon,
            "FacadeAzimuth": facade_az,
            **metrics
        })

        # 记录权重方便复现（可选）
        for k, v in asdict(weights).items():
            row[f"W_{k}"] = v

        rows.append(row)

        if (i + 1) % 100 == 0:
            print(f"[{i+1}/{args.n}] done")

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("[OK] saved:", args.out, "rows=", len(df))


if __name__ == "__main__":
    main()
