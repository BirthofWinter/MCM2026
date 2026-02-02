import argparse
import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor


# ---------------------------
# Utilities
# ---------------------------

def pick_existing_cols(df, candidates):
    return [c for c in candidates if c in df.columns]

def detect_target(df):
    for name in ["Total_Score", "total_score", "score", "TotalScore"]:
        if name in df.columns:
            return name
    raise ValueError("Cannot find target column. Expected one of: Total_Score / total_score / score / TotalScore")

def detect_scenario_cols(df):
    """
    你可以按需改：哪些列决定“不同情景”。
    常见：Latitude / FacadeAzimuth / ShadingType / WeatherFile / Scenario
    """
    candidates = ["Scenario", "WeatherFile", "Latitude", "Longitude", "FacadeAzimuth", "ShadingType"]
    cols = [c for c in candidates if c in df.columns]
    # 至少用纬度/朝向分组（如果存在）
    if not cols:
        cols = []
    return cols

def detect_feature_cols(df, target_col, scenario_cols, weight_cols_prefix="W_"):
    """
    特征列 = 数值参数 + 类别策略参数
    - 默认排除 target、场景列、权重列(以 W_ 开头)、时间序列中间量等
    - 你可以在 allowlist 里更精确控制
    """
    # 常见“可调参数列”（与你们模型一致）
    allow_numeric = [
        "overhang_depth", "fin_depth",
        "window_ratio", "u_window", "tau_window", "tau_visible",
        "k_const_absorb", "layer_thickness", "k_wall", "rho_wall", "c_wall",
        "ventilation_rate", "night_vent_rate", "night_cooling",
        "c_room_light",
        "module_width", "room_depth", "room_height",
        "FacadeAzimuth", "Latitude", "Longitude"
    ]
    allow_cat = ["ShadingType"]

    numeric_cols = pick_existing_cols(df, allow_numeric)
    cat_cols = pick_existing_cols(df, allow_cat)

    # 兜底：若 allow_numeric 太少，就自动从数据里挑“看起来像参数”的数值列
    if len(numeric_cols) < 3:
        auto_numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        auto_numeric = [c for c in auto_numeric
                        if c != target_col
                        and c not in scenario_cols
                        and not c.startswith(weight_cols_prefix)
                        and c not in ["Strategy_Cost"]]  # Strategy_Cost 可作为特征也可排除
        # 合并并去重
        numeric_cols = sorted(list(set(numeric_cols + auto_numeric)))

    return numeric_cols, cat_cols

def build_model(model_type="rf", random_state=42):
    if model_type == "dt":
        reg = DecisionTreeRegressor(
            max_depth=12,
            min_samples_leaf=5,
            random_state=random_state
        )
    else:
        reg = RandomForestRegressor(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=-1
        )
    return reg

def make_pipeline(numeric_cols, cat_cols, model):
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", numeric_cols),
        ],
        remainder="drop"
    )
    pipe = Pipeline([
        ("pre", pre),
        ("model", model)
    ])
    return pipe

def sample_candidates_within_ranges(df_group, numeric_cols, cat_cols, n_samples=20000, seed=0):
    """
    在“该情景的数据范围内”采样更多候选组合：
    - 数值列：在 min/max 间均匀采样；若只有少数离散值，则从离散集合里抽
    - 类别列：从该情景已有的类别中抽
    """
    rng = np.random.default_rng(seed)
    cand = {}

    for c in numeric_cols:
        vals = df_group[c].dropna().values
        if len(vals) == 0:
            continue
        uniq = np.unique(vals)
        # 如果是离散值较少（比如材料参数），直接从已有取值集合采样
        if len(uniq) <= 10:
            cand[c] = rng.choice(uniq, size=n_samples, replace=True)
        else:
            lo, hi = float(np.min(vals)), float(np.max(vals))
            if lo == hi:
                cand[c] = np.full(n_samples, lo)
            else:
                cand[c] = rng.uniform(lo, hi, size=n_samples)

    for c in cat_cols:
        cats = df_group[c].dropna().unique().tolist()
        if not cats:
            continue
        cand[c] = rng.choice(cats, size=n_samples, replace=True)

    return pd.DataFrame(cand)

def get_feature_importance(pipe, numeric_cols, cat_cols):
    """
    输出特征重要性（RF/DT 支持 feature_importances_）。
    对 OneHot 之后的类别特征，会展开成多个列名。
    """
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]

    if not hasattr(model, "feature_importances_"):
        return None

    # OneHot 特征名
    feature_names = []
    for name, trans, cols in pre.transformers_:
        if name == "cat":
            ohe = trans
            ohe_names = ohe.get_feature_names_out(cols).tolist()
            feature_names.extend(ohe_names)
        elif name == "num":
            feature_names.extend(cols)

    imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
    imp_df = imp_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return imp_df


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV dataset path, e.g. sungrove_mc_5000.csv")
    ap.add_argument("--model", choices=["rf", "dt"], default="rf", help="rf=RandomForest, dt=DecisionTree")
    ap.add_argument("--scenario_cols", default="", help="comma-separated scenario columns; blank=auto detect")
    ap.add_argument("--min_rows", type=int, default=80, help="minimum rows per scenario to train")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search_samples", type=int, default=30000, help="how many candidates sampled for model-based search")
    ap.add_argument("--out_dir", default="ml_results", help="output directory")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    target = detect_target(df)

    # scenario cols
    if args.scenario_cols.strip():
        scenario_cols = [c.strip() for c in args.scenario_cols.split(",") if c.strip() in df.columns]
    else:
        scenario_cols = detect_scenario_cols(df)

    # 如果你不想把 ShadingType/FacadeAzimuth 当“情景”，可以手动指定 scenario_cols
    # 例如: --scenario_cols Latitude
    print("Target:", target)
    print("Scenario cols:", scenario_cols if scenario_cols else "(none)")

    os.makedirs(args.out_dir, exist_ok=True)

    # 如果没有场景列，就全局做一次
    if not scenario_cols:
        groups = [(("ALL",), df)]
        scenario_cols = []
    else:
        groups = list(df.groupby(scenario_cols, dropna=False))

    summary_rows = []
    all_importances = []

    for key, g in groups:
        if isinstance(key, (int, float, str)):
            key = (key,)
        g = g.dropna(subset=[target]).copy()
        if len(g) < args.min_rows:
            continue

        # detect features per group（保证列存在）
        numeric_cols, cat_cols = detect_feature_cols(g, target, scenario_cols)

        # 训练数据
        X = g[numeric_cols + cat_cols].copy()
        y = g[target].astype(float).copy()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )

        model = build_model(args.model, random_state=args.seed)
        pipe = make_pipeline(numeric_cols, cat_cols, model)
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))

        # 1) Observed best (真实最优)
        best_obs_idx = g[target].astype(float).idxmin()
        best_obs = g.loc[best_obs_idx, :]

        # 2) Model-based best (采样搜索预测最优)
        cand = sample_candidates_within_ranges(g, numeric_cols, cat_cols,
                                               n_samples=args.search_samples,
                                               seed=args.seed)
        cand_pred = pipe.predict(cand)
        best_pred_i = int(np.argmin(cand_pred))
        best_pred_row = cand.iloc[best_pred_i].to_dict()
        best_pred_score = float(cand_pred[best_pred_i])

        # importances
        imp_df = get_feature_importance(pipe, numeric_cols, cat_cols)
        if imp_df is not None:
            imp_df["scenario_key"] = str(key)
            all_importances.append(imp_df)

        # 输出每个情景的详细结果 JSON
        scenario_name = "_".join([f"{scenario_cols[i]}={key[i]}" for i in range(len(scenario_cols))]) if scenario_cols else "ALL"
        out_json = os.path.join(args.out_dir, f"best_{scenario_name}.json")

        payload = {
            "scenario_cols": scenario_cols,
            "scenario_key": list(key),
            "rows": int(len(g)),
            "model": args.model,
            "metrics": {"MAE": mae, "R2": r2},
            "observed_best": {
                "Total_Score": float(best_obs[target]),
                "params": {c: (best_obs[c].item() if hasattr(best_obs[c], "item") else best_obs[c]) for c in (numeric_cols + cat_cols) if c in best_obs.index},
            },
            "predicted_best": {
                "Pred_Total_Score": best_pred_score,
                "params": best_pred_row
            }
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        # 汇总行（写 CSV）
        row_sum = {c: key[i] for i, c in enumerate(scenario_cols)}
        row_sum.update({
            "rows": len(g),
            "MAE": mae,
            "R2": r2,
            "ObservedBest_TotalScore": float(best_obs[target]),
            "PredBest_PredTotalScore": best_pred_score
        })
        # 把关键参数也放进汇总（若存在）
        for p in ["overhang_depth", "fin_depth", "window_ratio", "u_window", "tau_window", "k_const_absorb", "ventilation_rate", "night_cooling", "night_vent_rate", "ShadingType", "FacadeAzimuth"]:
            if p in best_pred_row:
                row_sum[f"PredBest_{p}"] = best_pred_row[p]
            if p in best_obs.index:
                row_sum[f"ObsBest_{p}"] = (best_obs[p].item() if hasattr(best_obs[p], "item") else best_obs[p])

        summary_rows.append(row_sum)

        print(f"[Scenario {scenario_name}] rows={len(g)} MAE={mae:.4f} R2={r2:.3f} "
              f"obs_best={float(best_obs[target]):.4f} pred_best={best_pred_score:.4f}")

    # 写汇总 CSV
    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "scenario_best_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    # 写特征重要性 CSV（可选）
    if all_importances:
        imp_all = pd.concat(all_importances, ignore_index=True)
        imp_path = os.path.join(args.out_dir, "feature_importances.csv")
        imp_all.to_csv(imp_path, index=False, encoding="utf-8-sig")

    print("\n[OK] Saved:")
    print(" -", summary_path)
    if all_importances:
        print(" -", os.path.join(args.out_dir, "feature_importances.csv"))
    print(" - per-scenario best_*.json files in", args.out_dir)


if __name__ == "__main__":
    main()
