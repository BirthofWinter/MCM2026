# ml_optimize_rf.py  (核心改版：条件优化，只搜索设计变量)
import argparse
import os
import json
import numpy as np
import pandas as pd
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor



def json_default(o):
    # numpy / pandas 标量 -> Python 标量
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    # pandas Timestamp / numpy datetime -> string
    if isinstance(o, (pd.Timestamp, np.datetime64)):
        return str(o)
    # 兜底：最后转字符串（避免再炸）
    return str(o)

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

def get_feature_importance(pipe, numeric_cols, cat_cols):
    model = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    if not hasattr(model, "feature_importances_"):
        return None

    feature_names = []
    for name, trans, cols in pre.transformers_:
        if name == "cat":
            ohe_names = trans.get_feature_names_out(cols).tolist()
            feature_names.extend(ohe_names)
        elif name == "num":
            feature_names.extend(cols)

    imp = model.feature_importances_
    imp_df = pd.DataFrame({"feature": feature_names, "importance": imp})
    return imp_df.sort_values("importance", ascending=False).reset_index(drop=True)

def sample_design_candidates(n_samples, seed, shading_types,
                             overhang_range=(0.0, 3.0), fin_range=(0.0, 3.0)):
    rng = np.random.default_rng(seed)
    cand = pd.DataFrame({
        "overhang_depth": rng.uniform(overhang_range[0], overhang_range[1], size=n_samples),
        "fin_depth": rng.uniform(fin_range[0], fin_range[1], size=n_samples),
        "ShadingType": rng.choice(shading_types, size=n_samples, replace=True),
    })

    # 结构约束：不同遮阳类型只激活对应深度
    mask_overhang = cand["ShadingType"] == "Overhang"
    mask_fins = cand["ShadingType"] == "Fins"
    mask_none = cand["ShadingType"] == "NoShading"
    cand.loc[mask_overhang, "fin_depth"] = 0.0
    cand.loc[mask_fins, "overhang_depth"] = 0.0
    cand.loc[mask_none, ["overhang_depth", "fin_depth"]] = 0.0

    return cand



def scenario_key_to_name(scenario_cols, key):
    if not scenario_cols:
        return "ALL"
    if isinstance(key, (str, int, float)):
        key = (key,)
    return "_".join([f"{scenario_cols[i]}={key[i]}" for i in range(len(scenario_cols))])


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV dataset path")
    ap.add_argument("--model", choices=["rf", "dt"], default="rf", help="rf=RandomForest, dt=DecisionTree")

    # 你指定：情景 = 自然因素 + 房屋内部参数（都固定），设计变量只优化遮阳相关
    ap.add_argument("--scenario_cols", default="", help="comma-separated scenario columns; blank=ALL")
    ap.add_argument("--min_rows", type=int, default=80, help="minimum rows per scenario to report a best")

    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--search_samples", type=int, default=50000, help="how many candidates sampled in design-space search")
    ap.add_argument("--out_dir", default="ml_results", help="output directory")

    # 设计空间约束（按你刚刚确认的）
    ap.add_argument("--azimuths", default="0,90,180,270", help="allowed FacadeAzimuth values, comma-separated")
    ap.add_argument("--overhang_min", type=float, default=0.0)
    ap.add_argument("--overhang_max", type=float, default=3.0)
    ap.add_argument("--fin_min", type=float, default=0.0)
    ap.add_argument("--fin_max", type=float, default=3.0)
    ap.add_argument("--shading_types", default="", help="allowed ShadingType values; blank=use values from data")

    args = ap.parse_args()

    df = pd.read_csv(args.data)
    target = detect_target(df)

    # scenario cols
    scenario_cols = []
    if args.scenario_cols.strip():
        scenario_cols = [c.strip() for c in args.scenario_cols.split(",") if c.strip() in df.columns]

    # 设计变量（固定为你要优化的四个）
    design_num = [c for c in ["overhang_depth", "fin_depth", "FacadeAzimuth"] if c in df.columns]
    design_cat = [c for c in ["ShadingType"] if c in df.columns]

    # 情景变量：你指定的 scenario_cols
    scenario_num = []
    scenario_cat = []
    for c in scenario_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            scenario_num.append(c)
        else:
            scenario_cat.append(c)

    # 全局训练特征 = 情景变量 + 设计变量
    numeric_cols = sorted(list(set(scenario_num + design_num)))
    cat_cols = sorted(list(set(scenario_cat + design_cat)))

    X = df[numeric_cols + cat_cols].copy()
    y = df[target].astype(float).copy()

    # train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed
    )

    model = build_model(args.model, random_state=args.seed)
    pipe = make_pipeline(numeric_cols, cat_cols, model)
    pipe.fit(X_train, y_train)

    pred = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    r2 = float(r2_score(y_test, pred))

    os.makedirs(args.out_dir, exist_ok=True)

    # feature importances (global)
    imp_df = get_feature_importance(pipe, numeric_cols, cat_cols)
    if imp_df is not None:
        imp_path = os.path.join(args.out_dir, "feature_importances.csv")
        imp_df.to_csv(imp_path, index=False, encoding="utf-8-sig")

    # 设计空间取值
    azimuths = [int(x.strip()) for x in args.azimuths.split(",") if x.strip()]
    if args.shading_types.strip():
        shading_types = [s.strip() for s in args.shading_types.split(",") if s.strip()]
    else:
        shading_types = df["ShadingType"].dropna().unique().tolist() if "ShadingType" in df.columns else ["Overhang", "Fins", "NoShading"]
        # 兜底：若数据里没有这三种，仍然允许它们
        for s in ["Overhang", "Fins", "NoShading"]:
            if s not in shading_types:
                shading_types.append(s)

    # 需要输出的 scenario 列表
    if scenario_cols:
        groups = list(df.groupby(scenario_cols, dropna=False))
    else:
        groups = [(("ALL",), df)]
        scenario_cols = []

    summary_rows = []

    print("Target:", target)
    print("Scenario cols:", scenario_cols if scenario_cols else "(none, ALL)")
    print(f"Global model: {args.model} | MAE={mae:.4f} R2={r2:.3f}")

    for key, g in groups:
        if isinstance(key, (int, float, str)):
            key = (key,)
        g = g.dropna(subset=[target]).copy()
        if len(g) < args.min_rows:
            continue

        # 固定情景：取该 scenario 的第一行（因为 group by 后情景列一致）
        scenario_fixed = {}
        for c in scenario_cols:
            scenario_fixed[c] = g.iloc[0][c]

        # 1) Observed best：真实数据里见过的最优
        best_obs_idx = g[target].astype(float).idxmin()
        best_obs = g.loc[best_obs_idx, :]

        # 2) Predicted best：固定情景 + 搜索设计变量
        cand_design = sample_design_candidates(
            n_samples=args.search_samples,
            seed=args.seed,
            shading_types=shading_types,
            overhang_range=(args.overhang_min, args.overhang_max),
            fin_range=(args.fin_min, args.fin_max),
        )

        # 拼接成模型输入：每个候选都带上同一套 scenario_fixed
        cand = cand_design.copy()
        for c, v in scenario_fixed.items():
            cand[c] = v

        # 对齐列
        cand_X = cand[numeric_cols + cat_cols].copy()
        cand_pred = pipe.predict(cand_X)
        best_pred_i = int(np.argmin(cand_pred))
        best_pred_row = cand.iloc[best_pred_i].to_dict()
        best_pred_score = float(cand_pred[best_pred_i])

        scenario_name = scenario_key_to_name(scenario_cols, key)

        # shorten filename: hash of scenario_name
        h = hashlib.md5(scenario_name.encode("utf-8")).hexdigest()[:10]
        out_json = os.path.join(args.out_dir, f"best_{h}.json")


        payload = {
            "scenario_cols": scenario_cols,
            "scenario_key": list(key) if scenario_cols else ["ALL"],
            "rows": int(len(g)),
            "global_model": args.model,
            "global_metrics": {"MAE": mae, "R2": r2},
            "observed_best": {
                "Total_Score": float(best_obs[target]),
                "params": {k: (best_obs[k].item() if hasattr(best_obs[k], "item") else best_obs[k]) for k in (design_num + design_cat) if k in best_obs.index},
            },
            "predicted_best": {
                "Pred_Total_Score": best_pred_score,
                "params": {k: best_pred_row[k] for k in (design_num + design_cat) if k in best_pred_row},
            },
            "scenario_fixed": scenario_fixed
        }
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default)


        # 汇总
        row_sum = {c: key[i] for i, c in enumerate(scenario_cols)}
        row_sum.update({
            "rows": len(g),
            "Global_MAE": mae,
            "Global_R2": r2,
            "ObservedBest_TotalScore": float(best_obs[target]),
            "PredBest_PredTotalScore": best_pred_score,
            "PredBest_overhang_depth": best_pred_row.get("overhang_depth", None),
            "PredBest_fin_depth": best_pred_row.get("fin_depth", None),
            "PredBest_FacadeAzimuth": best_pred_row.get("FacadeAzimuth", None),
            "PredBest_ShadingType": best_pred_row.get("ShadingType", None),
        })
        summary_rows.append(row_sum)

        print(f"[Scenario {scenario_name}] rows={len(g)} obs_best={float(best_obs[target]):.4f} pred_best={best_pred_score:.4f}")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "scenario_best_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n[OK] Saved:")
    print(" -", summary_path)
    if imp_df is not None:
        print(" -", os.path.join(args.out_dir, "feature_importances.csv"))
    print(" - per-scenario best_*.json files in", args.out_dir)


if __name__ == "__main__":
    main()
