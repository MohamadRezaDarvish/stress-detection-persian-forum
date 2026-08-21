from __future__ import annotations
import argparse, sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


def resolve_root(package_dir: str | Path) -> Path:
    root=Path(package_dir).resolve()
    if not (root/"src").exists(): raise FileNotFoundError("package-dir must be the project root")
    return root


def load_engineer(root: Path):
    sys.path.insert(0,str(root))
    try:
        return joblib.load(root/"data/processed/member1_preprocessor.joblib")
    except Exception as exc:
        from src.feature_engineering import Member1FeatureEngineer
        print(f"Warning: could not load joblib preprocessor ({exc}); using stateless Member1FeatureEngineer fallback.")
        return Member1FeatureEngineer()


def predict_file(input_path: Path, root: Path, output_path: Path) -> pd.DataFrame:
    engineer=load_engineer(root)
    raw=pd.read_csv(input_path)
    X=engineer.transform(raw)
    preds=[]
    for fold in range(5):
        model=CatBoostRegressor(); model.load_model(str(root/f"models/fold_{fold}/model.cbm")); preds.append(np.clip(model.predict(X),1,10))
    matrix=np.vstack(preds)
    out=pd.DataFrame({"unique_post_id":raw["unique_post_id"].astype(str),"prediction":matrix.mean(0),"prediction_std_optional":matrix.std(0,ddof=0)})
    output_path.parent.mkdir(parents=True,exist_ok=True); out.to_csv(output_path,index=False); return out


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',required=True); p.add_argument('--package-dir',default='.'); p.add_argument('--output',required=True); args=p.parse_args()
    out=predict_file(Path(args.input),resolve_root(args.package_dir),Path(args.output)); print(out.head().to_string(index=False)); print(f"Saved {len(out)} predictions to {args.output}")
if __name__=='__main__': main()
