from pathlib import Path
import pandas as pd
from catboost import CatBoostRegressor
from src.common import resolve_project_root, load_json
from src.validate_inputs import load_inputs
from src.prepare_data import prepare_data

def test_oof_no_in_sample_leakage_and_reproducibility():
    root=resolve_project_root(Path(__file__).resolve())
    cfg=load_json(root/'configs/project_config.json')
    bundle=load_inputs(root,cfg)
    prepared=prepare_data(bundle['handoff'],root,cfg)
    saved=pd.read_csv(root/'data/handoff/member1_oof_predictions.csv')
    merged=saved.merge(prepared['raw']['train'][['unique_post_id','oof_fold']],on='unique_post_id',validate='one_to_one')
    assert (merged['fold'].astype(int)==merged['oof_fold'].astype(int)).all()
    for fold in range(5):
        config=load_json(root/f'models/fold_{fold}/fold_config.json')
        held=set(config['heldout_ids'])
        predicted=set(saved.loc[saved.fold.eq(fold),'unique_post_id'].astype(str))
        assert held==predicted
