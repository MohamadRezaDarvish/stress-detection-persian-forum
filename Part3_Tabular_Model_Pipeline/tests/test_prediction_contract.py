from pathlib import Path
import pandas as pd
from src.common import resolve_project_root, load_json

def test_prediction_contract():
    root=resolve_project_root(Path(__file__).resolve())
    cfg=load_json(root/'configs/project_config.json')
    manifest=pd.read_csv(root/cfg['input_files']['manifest'])
    expected=manifest['model_role'].value_counts().to_dict()
    oof=pd.read_csv(root/'data/handoff/member1_oof_predictions.csv')
    val=pd.read_csv(root/'data/handoff/member1_validation_predictions.csv')
    test=pd.read_csv(root/'data/handoff/member1_test_predictions.csv')
    assert oof.columns.tolist()==['unique_post_id','fold','true_stress','prediction','prediction_std_optional']
    assert val.columns.tolist()==['unique_post_id','model_role','true_stress','prediction','prediction_std_optional']
    assert test.columns.tolist()==val.columns.tolist()
    assert len(oof)==expected['train'] and len(val)==expected['validation'] and len(test)==expected['test']
    assert not oof.unique_post_id.duplicated().any()
    assert oof.prediction.between(*cfg['prediction_range']).all()
    assert val.prediction_std_optional.notna().all() and test.prediction_std_optional.notna().all()
