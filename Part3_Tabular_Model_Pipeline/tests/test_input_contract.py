from pathlib import Path
from src.common import resolve_project_root, load_json
from src.validate_inputs import load_inputs, validate_inputs

def test_input_contract():
    root=resolve_project_root(Path(__file__).resolve())
    cfg=load_json(root/'configs/project_config.json')
    bundle=load_inputs(root,cfg)
    audit=validate_inputs(bundle,cfg)
    manifest_counts={k:int(v) for k,v in bundle['manifest']['model_role'].value_counts().to_dict().items()}
    manifest_folds={str(int(k)):int(v) for k,v in bundle['manifest'].loc[bundle['manifest']['model_role'].eq('train'),'oof_fold'].astype(int).value_counts().sort_index().to_dict().items()}
    assert audit['input_contract_passed']
    assert audit['role_counts']==manifest_counts
    assert audit['fold_counts']==manifest_folds
