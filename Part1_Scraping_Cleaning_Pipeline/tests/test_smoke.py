from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.cleaning import demo_raw_posts, clean_dataset
from src.common import load_json, PersianTools
from src.proxy_labeling import extract_base_post_features, initial_proxy, discover_lexicons, add_final_counts, refined_proxy

def test_early_pipeline_smoke(tmp_path):
    cfg=load_json(ROOT/'configs/project_config.json'); lex=load_json(ROOT/'configs/lexicons.json')
    ccfg=dict(cfg['cleaning']); ccfg['require_complete_category_after_repair']=False
    raw=demo_raw_posts(); clean,_=clean_dataset(raw,tmp_path/'no_profiles.csv',ccfg,lex,tmp_path/'audit.json',project_root=ROOT)
    assert clean['unique_post_id'].is_unique
    tools=PersianTools(); x=extract_base_post_features(clean,lex,tools); x['stress_proxy_initial']=initial_proxy(x)
    neg,pos,_,_=discover_lexicons(x,cfg['proxy'],lex,tools,demo_mode=True)
    x=add_final_counts(x,neg,pos,tools); x['stress_proxy']=refined_proxy(x)
    assert {'post_neg_count_temp','post_pos_count_temp','post_neg_count','post_pos_count','stress_proxy'} <= set(x.columns)
