from __future__ import annotations
from collections import Counter
from pathlib import Path
import re
import numpy as np
import pandas as pd
from .common import PersianTools, save_json

PERSIAN_TOKEN=re.compile(r'^[\u0600-\u06FF\u200C\u200D]+$', re.UNICODE)
PUNCT_RE=r'[،؛؟!\.\,"\'\(\)\[\]\{\}\<\>«»]'
EMOJI_RE=re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re.UNICODE)


def count_any(text, symbols):
    """Count non-overlapping emoji alternatives once per textual occurrence.

    The previous reconstruction summed str.count() independently for every symbol,
    which double-counted overlapping forms such as ❤, ❤️ and ❤️‍🔥. The historical
    notebook used one regex alternation, so one location should not automatically
    contribute to every overlapping symbol variant.
    """
    s=str(text or '')
    pats=sorted({re.escape(x) for x in symbols if x}, key=len, reverse=True)
    return len(re.findall('|'.join(pats), s)) if pats else 0


def extract_base_post_features(df, lexicons, tools: PersianTools):
    df=df.copy(); text=df['content'].fillna('').astype(str)
    df['post_char_count']=text.str.len(); df['post_punct_count']=text.str.count(PUNCT_RE)
    df['post_question_count']=text.str.count(r'[؟?]'); df['post_excl_count']=text.str.count(r'[!！]')
    df['post_emoji_count']=text.str.count(EMOJI_RE)
    pos_e=lexicons['positive_emojis']; neg_e=lexicons['negative_emojis']
    df['post_pos_emoji']=[count_any(x,pos_e) for x in text]; df['post_neg_emoji']=[count_any(x,neg_e) for x in text]
    old_neg={tools.stem(w) for w in lexicons['seed_negative_words']}; old_pos={tools.stem(w) for w in lexicons['seed_positive_words']}
    words=[]; neg=[]; pos=[]
    for x in text:
        stems=[tools.stem(t) for t in tools.tokenize(x)]
        words.append(len(stems)); neg.append(sum(s in old_neg for s in stems)); pos.append(sum(s in old_pos for s in stems))
    df['post_word_count']=words
    # Preserve these historical high-precision features explicitly.
    df['post_neg_count_temp']=neg; df['post_pos_count_temp']=pos
    return df


def initial_proxy(df):
    return (
      df['post_neg_count_temp']*4 + df['post_neg_emoji']*2 + df['post_excl_count']*0.5 +
      (df['post_question_count']>0).astype(int)*1.2 - df['post_pos_count_temp']*0.2 - df['post_pos_emoji']*0.5
    )


def count_persian_stems(series, tools: PersianTools):
    c=Counter()
    for text in series.fillna('').astype(str):
        for t in tools.tokenize(text):
            s=tools.stem(t)
            if PERSIAN_TOKEN.match(s) and len(s)>=2: c[s]+=1
    return c


def discover_lexicons(df, cfg, lexicons, tools: PersianTools, demo_mode=False):
    high_thr=df['stress_proxy_initial'].quantile(float(cfg['high_quantile']))
    high=df[df['stress_proxy_initial']>=high_thr]
    if cfg.get('low_mode')=='nsmallest':
        n=min(int(cfg.get('low_n',162000)), len(df))
        low=df.nsmallest(n,'stress_proxy_initial')
    else:
        low_thr=df['stress_proxy_initial'].quantile(0.25); low=df[df['stress_proxy_initial']<=low_thr]
    hc=count_persian_stems(high['content'],tools); lc=count_persian_stems(low['content'],tools)
    min_count=2 if demo_mode else int(cfg['min_count']); ratio=float(cfg['ratio_threshold'])
    neg=[]; pos=[]
    for w,h in hc.items():
        if h<min_count: continue
        l=lc.get(w,0); r=np.inf if l==0 else h/l
        if l==0 or r>ratio: neg.append((w,h,l,r))
    for w,l in lc.items():
        if l<min_count: continue
        h=hc.get(w,0); r=np.inf if h==0 else l/h
        if h==0 or r>ratio: pos.append((w,h,l,r))
    neg_noise=set(lexicons['negative_noise_stems']); pos_noise=set(lexicons['positive_noise_stems'])
    neg=[x for x in neg if x[0] not in neg_noise]; pos=[x for x in pos if x[0] not in pos_noise]
    table=pd.DataFrame([
      {'stem':w,'association':'negative/high_proxy','high_freq':h,'low_freq':l,'ratio':r} for w,h,l,r in neg
    ]+[
      {'stem':w,'association':'positive/low_proxy','high_freq':h,'low_freq':l,'ratio':r} for w,h,l,r in pos
    ])
    return set(x[0] for x in neg), set(x[0] for x in pos), table, {'high_threshold':float(high_thr),'high_rows':len(high),'low_rows':len(low),'negative_stems':len(neg),'positive_stems':len(pos),'min_count_used':min_count}


def add_final_counts(df, neg_stems, pos_stems, tools: PersianTools):
    # Historical cleaner3 did one additional Stemmer.stem() pass on the discovered
    # candidate stems before applying them to all posts. Hazm's stemmer is not
    # guaranteed to be perfectly idempotent, so reproducing that detail matters.
    final_neg_stems={tools.stem(w) for w in neg_stems}
    final_pos_stems={tools.stem(w) for w in pos_stems}

    df=df.copy(); n=[]; p=[]
    for text in df['content'].fillna('').astype(str):
        stems=[tools.stem(t) for t in tools.tokenize(text)]
        n.append(sum(s in final_neg_stems for s in stems))
        p.append(sum(s in final_pos_stems for s in stems))
    df['post_neg_count']=n
    df['post_pos_count']=p
    return df


def refined_proxy(df):
    return (
      df['post_neg_count_temp']*4 + df['post_neg_count']*0.5 + df['post_neg_emoji']*2 +
      df['post_excl_count'].clip(0,6)*1.5 + df['post_question_count'].clip(0,5)*0.5 -
      df['post_pos_count_temp']*0.3 - df['post_pos_count']*0.08 - df['post_pos_emoji']*0.5
    )


def sample_for_labeling(df, sampling, seed=42, demo_mode=False):
    # Important: is_starter is NOT used to calculate the proxy or rank posts.
    # It is retained only for audit/export because the corpus contains many replies.
    high_q=float(sampling['high_quantile']); high_thr=df['stress_proxy'].quantile(high_q)
    low=df[(df['stress_proxy']>=float(sampling['low_min'])) & (df['stress_proxy']<=float(sampling['low_max']))]
    mid=df[(df['stress_proxy']>float(sampling['low_max'])) & (df['stress_proxy']<high_thr)]
    high=df[df['stress_proxy']>=high_thr]
    req={'low':int(sampling['n_low']),'mid':int(sampling['n_mid']),'high':int(sampling['n_high'])}
    if demo_mode:
        req={k:min(v,max(5,len(pool)//3)) for (k,v),pool in zip(req.items(),[low,mid,high])}
    def take(pool,n):
        return pool.sample(n=min(n,len(pool)), random_state=seed) if len(pool) else pool
    parts=[]
    for name,pool in [('low',low),('mid',mid),('high',high)]:
        x=take(pool,req[name]).copy(); x['sampling_band']=name; parts.append(x)
    sampled=pd.concat(parts,ignore_index=True).drop_duplicates('unique_post_id').sample(frac=1,random_state=seed).reset_index(drop=True)
    return sampled, {'high_threshold':float(high_thr),'pool_sizes':{'low':len(low),'mid':len(mid),'high':len(high)},'sample_sizes':sampled['sampling_band'].value_counts().to_dict()}


def annotation_template(sampled, annotator_name):
    cols=[c for c in ['unique_post_id','content','is_starter','sampling_band','stress_proxy'] if c in sampled]
    out=sampled[cols].copy()
    out[f'{annotator_name}_stress']=''; out[f'{annotator_name}_anxiety']=''; out[f'{annotator_name}_depression']=''
    return out
