from __future__ import annotations
from pathlib import Path
import hashlib, re
import numpy as np
import pandas as pd
from .common import PersianTools, save_json

import pickle
PUNCT_RE = re.compile(r'[،؛؟!\.\,"\'\(\)\[\]\{\}\<\>«»]')
EMOJI_RE = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+", flags=re.UNICODE)


def demo_raw_posts():
    rows = [
      {'thread_id':'1','thread_title':'نگرانم','post_id':None,'author':'کاربر۱','profile_url':'u1','posted_at':'1404/01/01 10:00','content':'خیلی نگرانم و استرس دارم 😭 لطفا کمک کنید','likes':2,'reply_to':None,'user_post_count':120,'is_starter':True,'section_name':'demo'},
      {'thread_id':'1','thread_title':'نگرانم','post_id':'1001','author':'کاربر۲','profile_url':'u2','posted_at':'1404/01/01 10:02','content':'عزیزم آروم باش امیدوارم خوب بشی ❤️','likes':3,'reply_to':None,'user_post_count':50,'is_starter':False,'section_name':'demo'},
      {'thread_id':'2','thread_title':'سوال','post_id':None,'author':'کاربر۳','profile_url':'u3','posted_at':'1404/01/02 11:00','content':'خیلی میترسم! درد دارم و گریه میکنم','likes':0,'reply_to':None,'user_post_count':10,'is_starter':True,'section_name':'demo'},
      {'thread_id':'2','thread_title':'سوال','post_id':'1002','author':'کاربر۴','profile_url':'u4','posted_at':'1404/01/02 11:03','content':'انشاالله سالم و شاد باشی دوست من','likes':1,'reply_to':None,'user_post_count':800,'is_starter':False,'section_name':'demo'},
    ]
    # Repeat with small variations so proxy demo has enough observations.
    out=[]
    for i in range(60):
        for r in rows:
            x=dict(r); x['thread_id']=f"{r['thread_id']}_{i}"; x['post_id']=(f"{r['post_id']}_{i}" if r['post_id'] else None); x['posted_at']=r['posted_at']+f"_{i}"; out.append(x)
    return pd.DataFrame(out)


def load_raw_posts(root: Path, glob_pattern: str, use_demo=True, combined_all_file=None, prefer_combined_all=True):
    # Historical reproduction path:
    # if the already-combined raw scrape exists, use it as the single authoritative
    # input instead of concatenating scraper checkpoint fragments again.
    if combined_all_file:
        direct = root / combined_all_file
        if prefer_combined_all and direct.exists():
            df = pd.read_csv(direct, low_memory=False)
            if 'content' not in df.columns or 'thread_id' not in df.columns:
                raise ValueError(f"{direct} does not look like the expected combined Ninisite post table.")
            return df, [str(direct.relative_to(root))]

    files = [p for p in root.glob(glob_pattern)
             if p.is_file()
             and 'threads_manifest' not in p.name
             and 'thread_status' not in p.name]
    if not files:
        if use_demo:
            return demo_raw_posts(), ['DEMO_GENERATED']
        expected = f" or {combined_all_file}" if combined_all_file else ""
        raise FileNotFoundError(f"No raw post CSVs matched {glob_pattern}{expected}")

    frames=[]
    used=[]
    for p in files:
        try:
            df=pd.read_csv(p, low_memory=False)
            if 'content' in df.columns and 'thread_id' in df.columns:
                frames.append(df)
                used.append(str(p.relative_to(root)))
        except Exception:
            pass
    if not frames:
        if use_demo:
            return demo_raw_posts(), ['DEMO_GENERATED']
        raise ValueError('No usable raw post files found.')
    return pd.concat(frames, ignore_index=True, sort=False), used


def normalize_education(value, tools: PersianTools):
    if pd.isna(value) or not str(value).strip(): return 'نامشخص'
    t=tools.normalize(value).strip()
    patterns=[
      (r'فوق\s*لیسانس|کارشناسی\s*ارشد|ارشد','فوق لیسانس'),(r'دکتر[ایيى]|دکترا','دکترا'),
      (r'فوق\s*دیپلم','فوق دیپلم'),(r'کاردانی','کاردانی'),(r'کارشناس|کارشناسی|لیسانس','کارشناسی'),
      (r'دیپلم','دیپلم'),(r'سیکل','سیکل'),(r'دانشجو','دانشجو'),(r'نامشخص|درج نشده','نامشخص')]
    for pat,label in patterns:
        if re.search(pat,t): return label
    return 'سایر'


def parse_age(value):
    if pd.isna(value): return -1
    m=re.search(r'(\d{1,3})', str(value))
    if not m: return -1
    x=int(m.group(1)); return x if 10 <= x <= 110 else -1


def normalize_gender(frame, imputation='female'):
    g=frame.get('gender', pd.Series('', index=frame.index)).fillna('').astype(str)
    code=pd.to_numeric(frame.get('gender_code', pd.Series(np.nan,index=frame.index)), errors='coerce')
    female=g.str.contains('زن', na=False); male=g.str.contains('مرد', na=False)
    code=code.where(code.isin([0,1]), np.nan); code=code.mask(female,1).mask(male,0)
    if imputation == 'female': code=code.fillna(1)
    elif imputation == 'male': code=code.fillna(0)
    else: code=code.fillna(-1)
    frame['gender_code']=code.astype(int)
    frame['gender']=frame['gender_code'].map({1:'زن',0:'مرد',-1:'نامشخص'})
    return frame


def merge_profiles(df, profile_path: Path):
    if not profile_path.exists() or 'profile_url' not in df.columns:
        return df
    p=pd.read_csv(profile_path, low_memory=False).drop_duplicates('profile_url', keep='last')
    cols=[c for c in ['profile_url','gender','gender_code','join_date','age','education','status','children_count','signature'] if c in p.columns]
    if len(cols)<=1: return df
    out=df.merge(p[cols], on='profile_url', how='left', suffixes=('','_profile'))
    for c in cols:
        if c=='profile_url': continue
        pc=c+'_profile'
        if pc in out:
            if c not in out: out[c]=out[pc]
            else: out[c]=out[c].where(out[c].notna() & out[c].astype(str).str.strip().ne(''), out[pc])
            out.drop(columns=[pc], inplace=True)
    return out


def robust_deduplicate(df):
    before=len(df)
    df=df.drop_duplicates(keep='first').copy()
    # For rows with no post_id, remove exact identity duplicates.
    noid=df['post_id'].isna() | df['post_id'].astype(str).str.strip().isin(['','nan','None']) if 'post_id' in df else pd.Series(True,index=df.index)
    fallback_cols=[c for c in ['thread_id','author','posted_at','content'] if c in df]
    if fallback_cols:
        dup_noid=df.loc[noid].duplicated(fallback_cols, keep='first')
        drop_idx=df.loc[noid].index[dup_noid]
        df=df.drop(index=drop_idx)
    # Do NOT blindly drop repeated post_id when content differs; audit conflicts instead.
    conflict_ids=[]
    if 'post_id' in df:
        valid=df[~noid].copy()
        if len(valid):
            nuniq=valid.groupby(valid['post_id'].astype(str))['content'].nunique(dropna=False)
            conflict_ids=nuniq[nuniq>1].index.astype(str).tolist()
            safe_ids=set(nuniq[nuniq==1].index.astype(str))
            safe_mask=valid['post_id'].astype(str).isin(safe_ids)
            safe=valid[safe_mask].drop_duplicates(subset=['post_id'], keep='first')
            conflict=valid[~safe_mask]
            df=pd.concat([df[noid], safe, conflict], ignore_index=True, sort=False)
    return df.reset_index(drop=True), {'rows_before':before,'rows_after':len(df),'conflicting_post_ids':len(conflict_ids),'conflicting_post_id_examples':conflict_ids[:20]}


def _canonical_site_post_id(value):
    """Canonicalize historical numeric Ninisite post IDs.

    pandas may read an integer ID as a float because the column contains NaNs,
    e.g. 412966185 -> 412966185.0. The historical cleaner used int(post_id)
    before converting to string, so terminal '.0' must be removed.
    """
    if pd.isna(value):
        return ''
    s=str(value).strip()
    if not s or s in ('nan','None'):
        return ''
    if re.fullmatch(r'[+-]?\d+\.0+', s):
        return str(int(float(s)))
    return s


def make_unique_post_ids(df):
    ids=[]; seen={}
    for _,r in df.iterrows():
        pid=_canonical_site_post_id(r.get('post_id'))
        if pid:
            base=pid
        elif bool(r.get('is_starter',False)):
            base=f"thread_{r.get('thread_id','')}_starter"
        else:
            base=f"thread_{r.get('thread_id','')}_{r.get('author','')}_{r.get('posted_at','')}"
        key=base
        if key in seen:
            payload='|'.join(str(r.get(c,'')) for c in ['thread_id','author','posted_at','content'])
            suffix=hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]
            key=f"{base}__{suffix}"
        seen[key]=1; ids.append(key)
    df['unique_post_id']=ids
    assert df['unique_post_id'].is_unique
    return df


def signature_features(df, tools: PersianTools, neg_words, pos_words):
    sig=df.get('signature', pd.Series('', index=df.index)).fillna('').astype(str)
    missing=sig.str.strip().eq('')
    out=pd.DataFrame(index=df.index)
    out['sig_char_count']=sig.str.len(); out['sig_punct_count']=sig.str.count(PUNCT_RE)
    out['sig_question_count']=sig.str.count(r'[؟?]'); out['sig_excl_count']=sig.str.count(r'[!！]')
    out['sig_emoji_count']=sig.str.count(EMOJI_RE)
    neg={tools.stem(w) for w in neg_words}; pos={tools.stem(w) for w in pos_words}
    wc=[]; nc=[]; pc=[]
    for s in sig:
        stems=[tools.stem(t) for t in tools.tokenize(s)]; wc.append(len(stems)); nc.append(sum(x in neg for x in stems)); pc.append(sum(x in pos for x in stems))
    out['sig_word_count']=wc; out['sig_neg_count']=nc; out['sig_pos_count']=pc
    out.loc[missing,:]=-1
    return pd.concat([df,out], axis=1)



def repair_categories_from_cache(df, cfg, project_root: Path | None):
    """Replay the completed historical category/sub-category repair.

    IMPORTANT: this function performs NO scraping and does not need the three
    large historical source CSVs.

    During development, three source files were scraped without category and
    sub_category columns. The source filename nevertheless identified the forum
    section. The team manually supplied one category/sub-category pair per file
    and matched rows back into the cleaned table using:

        thread_id + '_' + author + '_' + posted_at

    `inputs/enrichment/category_recovery.csv` freezes that completed mapping.
    """
    df=df.copy()
    if 'category_source' not in df.columns:
        existing_category = (
            df.get('category', pd.Series('', index=df.index))
            .fillna('')
            .astype(str)
            .str.strip()
            .ne('')
        )
        df['category_source'] = np.where(
            existing_category,
            'original_combined_all',
            'unresolved_before_recovery'
        )

    report={
        'mode':'cached_historical_replay',
        'network_requests':0,
        'cache_loaded':False,
        'updated_rows':0,
        'remaining_missing_category':0,
        'remaining_missing_sub_category':0,
    }
    if project_root is None:
        return df, report

    p=project_root/cfg.get('category_recovery_csv','inputs/enrichment/category_recovery.csv')
    if not p.exists():
        raise FileNotFoundError(
            'Historical category-recovery cache is missing.\n'
            f'Expected: {p}\n'
            'Notebook 02 intentionally does not recreate the three large source-file '
            'repair interactively. Copy the frozen category_recovery.csv into '
            'inputs/enrichment/ and rerun.'
        )

    recovery=pd.read_csv(p,low_memory=False)
    need={'match_key','category','sub_category'}
    if not need.issubset(recovery.columns):
        raise ValueError(f'Category recovery file must contain {sorted(need)}')
    if recovery['match_key'].duplicated().any():
        raise ValueError('category_recovery.csv contains duplicate match_key values.')

    key=(df['thread_id'].astype(str)+'_'+df['author'].astype(str)+'_'+df['posted_at'].astype(str))
    rec=recovery.set_index('match_key')
    rec_cat=key.map(rec['category'])
    rec_sub=key.map(rec['sub_category'])

    category_missing=df['category'].isna() | df['category'].astype(str).str.strip().eq('')
    sub_missing=df['sub_category'].isna() | df['sub_category'].astype(str).str.strip().eq('')
    repairable=(category_missing | sub_missing) & rec_cat.notna()

    fill_cat=category_missing & rec_cat.notna()
    fill_sub=sub_missing & rec_sub.notna()
    df.loc[fill_cat,'category']=rec_cat[fill_cat]
    df.loc[fill_sub,'sub_category']=rec_sub[fill_sub]
    df.loc[repairable, 'category_source'] = 'cached_historical_source_file_repair'
    df.loc[~repairable & (
        df['category'].notna() & df['category'].astype(str).str.strip().ne('')
    ), 'category_source'] = 'original_combined_all'

    remaining_cat=df['category'].isna() | df['category'].astype(str).str.strip().eq('')
    remaining_sub=df['sub_category'].isna() | df['sub_category'].astype(str).str.strip().eq('')
    report.update({
        'cache_loaded':True,
        'cache_rows':int(len(recovery)),
        'updated_rows':int(repairable.sum()),
        'filled_category_values':int(fill_cat.sum()),
        'filled_sub_category_values':int(fill_sub.sum()),
        'remaining_missing_category':int(remaining_cat.sum()),
        'remaining_missing_sub_category':int(remaining_sub.sum()),
        'expected_historical_updates':cfg.get('historical_expected_category_updates'),
    })

    if cfg.get('require_complete_category_after_repair',False):
        if remaining_cat.any() or remaining_sub.any():
            raise AssertionError(
                'Historical category repair is incomplete: '
                f'{int(remaining_cat.sum())} category and '
                f'{int(remaining_sub.sum())} sub_category values remain missing.'
            )
    return df, report


def apply_gender_recovery(df, cfg, project_root: Path | None):
    """Replay the completed historical gender outcome without web requests.

    Historical notebook evidence:
    6,932 profile results were recorded, including exactly 1,437 male profiles.

    Because the complete checkpoint was not retained, the final male profile URL
    set was reconstructed once from the historical final dataset and
    combined_all.csv. That replay cache also contains exactly 1,437 unique URLs.

    Cleaning:
    - matching profile_url -> male;
    - explicit female observations -> female;
    - all remaining unresolved rows -> female under the historical project rule.
    """
    df = df.copy()
    report = {
        'mode': 'historical_male_profile_cache_replay',
        'network_requests': 0,
        'male_cache_loaded': False,
        'male_cache_profiles': 0,
        'rows_marked_male_from_cache': 0,
        'remaining_assumed_female_rows': 0,
    }

    code = pd.to_numeric(
        df.get('gender_code', pd.Series(np.nan, index=df.index)),
        errors='coerce'
    )
    gender_text = (
        df.get('gender', pd.Series('', index=df.index))
        .fillna('').astype(str).str.strip()
    )

    df['gender_source'] = np.where(
        code.eq(0) | gender_text.eq('مرد'),
        'original_male',
        np.where(
            code.eq(1) | gender_text.eq('زن'),
            'original_female',
            'unresolved_before_historical_recovery'
        )
    )

    if project_root is None:
        raise ValueError('project_root is required for historical gender replay.')

    cache_path = project_root / cfg.get(
        'historical_male_profile_cache_pkl',
        'inputs/enrichment/historical_male_profile_cache.pkl'
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f'Historical male-profile cache is missing: {cache_path}'
        )

    with open(cache_path, 'rb') as f:
        male_cache = pickle.load(f)

    if not isinstance(male_cache, dict):
        raise TypeError('Historical male-profile cache must be a dict.')
    if any(v != ('مرد', 0) for v in male_cache.values()):
        raise ValueError('Historical male-profile cache contains unexpected values.')

    male_urls = {str(u).strip() for u in male_cache.keys()}
    report['male_cache_loaded'] = True
    report['male_cache_profiles'] = int(len(male_urls))

    expected_profiles = cfg.get('historical_expected_unique_male_profiles')
    if expected_profiles is not None and len(male_urls) != int(expected_profiles):
        raise AssertionError(
            f'Expected {expected_profiles} male profiles, found {len(male_urls)}.'
        )

    if 'profile_url' not in df.columns:
        raise KeyError(
            'profile_url is required during historical gender replay. '
            'Keep it until enrichment is complete.'
        )

    profile = df['profile_url'].fillna('').astype(str).str.strip()
    male_match = profile.isin(male_urls)

    df.loc[male_match, 'gender'] = 'مرد'
    df.loc[male_match, 'gender_code'] = 0
    df.loc[male_match, 'gender_source'] = 'historical_scraped_male_profile_replay'
    report['rows_marked_male_from_cache'] = int(male_match.sum())

    current_code = pd.to_numeric(df['gender_code'], errors='coerce')
    current_gender = df['gender'].fillna('').astype(str).str.strip()

    explicit_female = (~male_match) & (
        current_code.eq(1) | current_gender.eq('زن')
    )
    df.loc[explicit_female, 'gender'] = 'زن'
    df.loc[explicit_female, 'gender_code'] = 1

    current_code = pd.to_numeric(df['gender_code'], errors='coerce')
    current_gender = df['gender'].fillna('').astype(str).str.strip()
    unresolved = (~male_match) & ~(
        current_code.eq(1) | current_gender.eq('زن')
    )

    if cfg.get('gender_unresolved_policy') == 'assume_female_after_historical_male_recovery':
        df.loc[unresolved, 'gender'] = 'زن'
        df.loc[unresolved, 'gender_code'] = 1
        df.loc[unresolved, 'gender_source'] = (
            'remaining_unresolved_assumed_female_after_historical_recovery'
        )
        report['remaining_assumed_female_rows'] = int(unresolved.sum())

    final_code = pd.to_numeric(df['gender_code'], errors='coerce')
    unresolved_final = ~final_code.isin([0, 1])
    if unresolved_final.any():
        raise AssertionError(
            f'{int(unresolved_final.sum())} gender rows remain unresolved.'
        )

    report['final_male_rows'] = int(final_code.eq(0).sum())
    report['final_female_rows'] = int(final_code.eq(1).sum())
    report['final_unresolved_rows'] = int(unresolved_final.sum())
    report['final_gender_source_counts'] = (
        df['gender_source'].value_counts(dropna=False).to_dict()
    )

    expected_male_rows = cfg.get('historical_expected_final_male_rows')
    report['expected_historical_final_male_rows'] = expected_male_rows
    report['male_row_count_matches_historical_final_dataset'] = (
        expected_male_rows is None
        or report['final_male_rows'] == int(expected_male_rows)
    )
    return df, report

def clean_dataset(raw, profiles_path: Path, cfg, lexicons, audit_path: Path, project_root: Path | None = None):
    # Cached historical category/gender repair needs the project root.
    # The notebook passes it explicitly. As a defensive fallback, infer it from
    # the standard audit path: ROOT/outputs/audits/<file>.json.
    if project_root is None:
        try:
            ap = Path(audit_path).resolve()
            if ap.parent.name == 'audits' and ap.parent.parent.name == 'outputs':
                project_root = ap.parent.parent.parent
        except Exception:
            project_root = None

    tools=PersianTools(); audit={'input_rows':len(raw),'persian_backend':tools.backend}
    df=merge_profiles(raw.copy(), profiles_path)
    if 'content' not in df: raise KeyError('content column is required')

    # Historical order matters: filter using RAW content first, then normalize.
    # Normalizing before the 3-word filter can move borderline posts across the threshold.
    valid_content=df['content'].notna() & df['content'].astype(str).str.strip().ne('')
    df=df[valid_content].copy()
    audit['after_nonempty_content']=len(df)
    raw_wc=df['content'].astype(str).str.split().str.len()
    df=df[raw_wc >= int(cfg.get('min_words',3))].copy()
    audit['after_raw_three_word_filter']=len(df)

    # The legacy notebook did not apply an author-blank filter in this stage.
    if cfg.get('drop_missing_author',False) and 'author' in df:
        df=df[df['author'].notna() & df['author'].astype(str).str.strip().ne('')].copy()
    audit['after_optional_author_filter']=len(df)

    # Historical fidelity:
    # The original cleaner filtered RAW content but did NOT globally overwrite
    # df['content'] with Hazm-normalized text. Later tokenization routines
    # normalized locally when needed. Preserve raw post text here because
    # character/punctuation/emoji counts in cleaner3 were computed on raw text.
    df['content'] = df['content'].astype(str)
    audit['content_globally_normalized'] = False

    # thread_url was removed in the historical cleaner before feature work.
    df.drop(columns=['thread_url'], inplace=True, errors='ignore')

    for c in ['thread_title','category','sub_category','signature','education','join_date']:
        if c not in df: df[c]=''
    if 'age' not in df.columns:
        df['age']=df['user_age'] if 'user_age' in df.columns else ''
    df['education_clean']=[normalize_education(x,tools) for x in df['education']]
    df['age_num']=[parse_age(x) for x in df['age']]

    # Historical category repair is replayed from one frozen cache; no source-file scraping/merging is repeated.
    df,cat_report=repair_categories_from_cache(df,cfg,project_root); audit['category_repair']=cat_report

    # Historical final cleaned corpus dropped the 9 rows whose thread title could not be recovered.
    missing_title=df['thread_title'].isna() | df['thread_title'].astype(str).str.strip().eq('')
    audit['missing_thread_title_before_drop']=int(missing_title.sum())
    if cfg.get('drop_missing_thread_title',True): df=df[~missing_title].copy()
    audit['after_title_filter']=len(df)

    # Historical gender repair is replayed from cached artifacts only.
    # NO profile scraping is performed in the cleaning stage.
    raw_code=pd.to_numeric(df.get('gender_code',pd.Series(np.nan,index=df.index)),errors='coerce')
    audit['gender_code_before_recovery']=raw_code.value_counts(dropna=False).to_dict()
    df,gender_report=apply_gender_recovery(df,cfg,project_root)
    audit['gender_recovery']=gender_report
    audit['gender_code_after_recovery']=pd.to_numeric(
        df['gender_code'],errors='coerce'
    ).value_counts(dropna=False).to_dict()

    # profile_url is required only for the historical gender replay above.
    # The historical cleaned/modeling table did not retain profile URLs.
    df.drop(columns=['profile_url'], inplace=True, errors='ignore')
    audit['profile_url_dropped_after_gender_recovery'] = True

    df,dedup=robust_deduplicate(df); audit.update(dedup)
    df=make_unique_post_ids(df)
    df=signature_features(df, tools, lexicons['seed_negative_words'], lexicons['seed_positive_words'])

    # Add historical signature emoji polarity counts as separate features.
    sig=df.get('signature',pd.Series('',index=df.index)).fillna('').astype(str)
    missing_sig=sig.str.strip().eq('')
    def count_nonoverlap(text, symbols):
        pats=sorted({re.escape(x) for x in symbols if x},key=len,reverse=True)
        return len(re.findall('|'.join(pats),str(text))) if pats else 0
    df['sig_pos_emoji']=[count_nonoverlap(x,lexicons['positive_emojis']) for x in sig]
    df['sig_neg_emoji']=[count_nonoverlap(x,lexicons['negative_emojis']) for x in sig]
    df.loc[missing_sig,['sig_pos_emoji','sig_neg_emoji']]=-1

    audit['output_rows']=len(df); audit['unique_post_id_unique']=bool(df['unique_post_id'].is_unique)
    audit['historical_reference_rows_after_text_filter']=cfg.get('historical_reference_rows_after_text_filter')
    audit['historical_reference_final_rows']=cfg.get('historical_reference_final_rows')
    save_json(audit,audit_path)
    return df.reset_index(drop=True),audit

