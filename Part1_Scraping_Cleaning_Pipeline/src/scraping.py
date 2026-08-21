from __future__ import annotations
from pathlib import Path
from urllib.parse import urljoin, urlparse, parse_qs, urlencode, urlunparse
import csv, random, re, time
import pandas as pd
import requests
from bs4 import BeautifulSoup


class TemporaryBlock(RuntimeError):
    pass


def with_page(url: str, page: int) -> str:
    if page == 1:
        return url
    parts = urlparse(url)
    q = parse_qs(parts.query)
    q["page"] = [str(page)]
    query = urlencode(q, doseq=True)
    return urlunparse((parts.scheme, parts.netloc, parts.path, parts.params, query, parts.fragment))


def request_html(session, url, cfg, attempts=None):
    attempts = attempts or cfg["max_retries"]
    last = None
    for i in range(attempts):
        try:
            r = session.get(url, headers={"User-Agent": cfg["user_agent"]}, timeout=cfg["request_timeout_seconds"])
            text = r.text
            lower = text.lower()
            blocked = r.status_code in (403, 429) or any(m.lower() in lower for m in cfg.get("block_markers", []))
            if blocked:
                raise TemporaryBlock(f"possible throttling/block page: HTTP {r.status_code}")
            r.raise_for_status()
            return text
        except Exception as e:
            last = e
            wait = cfg["backoff_base_seconds"] * (i + 1) + random.random()
            if i + 1 < attempts:
                time.sleep(wait)
    raise last


def parse_forum_page(html: str, base_domain: str, min_replies: int = 1):
    soup = BeautifulSoup(html, "html.parser")
    threads = []
    for link in soup.select('a:has(span.topic_subject)'):
        subject = link.select_one('span.topic_subject')
        if not subject or not link.get('href'):
            continue
        title = subject.get_text(strip=True)
        full_url = urljoin(base_domain, link['href'])
        m = re.search(r'/topic/(\d+)', full_url)
        if not m:
            continue
        thread_id = m.group(1)
        row = link.parent
        while row and not row.select_one('span.topic_number'):
            row = row.parent
        reply_count = 0
        if row:
            n = row.select_one('span.topic_number')
            if n:
                mm = re.search(r'(\d+)', n.get_text(strip=True).replace(',', ''))
                reply_count = int(mm.group(1)) if mm else 0
        if reply_count >= min_replies:
            threads.append({"thread_id":thread_id,"thread_title":title,"thread_url":full_url,"reply_count":reply_count})
    return threads


def last_page_from_html(html: str) -> int:
    soup = BeautifulSoup(html, "html.parser")
    last = 1
    for a in soup.select('ul.pagination a.page-link[href]'):
        m = re.search(r'page=(\d+)', a.get('href',''))
        if m:
            last = max(last, int(m.group(1)))
    return last


def parse_topic_page(html: str, page: int, base_domain: str):
    soup = BeautifulSoup(html, "html.parser")
    posts = []
    for i, article in enumerate(soup.select('article.topic-post')):
        post_id = None
        if article.get('id'):
            m = re.search(r'\d+', article['id'])
            if m:
                post_id = m.group(0)
        if post_id is None:
            share = article.select_one('a[href*="postId="]')
            if share:
                m = re.search(r'postId=(\d+)', share.get('href',''))
                if m:
                    post_id = m.group(1)
        author, profile_url = '', ''
        for a in article.select('a.nickname'):
            if a.find_parent('blockquote'):
                continue
            name = a.select_one('span[itemprop="name"]')
            author = name.get_text(strip=True) if name else a.get_text(strip=True)
            profile_url = urljoin(base_domain, a.get('href',''))
            break
        date_span = article.select_one('span.date')
        time_span = article.select_one('span.time')
        posted_at = (date_span.get_text(strip=True) if date_span else '')
        if time_span:
            posted_at = (posted_at + ' ' + time_span.get_text(strip=True)).strip()
        user_post_count = None
        pc = article.select_one('div.post-count')
        if pc:
            m = re.search(r'(\d+)', pc.get_text(' ',strip=True).replace(',',''))
            if m:
                user_post_count = int(m.group(1))
        msg = article.select_one('div.post-message')
        content = ''
        if msg:
            # Work on a detached copy so quoted/repeated content is not counted as the current post.
            msg = BeautifulSoup(str(msg), 'html.parser')
            for node in msg.select('.topic-post__signature, blockquote'):
                node.decompose()
            content = msg.get_text('\n', strip=True)
        likes = 0
        like = article.select_one('a.like-count')
        if like:
            raw = like.get('data-like-count') or like.get_text(strip=True)
            m = re.search(r'(\d+)', str(raw).replace(',',''))
            likes = int(m.group(1)) if m else 0
        reply_to = None
        rd = article.select_one('div.reply-message[data-id]')
        if rd:
            reply_to = rd.get('data-id')
        posts.append({
            "post_id": post_id, "author": author, "profile_url": profile_url,
            "posted_at": posted_at, "content": content, "likes": likes,
            "reply_to": reply_to, "user_post_count": user_post_count,
            "is_starter": bool(i == 0 and page == 1)
        })
    return posts


def scrape_thread(session, thread, cfg):
    first_html = request_html(session, thread['thread_url'], cfg)
    first = parse_topic_page(first_html, 1, cfg['base_domain'])
    if not first:
        return []
    last = last_page_from_html(first_html)
    rows = list(first)
    starter_id = first[0].get('post_id')
    for p in range(2, last + 1):
        time.sleep(cfg['thread_page_delay_seconds'])
        html = request_html(session, with_page(thread['thread_url'], p), cfg)
        page_rows = parse_topic_page(html, p, cfg['base_domain'])
        if starter_id and page_rows and page_rows[0].get('post_id') == starter_id:
            page_rows = page_rows[1:]
        rows.extend(page_rows)
    # Thread-local robust dedup.
    seen, out = set(), []
    for r in rows:
        key = ('id', str(r['post_id'])) if r.get('post_id') else ('fallback', r.get('author',''), r.get('posted_at',''), r.get('content',''))
        if key in seen:
            continue
        seen.add(key)
        r.update({"thread_id":str(thread['thread_id']),"thread_title":thread.get('thread_title',''),"thread_url":thread['thread_url'],"section_name":thread.get('section_name','')})
        out.append(r)
    return out


def append_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    if exists:
        old = pd.read_csv(path, nrows=0)
        fieldnames = list(dict.fromkeys(list(old.columns) + fieldnames))
        # Re-write only if new columns appear.
        if list(old.columns) != fieldnames:
            full = pd.read_csv(path)
            full = pd.concat([full, pd.DataFrame(rows)], ignore_index=True)
            full.to_csv(path, index=False, encoding='utf-8-sig')
            return
    with path.open('a', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if not exists:
            w.writeheader()
        w.writerows(rows)


def load_done_status(path: Path):
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    if 'status' not in df or 'thread_id' not in df:
        return set()
    return set(df.loc[df['status'].eq('complete'), 'thread_id'].astype(str))


def write_status(path: Path, thread_id: str, status: str, post_count: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([{\
        'thread_id':str(thread_id), 'status':status, 'post_count':post_count,
        'updated_at':pd.Timestamp.utcnow().isoformat()
    }])
    if path.exists():
        df = pd.read_csv(path, dtype=str)
        df = df[df['thread_id'].astype(str) != str(thread_id)]
        df = pd.concat([df, row], ignore_index=True)
    else:
        df = row
    df.to_csv(path, index=False, encoding='utf-8-sig')


def collect_thread_manifest(sections: pd.DataFrame, cfg):
    session = requests.Session()
    all_threads = []
    for _, sec in sections.iterrows():
        if str(sec.get('section_name','')).startswith('#'):
            continue
        base = str(sec['forum_url']).strip()
        if 'REPLACE_ME' in base:
            continue
        start = int(sec.get('start_page',1)); requested_end = int(sec.get('end_page',cfg['max_forum_pages']))
        first_html = request_html(session, with_page(base, start), cfg)
        discovered_last = last_page_from_html(first_html)
        end = min(requested_end, discovered_last, cfg['max_forum_pages'])
        for p in range(start, end+1):
            if p != start:
                time.sleep(cfg['forum_page_delay_seconds'])
                html = request_html(session, with_page(base,p), cfg)
            else:
                html = first_html
            rows = parse_forum_page(html, cfg['base_domain'], cfg['min_replies'])
            for r in rows:
                r['section_name'] = sec.get('section_name','')
                r['forum_page'] = p
            all_threads.extend(rows)
    if not all_threads:
        return pd.DataFrame(columns=['thread_id','thread_title','thread_url','reply_count','section_name','forum_page'])
    df = pd.DataFrame(all_threads).drop_duplicates(subset=['thread_id','thread_url'])
    return df


def scrape_manifest_resumable(manifest: pd.DataFrame, cfg, project_root: Path):
    checkpoint = project_root / cfg['posts_checkpoint']
    status_file = project_root / cfg['thread_status']
    session = requests.Session()
    done = load_done_status(status_file)
    todo = manifest[~manifest['thread_id'].astype(str).isin(done)].to_dict('records')
    for t in todo:
        try:
            rows = scrape_thread(session, t, cfg)
            if rows:
                append_csv(checkpoint, rows)  # immediate per-thread durability
                write_status(status_file, str(t['thread_id']), 'complete', len(rows))
            else:
                write_status(status_file, str(t['thread_id']), 'empty_or_failed', 0)
        except Exception:
            write_status(status_file, str(t['thread_id']), 'empty_or_failed', 0)
        time.sleep(cfg['between_thread_delay_seconds'])
    return pd.read_csv(checkpoint, low_memory=False) if checkpoint.exists() else pd.DataFrame()


def unresolved_threads(manifest: pd.DataFrame, posts: pd.DataFrame):
    have = set(posts['thread_id'].astype(str)) if len(posts) and 'thread_id' in posts else set()
    return manifest[~manifest['thread_id'].astype(str).isin(have)].copy()


def recovery_pass(manifest: pd.DataFrame, cfg, project_root: Path):
    checkpoint = project_root / cfg['posts_checkpoint']
    posts = pd.read_csv(checkpoint, low_memory=False) if checkpoint.exists() else pd.DataFrame()
    miss = unresolved_threads(manifest, posts)
    if miss.empty:
        return posts, miss
    session = requests.Session()
    for t in miss.to_dict('records'):
        try:
            rows = scrape_thread(session, t, cfg)
            if rows:
                append_csv(checkpoint, rows)
                write_status(project_root/cfg['thread_status'], str(t['thread_id']), 'complete', len(rows))
        except Exception:
            pass
        time.sleep(max(cfg['between_thread_delay_seconds'], 5.0))
    posts = pd.read_csv(checkpoint, low_memory=False) if checkpoint.exists() else pd.DataFrame()
    return posts, unresolved_threads(manifest, posts)



def _class_tokens(soup):
    """Return normalized CSS class tokens present in the profile HTML."""
    tokens = set()
    for node in soup.find_all(True):
        classes = node.get("class") or []
        if isinstance(classes, str):
            classes = classes.split()
        tokens.update(str(c).strip().lower() for c in classes if str(c).strip())
    return tokens


def detect_gender_from_profile_html(html: str):
    """Detect Ninisite profile gender from explicit text/marker evidence.

    Historical issue:
    the site used different marker families than first assumed. In particular,
    female pages may expose an `is-woman`/woman marker, while male pages may use
    an `is-user`/user-style marker. Older icon classes also existed.

    Detection order:
    1. explicit nearby Persian gender text, when present;
    2. known female marker family;
    3. known male marker family;
    4. otherwise unresolved (-1).

    We never infer male merely from "not female".
    """
    soup = BeautifulSoup(html, "html.parser")
    tokens = _class_tokens(soup)

    female_tokens = {
        "is-woman", "iconwoman", "icon-woman", "woman", "female"
    }
    male_tokens = {
        "is-user", "iconuser-01", "iconuser-1", "iconuser", "icon-user",
        "iconman", "icon-man", "man", "male"
    }

    # Prefer explicit visible text associated with known gender/profile markers.
    candidates = soup.select(
        ".is-woman, .is-user, i.iconwoman, i.iconuser-01, i.iconuser-1, i.iconuser, "
        "i.iconman, [class*='woman'], [class*='user'], [class*='man']"
    )
    for node in candidates:
        local_text = node.get_text(" ", strip=True)
        nxt = node.find_next("span")
        if nxt:
            local_text = (local_text + " " + nxt.get_text(" ", strip=True)).strip()
        if "زن" in local_text:
            return "زن", 1, "explicit_text_female"
        if "مرد" in local_text:
            return "مرد", 0, "explicit_text_male"

    female_hit = bool(tokens & female_tokens)
    male_hit = bool(tokens & male_tokens)

    if female_hit and not male_hit:
        marker = sorted(tokens & female_tokens)[0]
        return "زن", 1, f"class_marker:{marker}"
    if male_hit and not female_hit:
        marker = sorted(tokens & male_tokens)[0]
        return "مرد", 0, f"class_marker:{marker}"

    # Ambiguous/no marker stays unresolved and will be eligible for a later
    # recovery pass in the scraping stage, not silently converted to a gender.
    return "unknown", -1, "unresolved"


def parse_profile_page(html: str):
    soup = BeautifulSoup(html, 'html.parser')
    info = {
        'username':'', 'gender':'unknown', 'gender_code':-1,
        'gender_detection_source':'unresolved',
        'join_date':'', 'age':'', 'education':'',
        'status':'', 'children_count':0, 'signature':''
    }

    name = soup.select_one('div.profile__name h4')
    if name:
        info['username'] = name.get_text(strip=True)

    g, code, source = detect_gender_from_profile_html(html)
    info['gender'] = g
    info['gender_code'] = code
    info['gender_detection_source'] = source

    for selector, field in [
        ('i.iconbirthday-stroke','age'),
        ('i.iconeducation','education'),
        ('i.iconinfant','status'),
        ('i.iconsignature','signature')
    ]:
        node = soup.select_one(selector)
        if node:
            sp = node.find_next('span')
            if sp:
                info[field] = sp.get_text(' ',strip=True)

    cal = soup.select_one('i.iconcalender')
    if cal:
        parent = cal.parent
        spans = parent.find_all('span', recursive=False) if parent else []
        for i, sp in enumerate(spans[:-1]):
            if 'عضویت' in sp.get_text():
                info['join_date'] = spans[i+1].get_text(strip=True)
                break
    return info


def _upsert_profile_checkpoint(path: Path, row: dict):
    """Persist one profile result by profile_url, replacing an older unresolved row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path, low_memory=False)
        if 'profile_url' in old.columns:
            old = old[old['profile_url'].astype(str) != str(row['profile_url'])]
        out = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        out = new
    out.to_csv(path, index=False, encoding='utf-8-sig')


def scrape_profiles_resumable(posts: pd.DataFrame, cfg, project_root: Path):
    """Scrape profile metadata with gender-aware recovery.

    A profile is considered complete for gender purposes only when gender_code is
    0 or 1. Unresolved profiles remain eligible for a future scraping recovery
    pass. This prevents the historical error from being frozen into the data.
    """
    path = project_root / cfg['profile_checkpoint']
    existing = pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

    if len(existing) and 'profile_url' in existing:
        codes = pd.to_numeric(
            existing.get('gender_code', pd.Series(-1, index=existing.index)),
            errors='coerce'
        )
        resolved = existing.loc[codes.isin([0, 1]), 'profile_url'].astype(str)
        done = set(resolved)
    else:
        done = set()

    urls = sorted(
        set(posts.get('profile_url', pd.Series(dtype=str)).dropna().astype(str))
        - done
    )

    session = requests.Session()
    for url in urls:
        if not url:
            continue
        try:
            html = request_html(session, url, cfg, attempts=3)
            info = parse_profile_page(html)
            info['profile_url'] = url
            _upsert_profile_checkpoint(path, info)
        except Exception:
            # Transient failures are not marked complete; rerun can retry.
            pass
        time.sleep(cfg['profile_delay_seconds'])

    return pd.read_csv(path, low_memory=False) if path.exists() else pd.DataFrame()

