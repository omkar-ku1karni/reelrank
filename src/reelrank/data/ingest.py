"""Download KuaiRec 2.0, verify it, and turn the raw CSVs into typed, de-duplicated Parquet tables.

    python -m reelrank.data.ingest download   # Zenodo zip -> data/raw (MD5 verified)
    python -m reelrank.data.ingest ingest     # CSV -> data/processed/*.parquet + qa_report.json

Grain of each output table:
  interactions/      one observed big-matrix event (date-partitioned Parquet, unique event_id)
  dense.parquet      one small-matrix (user, video) observation; ~3.9% have no timestamp
  items.parquet      one video: tags, first-level category, caption, creator, upload day, duration
  item_daily.parquet one video/day of platform-wide statistics (used lagged by >= 1 day)
  users.parquet      one user: static profile features
"""

from __future__ import annotations

import ast
import hashlib
import shutil
import sys
import urllib.request
import zipfile

import duckdb
import numpy as np
import pandas as pd

from reelrank.config import ROOT, TZ_OFFSET_S, Config, cli, write_json

# English glosses for the 41 first-level category names in kuairec_caption_category.csv (author's translation).
CATEGORY_EN = {
    "民生资讯": "Local news", "时尚": "Fashion", "其他": "Other", "颜值": "Looks", "情感": "Relationships",
    "美食": "Food", "明星娱乐": "Celebrities", "舞蹈": "Dance", "生活": "Lifestyle", "自拍": "Selfie",
    "喜剧": "Comedy", "亲子": "Parenting", "运动": "Sports", "艺术": "Art", "二次元": "Anime & comics",
    "三农": "Rural life", "宠物": "Pets", "摄影": "Photography", "音乐": "Music", "健康": "Health",
    "健身": "Fitness", "美妆": "Beauty", "短剧": "Short drama", "旅游": "Travel", "汽车": "Cars",
    "游戏": "Gaming", "科学与法律": "Science & law", "军事": "Military", "读书": "Books",
    "影视综": "Film & TV", "教育": "Education", "人文": "Humanities", "时政资讯": "Current affairs",
    "房产家居": "Home & property", "高新数码": "Tech & gadgets", "星座命理": "Astrology", "UNKNOWN": "Unknown",
    "财经": "Finance", "奇人异象": "Oddities", "娱乐八卦": "Gossip", "生活泛拍": "Everyday vlog",
}

DAILY_COLS = [
    "show_cnt", "play_cnt", "complete_play_cnt", "valid_play_cnt", "short_time_play_cnt", "like_cnt",
    "share_cnt", "comment_cnt", "follow_cnt", "report_cnt", "reduce_similar_cnt", "play_progress",
]


def download(cfg: Config) -> None:
    d = cfg["download"]
    zpath = ROOT / d["zip"]
    raw = cfg.p("raw")
    if (raw / "big_matrix.csv").exists():
        print(f"raw data already present at {raw}")
        return
    zpath.parent.mkdir(parents=True, exist_ok=True)
    if not zpath.exists():
        print(f"downloading {d['url']} (~432 MB)")
        tmp = zpath.with_suffix(".part")
        urllib.request.urlretrieve(d["url"], tmp)
        tmp.rename(zpath)
    md5 = hashlib.md5()
    with open(zpath, "rb") as f:
        while b := f.read(1 << 20):
            md5.update(b)
    if md5.hexdigest() != d["md5"]:
        sys.exit(f"MD5 mismatch for {zpath}: {md5.hexdigest()} != {d['md5']}")
    print("md5 ok; extracting")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(zpath.parent)
    if not (raw / "big_matrix.csv").exists():
        sys.exit(f"expected {raw}/big_matrix.csv after extraction")


def _matrix_sql(path: str, label: str) -> str:
    # The `date` column disagrees with `time` on ~0.1% of rows; the Unix timestamp is the source of truth.
    return f"""
        SELECT CAST(user_id AS INTEGER) user_id, CAST(video_id AS INTEGER) video_id,
               CAST(play_duration AS BIGINT) play_ms, CAST(video_duration AS BIGINT) duration_ms,
               CAST(timestamp AS DOUBLE) ts, CAST(watch_ratio AS DOUBLE) watch_ratio, '{label}' src,
               CAST(date AS BIGINT) date_col
        FROM read_csv('{path}', header=true, columns={{
            'user_id':'BIGINT','video_id':'BIGINT','play_duration':'BIGINT','video_duration':'BIGINT',
            'time':'VARCHAR','date':'DOUBLE','timestamp':'DOUBLE','watch_ratio':'DOUBLE'}})
    """


def ingest(cfg: Config) -> dict:
    raw = cfg.p("raw")
    out = cfg.processed
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    qa: dict = {"source": str(raw), "tables": {}}

    # ---- big matrix: QA, de-duplication, deterministic order --------------------------------------
    con.execute(f"CREATE TABLE big_raw AS {_matrix_sql(str(raw / 'big_matrix.csv'), 'big')}")
    q = con.execute("""
        SELECT count(*) n, count(DISTINCT user_id) users, count(DISTINCT video_id) videos,
               count(*) FILTER (WHERE ts IS NULL) null_ts,
               count(*) FILTER (WHERE play_ms < 0) negative_play,
               count(*) FILTER (WHERE duration_ms <= 0) nonpositive_duration,
               count(*) FILTER (WHERE user_id IS NULL OR video_id IS NULL) null_ids,
               count(*) FILTER (WHERE abs(watch_ratio - play_ms / duration_ms) > 1e-3 * greatest(watch_ratio, 1))
                   watch_ratio_disagree,
               count(*) FILTER (WHERE watch_ratio > 1) replay_rows
        FROM big_raw""").df().iloc[0].to_dict()
    date_disagree = con.execute(f"""
        SELECT count(*) FROM big_raw WHERE date_col IS NOT NULL AND ts IS NOT NULL AND
        CAST(strftime(to_timestamp(ts + {TZ_OFFSET_S}) AT TIME ZONE 'UTC', '%Y%m%d') AS BIGINT) <> date_col
    """).fetchone()[0]
    con.execute("ALTER TABLE big_raw DROP COLUMN date_col")
    n_exact = con.execute("SELECT count(*) FROM (SELECT DISTINCT * FROM big_raw)").fetchone()[0]
    dup_days = con.execute(f"""
        SELECT strftime(to_timestamp(ts + {TZ_OFFSET_S}) AT TIME ZONE 'UTC', '%Y-%m-%d') d, sum(n - 1) dups
        FROM (SELECT user_id, video_id, play_ms, duration_ms, ts, watch_ratio, count(*) n FROM big_raw
              GROUP BY ALL HAVING count(*) > 1) GROUP BY 1 ORDER BY 2 DESC LIMIT 3""").fetchall()
    # Exact duplicate rows are logging artefacts; rows that share (user, video, ts) but differ elsewhere are
    # collapsed to the one with the longest play so the event key is unique.
    con.execute("""
        CREATE TABLE big AS
        SELECT * EXCLUDE (rn) FROM (
            SELECT *, row_number() OVER (PARTITION BY user_id, video_id, ts ORDER BY play_ms DESC, duration_ms) rn
            FROM (SELECT DISTINCT * FROM big_raw)
        ) WHERE rn = 1 AND ts IS NOT NULL AND duration_ms > 0 AND play_ms >= 0
    """)
    n_clean = con.execute("SELECT count(*) FROM big").fetchone()[0]
    con.execute(f"""
        CREATE TABLE interactions AS
        SELECT row_number() OVER (ORDER BY user_id, ts, video_id, play_ms) - 1 AS event_id,
               user_id, video_id, ts,
               CAST(floor((ts + {TZ_OFFSET_S}) / 86400) AS INTEGER) AS day,
               strftime(to_timestamp(ts + {TZ_OFFSET_S}) AT TIME ZONE 'UTC', '%Y-%m-%d') AS date,
               play_ms, duration_ms, CAST(watch_ratio AS FLOAT) watch_ratio
        FROM big
    """)
    same_ts = con.execute("""
        SELECT count(*) FROM (SELECT user_id, ts FROM interactions GROUP BY 1, 2 HAVING count(*) > 1)""").fetchone()[0]
    repeats = con.execute("""
        SELECT coalesce(sum(n - 1), 0) FROM (SELECT user_id, video_id, count(*) n FROM interactions GROUP BY 1, 2)
    """).fetchone()[0]
    per_day = con.execute("SELECT date, count(*) n FROM interactions GROUP BY 1 ORDER BY 1").df()
    qa["tables"]["interactions"] = {
        "raw_rows": int(q["n"]), "users": int(q["users"]), "videos": int(q["videos"]),
        "exact_duplicate_rows_removed": int(q["n"] - n_exact),
        "same_key_conflicts_removed": int(n_exact - n_clean),
        "rows_after_cleaning": int(n_clean),
        "null_timestamp": int(q["null_ts"]), "negative_play": int(q["negative_play"]),
        "nonpositive_duration": int(q["nonpositive_duration"]), "null_ids": int(q["null_ids"]),
        "watch_ratio_disagrees_with_play_over_duration": int(q["watch_ratio_disagree"]),
        "watch_ratio_gt_1_share": float(q["replay_rows"] / q["n"]),
        "user_timestamp_groups_with_ties": int(same_ts),
        "date_column_disagrees_with_timestamp": int(date_disagree),
        "duplicate_heaviest_days": {d: int(n) for d, n in dup_days},
        "repeat_consumption_events": int(repeats),
        "events_per_day": dict(zip(per_day["date"], per_day["n"].astype(int))),
    }
    part = out / "interactions"
    if part.exists():
        shutil.rmtree(part)
    con.execute(f"""COPY (SELECT * FROM interactions ORDER BY user_id, ts, video_id, play_ms)
                    TO '{part}' (FORMAT parquet, PARTITION_BY (date), COMPRESSION zstd)""")

    # ---- small (dense) matrix -----------------------------------------------------------------------
    con.execute(f"CREATE TABLE small_raw AS {_matrix_sql(str(raw / 'small_matrix.csv'), 'small')}")
    sq = con.execute("""
        SELECT count(*) n, count(DISTINCT user_id) users, count(DISTINCT video_id) videos,
               count(*) FILTER (WHERE ts IS NULL) null_ts,
               count(DISTINCT (user_id, video_id)) pairs FROM small_raw""").df().iloc[0].to_dict()
    overlap_pairs = con.execute("""
        SELECT count(*) FROM (SELECT DISTINCT user_id, video_id FROM small_raw) s
        JOIN (SELECT DISTINCT user_id, video_id FROM interactions) b USING (user_id, video_id)""").fetchone()[0]
    con.execute(f"""
        CREATE TABLE dense AS
        SELECT user_id, video_id, ts,
               CAST(floor((ts + {TZ_OFFSET_S}) / 86400) AS INTEGER) AS day,
               play_ms, duration_ms, CAST(watch_ratio AS FLOAT) watch_ratio
        FROM small_raw ORDER BY user_id, video_id""")
    n_users, n_items = int(sq["users"]), int(sq["videos"])
    qa["tables"]["dense"] = {
        "rows": int(sq["n"]), "users": n_users, "videos": n_items, "pairs": int(sq["pairs"]),
        "density": float(sq["pairs"] / (n_users * n_items)), "null_timestamp": int(sq["null_ts"]),
        "pairs_also_in_big_matrix": int(overlap_pairs),
    }
    con.execute(f"COPY dense TO '{out / 'dense.parquet'}' (FORMAT parquet, COMPRESSION zstd)")

    # ---- items -------------------------------------------------------------------------------------
    cats = pd.read_csv(raw / "item_categories.csv")
    cats["tags"] = cats["feat"].map(lambda s: sorted(int(x) for x in ast.literal_eval(s)))
    cap = pd.read_csv(raw / "kuairec_caption_category.csv", dtype=str, keep_default_na=False)
    cap["video_id"] = pd.to_numeric(cap["video_id"], errors="coerce")
    malformed = int(cap["video_id"].isna().sum())
    cap = cap.dropna(subset=["video_id"]).astype({"video_id": int}).drop_duplicates("video_id")
    cap = cap[cap["video_id"].isin(cats["video_id"])]
    daily = pd.read_csv(raw / "item_daily_features.csv")
    daily["day"] = pd.to_datetime(daily["date"].astype(str)).map(lambda d: (d.date() - pd.Timestamp(0).date()).days)
    meta = (daily.sort_values(["video_id", "day"]).groupby("video_id")
            .agg(author_id=("author_id", "first"), upload_dt=("upload_dt", "first"),
                 duration_daily=("video_duration", "median"),
                 is_ad=("video_type", lambda s: bool((s == "AD").any())),
                 visible_status=("visible_status", "last")).reset_index())
    dur_obs = con.execute("SELECT video_id, median(duration_ms) d FROM interactions GROUP BY 1").df()
    items = cats[["video_id", "tags"]].merge(cap, on="video_id", how="left").merge(meta, on="video_id", how="left")
    items = items.merge(dur_obs, on="video_id", how="left")
    items["duration_ms"] = items["duration_daily"].fillna(items["d"]).fillna(items["d"].median()).astype(np.int64)
    items["first_cat_name"] = items["first_level_category_name"].replace("", "UNKNOWN").fillna("UNKNOWN")
    items["first_cat_en"] = items["first_cat_name"].map(CATEGORY_EN).fillna("Unknown")
    names = sorted(items["first_cat_name"].unique(), key=lambda s: (s == "UNKNOWN", s))
    items["first_cat"] = items["first_cat_name"].map({n: i for i, n in enumerate(names)}).astype(np.int16)
    items["second_cat_name"] = items["second_level_category_name"].replace("", "UNKNOWN").fillna("UNKNOWN")
    items["caption"] = items["caption"].fillna("")
    items["cover_text"] = items["manual_cover_text"].replace("UNKNOWN", "").fillna("")
    items["upload_day"] = pd.to_datetime(items["upload_dt"]).map(
        lambda d: (d.date() - pd.Timestamp(0).date()).days if pd.notna(d) else -1).astype(np.int32)
    items["author_id"] = items["author_id"].fillna(-1).astype(np.int64)
    items["is_ad"] = items["is_ad"].fillna(False).astype(bool)
    items = items[["video_id", "tags", "first_cat", "first_cat_name", "first_cat_en", "second_cat_name",
                   "caption", "cover_text", "author_id", "upload_day", "duration_ms", "is_ad",
                   "visible_status"]].sort_values("video_id").reset_index(drop=True)
    assert items["video_id"].is_unique and len(items) == len(cats)
    items.to_parquet(out / "items.parquet", index=False)
    pd.DataFrame({"first_cat": range(len(names)), "name": names,
                  "name_en": [CATEGORY_EN.get(n, "Unknown") for n in names]}).to_parquet(
        out / "categories.parquet", index=False)

    item_daily = daily[["video_id", "day"] + DAILY_COLS].copy()
    item_daily.to_parquet(out / "item_daily.parquet", index=False)
    unknown_videos = con.execute(
        "SELECT count(DISTINCT video_id) FROM interactions WHERE video_id NOT IN (SELECT video_id FROM items)").fetchone()[0]
    qa["tables"]["items"] = {
        "videos": len(items), "caption_rows_malformed_dropped": malformed,
        "videos_without_caption_row": int((~items["video_id"].isin(cap["video_id"])).sum()),
        "first_level_categories": len(names), "tags": int(max(max(t) for t in items["tags"]) + 1),
        "creators": int(items.loc[items["author_id"] >= 0, "author_id"].nunique()),
        "ads": int(items["is_ad"].sum()),
        "interaction_videos_missing_from_items": int(unknown_videos),
    }
    qa["tables"]["item_daily"] = {"rows": len(item_daily), "videos": int(item_daily["video_id"].nunique()),
                                  "first_day": int(item_daily["day"].min()), "last_day": int(item_daily["day"].max())}

    # ---- users -------------------------------------------------------------------------------------
    users = pd.read_csv(raw / "user_features.csv")
    keep = ["user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
            "follow_user_num", "fans_user_num", "friend_user_num", "register_days"]
    users = users[keep + [c for c in users.columns if c.startswith("onehot_feat")]]
    users["active_degree"] = users["user_active_degree"].map(
        {"high_active": 3, "full_active": 2, "middle_active": 1}).fillna(0).astype(np.int8)
    assert users["user_id"].is_unique
    users.to_parquet(out / "users.parquet", index=False)
    qa["tables"]["users"] = {"users": len(users)}

    write_json(out / "qa_report.json", qa)
    print_qa(qa)
    return qa


def print_qa(qa: dict) -> None:
    for name, t in qa["tables"].items():
        print(f"[{name}]")
        for k, v in t.items():
            if k != "events_per_day":
                print(f"  {k:48s} {v:,}" if isinstance(v, int) else f"  {k:48s} {v}")


def main() -> None:
    cfg, ns = cli(__doc__, [(("step",), {"choices": ["download", "ingest"]})])
    download(cfg) if ns.step == "download" else ingest(cfg)


if __name__ == "__main__":
    main()
