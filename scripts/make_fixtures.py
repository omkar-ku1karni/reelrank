"""Generate a tiny synthetic dataset with KuaiRec 2.0's exact file layout and quirks.

    python scripts/make_fixtures.py [out_dir]     (default data/fixture/raw)

Planted structure: users prefer a few categories, short videos get higher watch ratios, fresh uploads trend,
and the same three-burst calendar as the real big matrix. Planted quirks: exact duplicate rows, small-matrix
rows without timestamps, a malformed caption row, and no (user, video) pair shared between the matrices.
Nothing here is real KuaiRec data.
"""

from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BURSTS = [("2020-07-05", 8), ("2020-08-01", 10), ("2020-08-27", 10)]
CATS = ["美食", "舞蹈", "宠物", "游戏", "旅游", "音乐", "运动", "喜剧"]


def main(out: Path, n_users: int = 90, n_items: int = 360, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    n_cat = len(CATS)
    item_cat = rng.integers(0, n_cat, n_items)
    item_dur = np.round(np.exp(rng.normal(2.4, 0.6, n_items)) * 1000).astype(int) + 2000
    item_author = rng.integers(0, n_items // 3, n_items) + 1000
    day0 = dt.date(2020, 7, 5)
    # upload days spread over the calendar so later bursts see fresh content
    upload = np.array([day0 + dt.timedelta(days=int(d)) for d in rng.integers(-30, 60, n_items)])
    upload[: n_items // 2] = day0 - dt.timedelta(days=10)
    user_pref = rng.dirichlet(np.ones(n_cat) * 0.3, n_users)
    quality = rng.normal(0, 0.3, n_items)

    dense_users = np.arange(0, 24)
    dense_items = np.arange(n_items - 60, n_items)
    days = []
    for start, n in BURSTS:
        s = dt.date.fromisoformat(start)
        days += [s + dt.timedelta(days=i) for i in range(n)]
    big_rows = []
    for u in range(n_users):
        for d in days:
            avail = np.where(upload <= d)[0]
            avail = avail[~(np.isin(u, dense_users) & np.isin(avail, dense_items))]
            fresh = np.exp(-np.array([(d - upload[i]).days for i in avail]) / 10.0)
            w = user_pref[u, item_cat[avail]] * (0.3 + fresh) + 1e-3
            k = rng.integers(4, 12)
            pick = rng.choice(avail, size=min(k, len(avail)), replace=False, p=w / w.sum())
            for i in pick:
                aff = user_pref[u, item_cat[i]] * n_cat
                mu = 0.2 + 0.6 * aff + quality[i] + 0.8 * (10000 / item_dur[i] - 0.6)
                wr = float(np.clip(rng.lognormal(np.log(max(mu, 0.05)), 0.5), 0.01, 8.0))
                ts = dt.datetime.combine(d, dt.time()) + dt.timedelta(seconds=int(rng.integers(0, 86000)))
                epoch = (ts - dt.timedelta(hours=8)).replace(tzinfo=dt.UTC).timestamp() + rng.random()
                big_rows.append([u, int(i), int(wr * item_dur[i]), int(item_dur[i]),
                                 ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3], int(d.strftime("%Y%m%d")),
                                 round(epoch, 3), wr])
    big_rows += [list(r) for r in rng.choice(np.array(big_rows, dtype=object), 40, replace=False)]  # exact duplicates
    hdr = ["user_id", "video_id", "play_duration", "video_duration", "time", "date", "timestamp", "watch_ratio"]
    with open(out / "big_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(big_rows)
    with open(out / "small_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        for u in dense_users:
            for i in dense_items:
                aff = user_pref[u, item_cat[i]] * n_cat
                mu = 0.2 + 0.6 * aff + quality[i] + 0.8 * (10000 / item_dur[i] - 0.6)
                wr = float(np.clip(rng.lognormal(np.log(max(mu, 0.05)), 0.5), 0.01, 8.0))
                if rng.random() < 0.05:
                    w.writerow([u, i, int(wr * item_dur[i]), item_dur[i], "", "", "", wr])
                    continue
                d = days[int(rng.integers(0, len(days)))]
                ts = dt.datetime.combine(d, dt.time()) + dt.timedelta(seconds=int(rng.integers(0, 86000)))
                epoch = (ts - dt.timedelta(hours=8)).replace(tzinfo=dt.UTC).timestamp()
                w.writerow([u, i, int(wr * item_dur[i]), item_dur[i], ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                            float(d.strftime("%Y%m%d")), round(epoch, 3), wr])
    with open(out / "item_categories.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "feat"])
        for i in range(n_items):
            tags = sorted({int(item_cat[i]), int(rng.integers(0, 12))})
            w.writerow([i, str(tags)])
    with open(out / "kuairec_caption_category.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "manual_cover_text", "caption", "topic_tag", "first_level_category_id",
                    "first_level_category_name", "second_level_category_id", "second_level_category_name",
                    "third_level_category_id", "third_level_category_name"])
        for i in range(n_items):
            w.writerow([i, "UNKNOWN", f"合成视频 {i}", "[]", int(item_cat[i]), CATS[item_cat[i]], -124, "UNKNOWN", -124,
                        "UNKNOWN"])
        w.writerow(["！”", "", "", "", "", "", "", "", "", ""])  # malformed row, as in the real file
    with open(out / "item_daily_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["video_id", "date", "author_id", "video_type", "upload_dt", "upload_type", "visible_status",
                "video_duration", "video_width", "video_height", "music_id", "video_tag_id", "video_tag_name",
                "show_cnt", "show_user_num", "play_cnt", "play_user_num", "play_duration", "complete_play_cnt",
                "complete_play_user_num", "valid_play_cnt", "valid_play_user_num", "long_time_play_cnt",
                "long_time_play_user_num", "short_time_play_cnt", "short_time_play_user_num", "play_progress",
                "comment_stay_duration", "like_cnt", "like_user_num", "click_like_cnt", "double_click_cnt",
                "cancel_like_cnt", "cancel_like_user_num", "comment_cnt", "comment_user_num", "direct_comment_cnt",
                "reply_comment_cnt", "delete_comment_cnt", "delete_comment_user_num", "comment_like_cnt",
                "comment_like_user_num", "follow_cnt", "follow_user_num", "cancel_follow_cnt",
                "cancel_follow_user_num", "share_cnt", "share_user_num", "download_cnt", "download_user_num",
                "report_cnt", "report_user_num", "reduce_similar_cnt", "reduce_similar_user_num", "collect_cnt",
                "collect_user_num", "cancel_collect_cnt", "cancel_collect_user_num"]
        w.writerow(cols)
        for i in range(n_items):
            for d in [day0 + dt.timedelta(days=k) for k in range(0, 63, 3)]:
                if d < upload[i]:
                    continue
                show = int(rng.integers(100, 5000))
                play = int(show * rng.uniform(0.5, 1.0))
                row = {c: 0 for c in cols}
                row.update(video_id=i, date=d.strftime("%Y%m%d"), author_id=int(item_author[i]),
                           video_type="AD" if i == 5 else "NORMAL", upload_dt=upload[i].isoformat(),
                           upload_type="ShortImport", visible_status="public", video_duration=float(item_dur[i]),
                           video_width=720, video_height=1280, music_id=1, video_tag_id=int(item_cat[i]),
                           video_tag_name=CATS[item_cat[i]], show_cnt=show, play_cnt=play,
                           complete_play_cnt=int(play * rng.uniform(0.2, 0.6)), like_cnt=int(play * 0.05),
                           share_cnt=int(play * 0.01), play_progress=round(float(rng.uniform(0.3, 0.9)), 4))
                w.writerow([row[c] for c in cols])
    with open(out / "user_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        cols = ["user_id", "user_active_degree", "is_lowactive_period", "is_live_streamer", "is_video_author",
                "follow_user_num", "follow_user_num_range", "fans_user_num", "fans_user_num_range", "friend_user_num",
                "friend_user_num_range", "register_days", "register_days_range"] + [f"onehot_feat{i}" for i in range(18)]
        w.writerow(cols)
        for u in range(n_users):
            w.writerow([u, ["high_active", "full_active", "middle_active", "UNKNOWN"][u % 4], 0, 0, u % 5 == 0,
                        int(rng.integers(0, 300)), "(0,10]", int(rng.integers(0, 50)), "0", 0, "0",
                        int(rng.integers(20, 900)), "61-90"] + [0] * 18)
    (out / "social_network.csv").write_text("user_id,friend_list\n0,[1]\n")
    print(f"fixture written to {out}: {len(big_rows):,} big-matrix rows")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "fixture" / "raw")
