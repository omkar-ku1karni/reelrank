import polars as pl

from reelrank.ranking.dataset import sample_train_groups


def test_sampling_keeps_whole_user_days_and_respects_cap():
    df = pl.DataFrame({"user_id": [u for u in range(50) for _ in range(30)],
                       "day": [d % 3 for d in range(1500)], "video_id": list(range(1500))})
    out = sample_train_groups(df, 600, seed=1)
    assert out.height <= 600
    sizes = out.group_by(["user_id", "day"]).len()
    full = df.group_by(["user_id", "day"]).len()
    assert sizes.join(full, on=["user_id", "day"]).filter(pl.col("len") != pl.col("len_right")).height == 0
    assert sample_train_groups(df, 600, seed=1).equals(out)  # deterministic
    assert sample_train_groups(df, None, seed=1).height == 1500
