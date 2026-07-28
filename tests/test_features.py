from stock_trend.labels import make_labels


def test_make_labels():
    import pandas as pd

    close = pd.Series([100, 101, 102, 103, 104, 105, 106])
    labels = make_labels(close, forward_days=2)
    assert labels.iloc[0] == 1  # 102/100 - 1 > 0
    assert pd.isna(labels.iloc[-1])  # NaN at tail
