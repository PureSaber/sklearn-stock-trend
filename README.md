# Scikit-Learn Stock Trend

使用传统机器学习模型预测股票涨跌：构建技术指标特征，定义未来 N 日涨跌标签，用随机森林等模型训练与评估。

> 参考思路：[PureSaber/Stock-Trend-Forecast-SKL](https://github.com/PureSaber/Stock-Trend-Forecast-SKL)

## 技术栈

Python · Scikit-learn · XGBoost · Pandas · AKShare · Matplotlib

## 目录结构

```
sklearn-stock-trend/
├── configs/
│   ├── default.yaml
│   ├── xgboost.yaml
│   ├── walkforward_tuned.yaml
│   ├── multisymbol.yaml
│   ├── multisymbol_xgb.yaml
│   ├── multisymbol_ensemble.yaml
│   ├── ensemble.yaml
│   ├── walkforward_xgb.yaml
│   ├── symbols_top.yaml
│   ├── symbols.yaml
│   └── experiments/
├── src/stock_trend/
│   ├── data.py
│   ├── features.py
│   ├── labels.py
│   ├── dataset.py
│   ├── tuning.py
│   ├── models.py
│   ├── ensemble.py
│   ├── symbol_tuning.py
│   ├── train.py
│   ├── evaluate.py
│   ├── backtest.py
│   ├── walkforward.py
│   ├── experiment.py
│   ├── multisymbol.py
│   └── multisymbol_walkforward.py
├── .github/workflows/ci.yml
├── notebooks/
│   └── 01_eda_backtest.ipynb
├── tests/
└── outputs/
```

## 快速开始

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -e ".[dev]"

# CLI 入口（安装后可用）
st-train --config configs/default.yaml
st-evaluate --config configs/default.yaml
st-backtest --config configs/default.yaml
st-walkforward --config configs/default.yaml

# 等价 python -m 形式
python -m stock_trend.train --config configs/default.yaml
python -m stock_trend.evaluate --config configs/default.yaml
python -m stock_trend.backtest --config configs/default.yaml

# Walk-forward 滚动 OOS 评估
python -m stock_trend.walkforward --config configs/default.yaml

# Walk-forward + 每折调参
python -m stock_trend.walkforward --config configs/walkforward_tuned.yaml

# XGBoost 训练
python -m stock_trend.train --config configs/xgboost.yaml

# 多标的泛化验证（RF / XGBoost）
python -m stock_trend.multisymbol --config configs/multisymbol.yaml --symbols configs/symbols.yaml
python -m stock_trend.multisymbol --config configs/multisymbol_xgb.yaml --symbols configs/symbols.yaml

# RF + XGBoost 概率集成 + 按标的调 proba 阈值
python -m stock_trend.train --config configs/ensemble.yaml
python -m stock_trend.backtest --config configs/ensemble.yaml

# 多标的 + 集成 + 按标的调参
python -m stock_trend.multisymbol --config configs/multisymbol_ensemble.yaml --symbols configs/symbols_top.yaml

# 亮眼标的 walk-forward 批量验证（XGB + 每折调参）
python -m stock_trend.multisymbol_walkforward --config configs/walkforward_xgb.yaml --symbols configs/symbols_top.yaml

# 批量对比实验
python -m stock_trend.experiment --configs-dir configs/experiments
pytest -q
```

## 新增功能

| 功能 | 说明 |
|------|------|
| `st-*` CLI | `st-train`, `st-backtest`, `st-evaluate`, `st-walkforward`, `st-multisymbol`, `st-experiment` |
| `signals.export` | 导出 `outputs/proba_signals.parquet`（date, symbol, proba_up, signal） |
| HTML 报告 | 回测后生成 `outputs/latest/report.html` |
| `backtest.retail_mode` | A 股散户约束：1 万本金、100 股整手、最低佣金、印花税、T+1 |
| `configs/run_retail_10k.yaml` | 散户 1 万回测示例配置 |
| `tuning.cv_mode: purged` | Purged + Embargo 时序 CV，减少标签重叠泄露 |
| TaskSolver | `goals/sklearn-complete.yaml` 自动化验收（见 `AGENTS.md`） |

## 配置要点

| 配置块 | 作用 |
|--------|------|
| `label.threshold` | 未来 N 日收益超过阈值才标为涨 |
| `tuning.enabled` | 训练集上 CV 自动搜参 |
| `tuning.cv_mode` | `timeseries` 或 `purged`（推荐，配合 `label_horizon`） |
| `signals.export` | 是否导出概率信号 Parquet |
| `backtest.retail_mode` | 启用离散整手 + T+1 散户回测 |
| `backtest.proba_threshold` | 概率信号阈值（实验最优约 `0.65`） |
| `backtest.commission_rate` | 单边佣金率（如 `0.0003`） |
| `backtest.slippage` | 单边滑点（如 `0.0001`） |
| `model.type` | `random_forest`、`xgboost` 或 `ensemble`（RF+XGB 概率平均） |
| `symbol_tuning.tune_proba_threshold` | 训练集尾部切验证集，按 F1 选最优 `proba_threshold` |
| `walkforward.tune_per_fold` | 每折 walk-forward 内嵌 TimeSeriesSplit 调参 |
| `configs/symbols.yaml` | 多标的列表 |

训练后查看 `outputs/feature_importance.csv` 了解各特征贡献。

## 学习目标

- 监督学习在金融中的应用流程
- 特征工程与过拟合防范（时序 CV、正则化）
- 模型评估与时间序列交叉验证
- 预测信号 → 策略回测 → 与基准对比
- Walk-forward 滚动样本外验证与多标的泛化测试

## 相关仓库

- [a-share-multifactor](../a-share-multifactor) — 多因子选股
- [currency-converter](../currency-converter) — 辅助工具
- [quant-research-notes](../quant-research-notes) — 学习路线图

