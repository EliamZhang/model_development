# Skill: 新增交易特征挖掘与接入

## 定位

这是一个用于银行流水交易特征挖掘的工程 skill。适用于新增高净值、高风险、高负债、特殊消费、收入稳定性、现金流压力、交易行为异常等特征模块。

核心目标不是“尽量多造变量”，而是让每一批新增特征都满足：

- 业务假设清晰：知道变量想捕捉什么风险或能力信号。
- 时间口径安全：只使用 `sample_datetime` 之前的交易，避免未来信息和贷后信息。
- 输出可对齐：代码输出、`dict.csv`、空交易兜底三者一致。
- 可验证可筛选：能用覆盖率、稳定性、相关性、异常值、性能等证据决定保留或剔除。

新增挖掘变量只接入变量回溯链路，默认不改模型打分链路。

## 两条链路

| 链路 | 入口 | 核心函数 | 输出 | 新增挖掘变量时 |
| --- | --- | --- | --- | --- |
| 模型分打分 | `model_main.py` -> `run_model.py` | `generate_score()` | score + 固定入模特征 | 不改 |
| 变量回溯 | `main_traceback.ipynb` -> `feature_traceback.py` -> `run_model_feature.py` | `run_model_feature()` | 按 `dict.csv` 固定列对齐的全部回溯特征 | 改 |

只有当某个新变量经过回溯验证并决定入模时，才单独修改 `run_model.py` 的 `model_vars` 并重新训练/验证模型。

## 先做特征评审

收到一个新方向后，先输出一张候选特征设计表，再写代码。

| 字段 | 要求 |
| --- | --- |
| `feature_group` | 对应代码里的方法名，如 `amount_by_type`、`ratio`、`velocity` |
| `feature_name` | 最终字段名，必须全小写 |
| `business_hypothesis` | 变量捕捉的业务信号 |
| `population` | 使用哪些交易，按 `tag_level1/tag_level2/category/third_party/dr_cr/account_type` 说明 |
| `time_window` | 使用哪些窗口，以及为什么需要这些窗口 |
| `formula` | 明确分子、分母、聚合方式和单位 |
| `expected_direction` | 风险方向：高更好、高更差、不确定 |
| `default_value` | 无数据/分母无效/缺失时的处理 |
| `leakage_check` | 是否只依赖申请时点前数据，是否包含贷后/结果/人工标签 |
| `priority` | `must_have` / `nice_to_have` / `reject` |

筛选原则：

- 优先做有业务解释的强信号：金额、频次、占比、趋势、间隔、连续性、波动、集中度、最近一次行为。
- 少做机械笛卡尔积。不要无脑展开 `所有实体 x 所有统计量 x 所有窗口`。
- 每个模块先控制在约 20-80 个候选变量，验证后再扩展。
- 窗口要有理由。默认窗口可用 `[7, 14, 28, 56, 84, 168, 182]`，但不是每个特征都必须全窗口展开。
- 高基数 `third_party` 只用稳定名单或明确业务名单，不根据单个样本动态生成特征名。
- 比率类变量必须定义分母无效时的行为；分母为 0 不要硬填 0。

## 泄漏与口径硬规则

实现前必须检查：

- 用 `sample_datetime` 作为观察时点，计算 `trac_days = sample_datetime - transaction_date`。
- 过滤 `trac_days < 0` 的未来交易。
- 不使用申请结果、放款结果、还款表现、逾期标签、模型分、人工审核结论等贷后/结果字段。
- 不用 `application_id`、`user_id` 本身派生业务特征；它们只作为 merge key。
- 对金额统一确认方向：交易模块通常使用 `amount.abs()`，方向由 `tag_level1/tag_level2/dr_cr` 决定。
- 使用交易类别映射表生成 `tag_level1/tag_level2`，不要复制一套分散的人工规则，除非业务特征明确需要。

## 工程改动范围

| 步骤 | 文件 | 动作 |
| --- | --- | --- |
| 1 | `v1_0_20260327/txn_tool/txn_{module}.py` | 新建特征工程脚本 |
| 2 | `v1_0_20260327/txn_tool/__init__.py` | 注册模块 import |
| 3 | `v1_0_20260327/run_model_feature.py` | 在 `data_prepare()` 接入新模块 |
| 4 | `dict.csv` | 追加新特征行，作为回溯输出列权威 |
| 5 | `run_demo.py` / `test_run.ipynb` | 可选，仅做展示或批量 smoke test |

不改：

- `run_model.py`：打分链路，不因探索变量而改。
- `model_main.py`：薄封装，不改。
- `score_traceback.py`：打分回溯，不改。
- `feature_traceback.py`：通过 `run_func` 动态传入，通常不改。
- `main_traceback.ipynb`：入口不变，除非用户明确要求改 notebook。

## 模块命名规范

| 项目 | 约定 | 示例 |
| --- | --- | --- |
| 文件名 | `txn_{module}.py` | `txn_cashflow_pressure.py` |
| 类名 | `SingleApplication{Module}FeatureEngineer` | `SingleApplicationCashflowPressureFeatureEngineer` |
| 入口函数 | `generate_{module}_feature(df, feature_groups=None)` | `generate_cashflow_pressure_feature(df, feature_groups=None)` |
| 特征名前缀 | `bank_txn_{abbrev}_` | `bank_txn_cfp_` |
| 兜底字典 | `model_vars_{module}` | `model_vars_cashflow_pressure` |
| `dict.csv` 分组 | `model_vars_{module}` | `model_vars_cashflow_pressure` |

特征命名格式：

```text
bank_txn_{abbrev}_{population}_{metric}_{stat}_{window}d
bank_txn_{abbrev}_{population}_{metric}_{comparison}
```

命名要求：

- 全小写。
- 不含空格。
- `third_party/category` 进入字段名前必须清洗。
- 同一统计口径只保留一种命名，不同时出现 `count/cnt`、`amount/amt` 混用。
- 新模块内尽量使用短前缀，避免字段过长。

## 新建特征脚本

优先参考 `v1_0_20260327/txn_tool/txn_income_v1_1.py` 或 `txn_expense.py` 的结构，保留本仓库已有模式。

脚本必须包含：

```python
class SingleApplication{Module}FeatureEngineer:
    def __init__(self, df, time_windows=None, ...):
        self.original_df = df.copy()
        self.time_windows = sorted(time_windows) if time_windows else [7, 14, 28, 56, 84, 168, 182]
        self.features = {}
        self.df = self._map(df)
        self.raw_df = self.df.copy()
        self._prepare_data()

    def _map(self, df, mapping_file=None):
        # 复用现有模块的映射表逻辑：account_type/category/dr_cr -> tag_level1/tag_level2
        ...

    def _prepare_data(self):
        # 生成 date、sample_datetime、trac_days、month_num
        # amount 转数值并按业务需要取 abs
        # 过滤 trac_days >= 0
        # 构建 window/type/entity 缓存
        ...

    def generate_all_features(self, df=None, feature_groups=None, include_metadata=True):
        ...


def generate_{module}_feature(df: pd.DataFrame, feature_groups: List[str] = None):
    engineer = SingleApplication{Module}FeatureEngineer(
        df=df,
        time_windows=[7, 14, 28, 56, 84, 168, 182],
    )
    return engineer.generate_all_features(df=df, feature_groups=feature_groups)
```

实现要求：

- 每个 `feature_group` 返回 `Dict`，并 `self.features.update(features)`。
- `generate_all_features()` 输出一行 DataFrame，包含 `user_id`、`sample_datetime`。
- 输出列统一 `.str.strip().str.replace(" ", "", regex=False).str.lower()`。
- 对不存在的 feature group 给出警告或显式报错，避免静默漏算。
- 优先用预聚合缓存，避免在每个变量里重复过滤和 `groupby`。
- 大量实体特征用配置列表控制，不在运行时动态增加未知列。

默认值建议：

| 场景 | 变量原始值 | 进入 `run_model_feature` 后 |
| --- | --- | --- |
| count/sum 且无匹配交易 | `0` | 保持 `0` |
| mean/median/max/min/std/cv 且无匹配交易 | `np.nan` | 由 `fillna_clip_numeric()` 填成模块默认值 |
| ratio 分母无效 | `np.nan` | 由 `fillna_clip_numeric()` 填成模块默认值 |
| lender 类历史口径 | 按现有 lender 规则 | 通常填 `-1` |
| 普通交易探索变量 | `np.nan` 或数值 | 通常填 `-999999.99999` |

## 注册模块

修改 `v1_0_20260327/txn_tool/__init__.py`，只追加新模块 import：

```python
from . import txn_{module}
```

不要顺手整理 unrelated import，除非本次任务明确要求。当前文件里 `txn_lender` 已有重复 import，新增模块时不需要扩大改动面。

## 接入 `run_model_feature.py`

### 1. 新增兜底字典

在现有 `model_vars_xxx` 后追加：

```python
model_vars_{module} = {
    "bank_txn_{abbrev}_xxx_sum_28d": [-999999.99999],
}
```

说明：

- `dict.csv` 是最终回溯列权威。
- `model_vars_{module}` 用于 `raw_data` 为空但仍需要返回模块默认列的场景。
- 如果这里留空，而 `dict.csv` 有新特征，那么完全空输入时仍会由 `run_model_feature()` 的 `DEFAULT_MAP` 对齐；但交易空、余额非空等走 `data_prepare()` 的场景会更依赖兜底字典，建议同步维护。

### 2. 正常交易分支接入

在 `expense/surplus` 后、`lender` 前新增：

```python
df_txn_{module} = txn_tool.txn_{module}.generate_{module}_feature(
    df=raw_data,
    feature_groups=None,
)
df_txn_{module}["user_id"] = user_id
df_txn_{module}["sample_datetime"] = pd.to_datetime(send_time)
df_txn_{module} = fillna_clip_numeric(df_txn_{module}, -999999.99999)
```

### 3. 交易空兜底分支接入

在 `raw_data is None or raw_data.shape[0] == 0` 分支中新增：

```python
df_txn_{module} = build_default_df(model_vars_{module}, -999999.99999)
```

### 4. merge 列表接入

把新模块加入 merge 列表：

```python
for df in [
    df_txn_cate,
    df_txn_eod,
    df_txn_income,
    df_txn_expense,
    df_txn_surplus,
    df_txn_{module},
    df_txn_lender,
]:
```

## 更新 `dict.csv`

每个新特征追加一行：

```csv
feature,model_vars,default
bank_txn_{abbrev}_xxx_sum_28d,model_vars_{module},-999999.99999
```

规则：

- 第一列必须与代码输出完全一致。
- 第二列必须与 `run_model_feature.py` 的字典名一致。
- 第三列是回溯固定列缺失时的最终填充值。
- 不把 `user_id/application_id/sample_datetime/send_time/txn_raw_input_cnt` 等元信息写进 `dict.csv`。

追加后必须做 dict 对齐检查：

```python
import pandas as pd

dict_df = pd.read_csv("dict.csv")
assert dict_df["feature"].is_unique
assert dict_df["feature"].str.lower().equals(dict_df["feature"])
assert dict_df["feature"].str.contains(" ").sum() == 0
```

## 验证清单

### 工程验证

必须至少验证：

```bash
python -c "from v1_0_20260327.txn_tool import txn_{module}; print('OK')"
```

```bash
python -c "from v1_0_20260327.txn_tool.txn_{module} import generate_{module}_feature; import pandas as pd; df = pd.DataFrame({'user_id':[1], 'sample_datetime':['2025-06-01'], 'transaction_date':['2025-05-20'], 'amount':[100], 'balance':[1000], 'dr_cr':['credit'], 'category':['Wages'], 'third_party':['Employer'], 'account_type':['TX']}); out = generate_{module}_feature(df); print(out.shape); print(out.columns[:5].tolist())"
```

```bash
python -c "from v1_0_20260327.run_model_feature import run_model_feature; result = run_model_feature({'userId':1, 'applicationId':1, 'flowTime':'2025-06-01 12:00:00.0', 'bank_accounts':[], 'illion_raw_transactions':[], 'illion_day_end_balances':[]}); print(type(result), len(result)); print(result.get('bank_txn_{abbrev}_xxx_sum_28d'))"
```

### 数据质量验证

如果有样本回溯结果，必须输出以下 profiling：

- 覆盖率：非默认、非空占比。
- 零值率：对 count/sum 特征尤其重要。
- 分位数：`min/p1/p5/p50/p95/p99/max`。
- 唯一值数：剔除常数列或近常数列。
- 极端值：检查 clip/default 是否吞掉真实值。
- 相关性：同模块内 `abs(corr) > 0.98` 的高度重复变量只保留业务更清晰的一个。
- 窗口冗余：相邻窗口表现几乎一致时，优先保留更有业务解释的窗口。
- 性能：单样本耗时明显增加时，回到缓存和 groupby 逻辑优化。

建议产出一张评审表：

| feature | coverage | zero_rate | p50 | p95 | p99 | corr_max | decision | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

`decision` 只能是：

- `keep`：保留进 `dict.csv`。
- `drop`：不接入回溯。
- `revise`：调整口径后再验证。

## 常见特征族

优先从这些特征族里选，不要一次全做。

| 特征族 | 适合场景 | 示例 |
| --- | --- | --- |
| 金额规模 | 收入、支出、负债、博彩、转账 | `sum/mean/max/min` |
| 频次 | 高频小额、异常活跃、短期压力 | `count/daily_count/max_daily_count` |
| 占比 | 结构变化、依赖度、集中度 | `type_sum / total_sum` |
| 最近行为 | 临近申请前异常 | `latest_amount/latest_days_since` |
| 趋势 | 收入下降、支出上升、负债加速 | `1m_vs_3m/trend_slope` |
| 波动 | 不稳定现金流 | `std/cv/max_min_gap` |
| 间隔 | 工资周期、还款周期、博彩频率 | `interval_mean/std/max` |
| 连续性 | 持续收入、持续负债、连续消费 | `max_consecutive_days/months` |
| 集中度 | 单一商户/类别依赖 | `top1_share/top3_share/hhi` |

## 输出格式

完成一次新增特征后，回复中必须包含：

- 新增/修改了哪些文件。
- 新增了多少个代码输出特征，多少个写入 `dict.csv`。
- 哪些特征被保留、剔除或待验证。
- 运行了哪些验证命令，结果如何。
- 是否改动打分链路；默认应为“未改动”。
