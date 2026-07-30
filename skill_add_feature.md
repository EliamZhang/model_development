# Skill: 新增交易特征挖掘与接入

## 定位

这是一个用于银行流水交易特征挖掘的工程 skill。适用于新增高净值、高风险、高负债、特殊消费、收入稳定性、现金流压力、交易行为异常等特征模块。

核心目标不是"尽量多造变量"，而是让每一批新增特征都满足：

- 业务假设清晰：知道变量想捕捉什么风险或能力信号。
- 时间口径安全：只使用 `sample_datetime` 之前的交易，避免未来信息和贷后信息。
- 输出可对齐：代码输出、`dict.csv`、空交易兜底三者一致。
- 可验证可筛选：能用覆盖率、稳定性、相关性、异常值、性能等证据决定保留或剔除。

新增挖掘变量只接入变量回溯链路，默认不改模型打分链路。

## 核心规则：每一步都要确认

**每完成一个步骤，必须暂停，输出当前步骤做了什么、结果是什么，然后等用户确认后再继续下一步。**

不要连续执行多个步骤。不要因为"下一步很简单"就跳过确认。即使用户说"直接做"，也要在每个有实际代码变更的步骤完成后停下来汇报。

确认格式：

```
---
### 步骤 N 完成：[步骤名称]

**做了什么：**
- 具体操作 1
- 具体操作 2

**当前状态：**
- 变更了哪些文件
- 有无报错

**下一步：** [下一步做什么]

是否继续？
---
```

## 两条链路

| 链路 | 入口 | 核心函数 | 输出 | 新增挖掘变量时 |
| --- | --- | --- | --- | --- |
| 模型分打分 | `model_main.py` -> `run_model.py` | `generate_score()` | score + 固定入模特征 | 不改 |
| 变量回溯 | `main_traceback.ipynb` -> `feature_traceback.py` -> `run_model_feature.py` | `run_model_feature()` | 按 `dict.csv` 固定列对齐的全部回溯特征 | 改 |

只有当某个新变量经过回溯验证并决定入模时，才单独修改 `run_model.py` 的 `model_vars` 并重新训练/验证模型。

---

## 阶段一：需求理解与特征评审（先讨论清楚，再写代码）

### 步骤 0：理解需求方向

收到一个新方向后，**不写代码，先复述需求**：

- 这个特征方向想捕捉什么业务信号？（风险识别 / 还款能力 / 收入稳定性 / 消费能力 / ...）
- 预期什么人群会有差异？（高收入 vs 低收入、稳定收入 vs 不稳定收入、高频交易 vs 低频交易...）
- 是否已有类似特征？和现有模块（income、expense、lender、surplus、category、balance）的关系是什么？

> **确认点 0：用自己的话复述需求方向，确认理解正确。如果有不确定的地方，提出来让用户澄清。等用户确认方向后再进入设计。**

### 步骤 1：调研现有数据

在写设计表之前，先快速调研数据基础：

- 相关的 `tag_level1` / `tag_level2` 在数据中是否存在、覆盖率和频次如何。
- 相关的 `third_party` 名单是否稳定，是否需要新建 reference 文件。
- 相关的 `category` / `account_type` 是否有足够的区分度。
- 是否依赖 balance 数据、eod 数据还是纯交易数据。

> **确认点 1：汇报数据调研结果。如果某个数据源覆盖率极低（如 < 1% 样本有该类型交易），提醒用户这个方向可能产出大量默认值。等用户确认后再进入特征设计。**

### 步骤 2：输出候选特征设计表

输出一张候选特征设计表。表中每行对应一个候选特征：

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

> **确认点 2：输出设计表后暂停，等用户确认特征列表、修改或增减后再继续。**

### 步骤 3：确认模块命名与字段前缀

在设计表确认后，确定模块的技术命名：

- 文件名：`txn_{module}.py`
- 类名：`SingleApplication{Module}FeatureEngineer`
- 入口函数：`generate_{module}_feature`
- 特征名前缀：`bank_txn_{abbrev}_`
- 兜底字典名：`model_vars_{module}`
- `dict.csv` 分组名：`model_vars_{module}`

> **确认点 3：输出模块命名方案，让用户确认。命名一旦确定，后面不要改，否则 dict.csv、兜底字典、merge 代码全部要同步修改。**

### 步骤 4：泄漏与口径检查

实现前逐条检查每个候选特征：

- 用 `sample_datetime` 作为观察时点，计算 `trac_days = sample_datetime - transaction_date`。
- 过滤 `trac_days < 0` 的未来交易。
- 不使用申请结果、放款结果、还款表现、逾期标签、模型分、人工审核结论等贷后/结果字段。
- 不用 `application_id`、`user_id` 本身派生业务特征；它们只作为 merge key。
- 对金额统一确认方向：交易模块通常使用 `amount.abs()`，方向由 `tag_level1/tag_level2/dr_cr` 决定。
- 使用交易类别映射表生成 `tag_level1/tag_level2`，不要复制一套分散的人工规则，除非业务特征明确需要。
- 时间窗口上限不超过回溯数据范围；如果回溯数据只覆盖 6 个月，182d 窗口对大多数样本无意义。

> **确认点 4：逐条汇报泄漏检查结果，每个候选特征一行。有风险的点标出并说明为什么是风险。等用户确认无泄漏风险后，进入工程实现阶段的大门。**

### 步骤 5：确认是否进入工程实现

这是阶段一到阶段二的**大门确认**：

- 汇总阶段一的全部结论：需求方向、候选特征数、模块命名、泄漏检查结果。
- 明确接下来要新建/修改哪些文件。
- 用户确认后，才开始写第一行代码。

> **确认点 5：汇总阶段一全部结论，明确工程改动的文件清单。等用户说"开始写代码"后，进入阶段二。**

---

## 阶段二：工程实现（逐步落地，每步确认）

### 步骤 6：搭建脚本骨架（类 + 基础方法）

先写类的骨架和基础工具方法，**不写业务特征方法**：

- `__init__`：参数接收、配置解析、`_map()` → `_prepare_data()`
- `_map()`：从 `txn_income_v1_1.py` 复用映射表逻辑
- `_clean_entity_name()`：实体名称清洗
- `_safe_stats()`：统一金额统计
- `_calc_slope()`：线性回归斜率
- `_max_consecutive()`：最大连续值
- `_prepare_data()`：数据预处理 + 多级缓存（按 type、window、entity 预分组）
- 空的 `generate_all_features()` 框架

> **确认点 6：汇报脚本骨架完成。展示文件路径、`__init__` 参数设计、`_prepare_data()` 中构建了哪些缓存。等用户确认骨架结构没问题后，再开始写业务特征方法。**

### 步骤 7：实现第一个 feature_group

先只实现一个 feature_group 方法，跑通端到端流程：

- 实现第一个 feature_group 方法（比如 `amount_by_type`）
- 实现 `generate_{module}_feature()` 入口函数
- 实现模块级配置（`{module}_type_tp_pairs` 等）
- 用简单测试数据验证能跑通

> **确认点 7：汇报第一个 feature_group 实现。展示代码、测试数据、输出结果。确认特征名、统计口径、默认值行为都正确后，再批量实现剩余 feature_group。**

### 步骤 8：实现剩余所有 feature_group

逐个实现步骤 2 设计表中的其他 feature_group：

- 每个 feature_group 写完后，用同样的测试数据验证一遍。
- 检查所有特征名是否全小写、无空格。

> **确认点 8：汇报所有 feature_group 实现完成。列出每个 group 产出的特征数量和示例特征名。等用户确认数量和命名无误后，进入模块注册。**

### 步骤 9：注册模块

修改 `v1_0_20260327/txn_tool/__init__.py`，只追加一行新模块 import：

```python
from . import txn_{module}
```

不要顺手整理 unrelated import，除非本次任务明确要求。

> **确认点 9：展示 __init__.py 的 diff。在 bash 中测试 `from v1_0_20260327.txn_tool import txn_{module}` 是否通过。等用户确认后再继续。**

### 步骤 10：接入 run_model_feature.py — 兜底字典

在现有 `model_vars_xxx` 系列后追加兜底字典：

```python
model_vars_{module} = {
    "bank_txn_{abbrev}_xxx_sum_28d": [-999999.99999],
    # ... 列出本模块全部特征
}
```

说明：

- `dict.csv` 是最终回溯列权威。
- `model_vars_{module}` 用于 `raw_data` 为空但仍需要返回模块默认列的场景。
- 如果这里和 `dict.csv` 不一致，空输入回溯时可能列数不对齐。
- 建议和 `dict.csv` 保持同步维护。

> **确认点 10：展示兜底字典的完整内容。确认字典中列出的特征数量与步骤 2 设计表中的特征数量一致。等用户确认后，继续注入正常交易分支。**

### 步骤 11：接入 run_model_feature.py — 正常交易分支

在 `expense/surplus` 后、`lender` 前新增：

```python
# ⭐===== {module} (NEW) =====
df_txn_{module} = txn_tool.txn_{module}.generate_{module}_feature(
    df=raw_data,
    feature_groups=None,   # 或指定 feature_groups 列表
)
df_txn_{module}["user_id"] = user_id
df_txn_{module}["sample_datetime"] = pd.to_datetime(send_time)
df_txn_{module} = fillna_clip_numeric(df_txn_{module}, -999999.99999)
```

> **确认点 11：展示正常交易分支的 diff。确认 fillna_clip_numeric 的默认值是否与模块约定一致。等用户确认后，继续空交易兜底分支。**

### 步骤 12：接入 run_model_feature.py — 空交易兜底分支

在 `raw_data is None or raw_data.shape[0] == 0` 分支中新增：

```python
df_txn_{module} = build_default_df(model_vars_{module}, -999999.99999)
```

> **确认点 12：展示空交易兜底分支的 diff。确认这里使用的 model_vars 字典名与步骤 10 一致、默认值与步骤 11 一致。等用户确认后，继续 merge 列表。**

### 步骤 13：接入 run_model_feature.py — merge 列表

在 merge 列表中追加 `df_txn_{module}`：

```python
for df in [
    df_txn_cate,
    df_txn_eod,
    df_txn_income,
    df_txn_expense,
    df_txn_surplus,
    df_txn_{module},    # ← 新增
    df_txn_lender,
]:
```

> **确认点 13：展示 merge 列表的 diff。明确汇报"generate_score 函数没有改动，model_vars 列表没有改动，打分链路不受影响"。等用户确认后，进入 dict.csv 更新。**

### 步骤 14：更新 dict.csv — 先列清单再追加

先把要追加的所有特征行列成清单给用户过目，然后再追加到文件末尾：

```csv
feature,model_vars,default
bank_txn_{abbrev}_xxx_sum_28d,model_vars_{module},-999999.99999
bank_txn_{abbrev}_xxx_count_28d,model_vars_{module},-999999.99999
...
```

规则：

- 第一列必须与代码输出完全一致。
- 第二列必须与 `run_model_feature.py` 的字典名一致。
- 第三列是回溯固定列缺失时的最终填充值。
- 不把 `user_id/application_id/sample_datetime/send_time/txn_raw_input_cnt` 等元信息写进 `dict.csv`。

> **确认点 14a：先列清单。展示所有要写入 dict.csv 的特征行（feature_name, model_vars, default），让用户逐行确认字段名和默认值是否正确。**

追加后做 dict 对齐检查：

```python
import pandas as pd

dict_df = pd.read_csv("dict.csv")
assert dict_df["feature"].is_unique
assert dict_df["feature"].str.lower().equals(dict_df["feature"])
assert dict_df["feature"].str.contains(" ").sum() == 0
```

> **确认点 14b：展示对齐检查结果。汇报 dict.csv 新增了多少行、总行数多少、检查是否全部通过。等用户确认后，进入验证阶段的大门。**

### 步骤 15：确认是否进入验证阶段

这是阶段二到阶段三的**大门确认**：

- 汇总阶段二的全部变更：新建了哪个脚本、改了几个文件、每个文件改了什么。
- 展示完整的 git diff 摘要。
- 确认没有改到打分链路相关的代码。

> **确认点 15：汇总阶段二全部变更。展示 git diff --stat。等用户说"开始验证"后，进入阶段三。**

---

## 阶段三：验证（每项验证单独确认）

### 步骤 16：模块导入验证

```bash
python -c "from v1_0_20260327.txn_tool import txn_{module}; print('OK')"
```

> **确认点 16：汇报 import 是否成功。如果失败，排查报错原因，修复后重新确认。**

### 步骤 17：单样本特征生成验证

用最小构造数据测试特征脚本本身能否跑通：

```bash
python -c "from v1_0_20260327.txn_tool.txn_{module} import generate_{module}_feature; import pandas as pd; df = pd.DataFrame({'user_id':[1], 'sample_datetime':['2025-06-01'], 'transaction_date':['2025-05-20'], 'amount':[100], 'balance':[1000], 'dr_cr':['credit'], 'category':['Wages'], 'third_party':['Employer'], 'account_type':['TX']}); out = generate_{module}_feature(df); print('shape:', out.shape); print('columns:', out.columns[:10].tolist())"
```

> **确认点 17：汇报 DataFrame shape、前 10 列列名、有无报错。如果有报错，修复后重新确认。**

### 步骤 18：空数据特征生成验证

测试空数据输入时特征脚本的行为：

```bash
python -c "from v1_0_20260327.txn_tool.txn_{module} import generate_{module}_feature; import pandas as pd; df = pd.DataFrame(columns=['user_id','sample_datetime','transaction_date','amount','balance','dr_cr','category','third_party','account_type']); out = generate_{module}_feature(df); print('shape:', out.shape); print(out.iloc[0].to_dict())"
```

> **确认点 18：汇报空数据下的返回结果。检查默认值行为是否与设计表一致（count/sum 应为 0，ratio/std 应为 nan 或默认值）。如果行为不对，修复后重新确认。**

### 步骤 19：run_model_feature 空数据兜底验证

测试全空输入走回溯入口的结果：

```bash
python -c "from v1_0_20260327.run_model_feature import run_model_feature; result = run_model_feature({'userId':1, 'applicationId':1, 'flowTime':'2025-06-01 12:00:00.0', 'bank_accounts':[], 'illion_raw_transactions':[], 'illion_day_end_balances':[]}); print('type:', type(result), 'len:', len(result)); hnw_keys = [k for k in result.keys() if '{abbrev}' in k]; print('feature count:', len(hnw_keys)); print('sample keys:', hnw_keys[:5])"
```

> **确认点 19：汇报 run_model_feature 空数据兜底的返回结果。检查：特征数量是否与 dict.csv 中本模块的行数一致、默认值是否正确填充、`user_id/application_id/sample_datetime` 等基础列是否存在。**

### 步骤 20：run_model_feature 正常数据验证

用构造数据走完整的回溯入口：

```bash
python -c "from v1_0_20260327.run_model_feature import run_model_feature; result = run_model_feature({'userId':1, 'applicationId':1, 'flowTime':'2025-06-01 12:00:00.0', 'bank_accounts':[], 'illion_raw_transactions':[{'amount':100,'balance':1000,'bank_account_id':'1','category':'Wages','dr_cr':'credit','illion_trx_uuid':'a','text':'Salary','third_party':'Employer','transaction_date':'2025-05-20','transaction_id':'1','trx_type':'Transfer'}], 'illion_day_end_balances':[]}); hnw_keys = [k for k in result.keys() if '{abbrev}' in k]; print('feature count:', len(hnw_keys)); print('sample:', {k:result[k] for k in hnw_keys[:5]})"
```

> **确认点 20：汇报正常数据下的返回结果。检查：特征值是否合理（不是全部默认值）、特征数量是否与 dict.csv 一致。**

### 步骤 21：dict.csv 列对齐验证

确认 dict.csv 中的特征列与 run_model_feature 输出完全对齐：

```python
import pandas as pd
from v1_0_20260327.run_model_feature import run_model_feature

# 获取输出列
result = run_model_feature({
    'userId': 1, 'applicationId': 1, 'flowTime': '2025-06-01 12:00:00.0',
    'bank_accounts': [], 'illion_raw_transactions': [], 'illion_day_end_balances': []
})
output_keys = set(result.keys())

# 获取 dict.csv 列
dict_df = pd.read_csv("dict.csv")
dict_features = set(dict_df["feature"].tolist())

# 对齐检查
in_dict_not_in_output = dict_features - output_keys
in_output_not_in_dict = {k for k in output_keys if k.startswith('bank_txn_')} - dict_features

print("in dict.csv but NOT in output:", in_dict_not_in_output)
print("in output but NOT in dict.csv:", in_output_not_in_dict)
assert len(in_dict_not_in_output) == 0, "dict.csv has features not in output!"
assert len(in_output_not_in_dict) == 0, "output has features not in dict.csv!"
print("OK")
```

> **确认点 21：汇报 dict.csv 与输出的对齐结果。如果有不一致，定位是代码输出的特征名写错了还是 dict.csv 写错了，修复后重新确认。**

### 步骤 22：性能验证

用模拟数据测试单样本耗时：

```python
import time
import pandas as pd
from v1_0_20260327.run_model_feature import run_model_feature

# 构造一个有 200 条交易的样本
np.random.seed(42)
n = 200
dates = pd.date_range('2025-01-01', '2025-05-31', freq='D')
input_data = {
    'userId': 1, 'applicationId': 1, 'flowTime': '2025-06-01 12:00:00.0',
    'bank_accounts': [],
    'illion_raw_transactions': [
        {
            'amount': np.random.uniform(-500, 5000),
            'balance': np.random.uniform(1000, 50000),
            'bank_account_id': '1',
            'category': np.random.choice(['Wages', 'Rent', 'Shopping', 'Internal', 'Insurance']),
            'dr_cr': np.random.choice(['credit', 'debit']),
            'illion_trx_uuid': str(i),
            'text': '',
            'third_party': np.random.choice(['Employer', 'Store', 'Bank', 'Insurance Co', '']),
            'transaction_date': str(np.random.choice(dates).date()),
            'transaction_id': str(i),
            'trx_type': 'Transfer',
        }
        for i in range(n)
    ],
    'illion_day_end_balances': [],
}

start = time.time()
result = run_model_feature(input_data)
elapsed = time.time() - start

print(f"单样本耗时: {elapsed:.3f}s (交易数: {n})")
```

> **确认点 22：汇报单样本耗时。如果耗时超过 2s（200 条交易），排查是否有重复 groupby 或冗余计算，优化后重新测试。等用户确认性能可接受。**

### 步骤 23：数据质量验证（有回溯结果时）

如果有批量样本回溯结果，输出以下 profiling：

- 覆盖率：非默认、非空占比。
- 零值率：对 count/sum 特征尤其重要。
- 分位数：`min/p1/p5/p50/p95/p99/max`。
- 唯一值数：剔除常数列或近常数列。
- 极端值：检查 clip/default 是否吞掉真实值。
- 相关性：同模块内 `abs(corr) > 0.98` 的高度重复变量只保留业务更清晰的一个。
- 窗口冗余：相邻窗口表现几乎一致时，优先保留更有业务解释的窗口。

> **确认点 23a：汇报覆盖率与零值率。标出覆盖率 < 5% 的特征，确认是否要剔除。等用户确认后继续分位数和相关性分析。**

> **确认点 23b：汇报分位数、极端值、相关性、窗口冗余分析。如果有高度相关或窗口冗余的特征，给出精简建议。**

产出评审表：

| feature | coverage | zero_rate | p50 | p95 | p99 | corr_max | decision | reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

`decision` 只能是：

- `keep`：保留进 `dict.csv`。
- `drop`：不接入回溯。
- `revise`：调整口径后再验证。

> **确认点 23c：汇报完整的评审表。逐条说明哪些保留、哪些剔除、哪些需要修改。等用户确认后，根据最终决定更新 dict.csv（如果和步骤 14 时有变化）。**

### 步骤 24：最终汇总

全部步骤完成后，做最终汇报：

- 新增/修改了哪些文件（完整清单）。
- 新增了多少个代码输出特征，最终多少个写入 `dict.csv`。
- 被剔除的特征及原因。
- 所有验证命令的结果摘要。
- 是否改动打分链路（默认应为"未改动"）。
- 后续建议：是否需要补充 reference 文件、是否需要增加测试用例、建议的下一次回溯窗口。

> **确认点 24：最终汇总。等用户说"完成"后，整个流程结束。**

---

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

---

## 模块命名规范速查

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

命名要求：全小写、不含空格、同一统计口径只用一种命名。

---

## 默认值规范

| 场景 | 变量原始值 | 进入 `run_model_feature` 后 |
| --- | --- | --- |
| count/sum 且无匹配交易 | `0` | 保持 `0` |
| mean/median/max/min/std/cv 且无匹配交易 | `np.nan` | 由 `fillna_clip_numeric()` 填成模块默认值 |
| ratio 分母无效 | `np.nan` | 由 `fillna_clip_numeric()` 填成模块默认值 |
| lender 类历史口径 | 按现有 lender 规则 | 通常填 `-1` |
| 普通交易探索变量 | `np.nan` 或数值 | 通常填 `-999999.99999` |

---

## 工程改动范围速查

| 步骤 | 文件 | 动作 | 确认点 |
| --- | --- | --- | --- |
| 0 | — | 理解需求方向 | 确认点 0 |
| 1 | — | 调研现有数据 | 确认点 1 |
| 2 | — | 输出候选特征设计表 | 确认点 2 |
| 3 | — | 确认模块命名与字段前缀 | 确认点 3 |
| 4 | — | 泄漏与口径检查 | 确认点 4 |
| 5 | — | 阶段一 → 阶段二大门确认 | 确认点 5 |
| 6 | `txn_{module}.py` | 搭建脚本骨架（类+基础方法） | 确认点 6 |
| 7 | `txn_{module}.py` | 实现第一个 feature_group | 确认点 7 |
| 8 | `txn_{module}.py` | 实现剩余所有 feature_group | 确认点 8 |
| 9 | `__init__.py` | 注册模块 import | 确认点 9 |
| 10 | `run_model_feature.py` | 新增兜底字典 | 确认点 10 |
| 11 | `run_model_feature.py` | 正常交易分支接入 | 确认点 11 |
| 12 | `run_model_feature.py` | 空交易兜底分支接入 | 确认点 12 |
| 13 | `run_model_feature.py` | merge 列表接入 | 确认点 13 |
| 14a | — | 列出 dict.csv 待追加清单 | 确认点 14a |
| 14b | `dict.csv` | 追加 + 对齐检查 | 确认点 14b |
| 15 | — | 阶段二 → 阶段三大门确认 | 确认点 15 |
| 16 | — | 模块导入验证 | 确认点 16 |
| 17 | — | 单样本特征生成验证 | 确认点 17 |
| 18 | — | 空数据特征生成验证 | 确认点 18 |
| 19 | — | run_model_feature 空数据兜底验证 | 确认点 19 |
| 20 | — | run_model_feature 正常数据验证 | 确认点 20 |
| 21 | — | dict.csv 列对齐验证 | 确认点 21 |
| 22 | — | 性能验证 | 确认点 22 |
| 23a | — | 覆盖率与零值率分析 | 确认点 23a |
| 23b | — | 分位数/相关性/窗口冗余分析 | 确认点 23b |
| 23c | — | 最终评审表与决策 | 确认点 23c |
| 24 | — | 最终汇总 | 确认点 24 |

不改的文件：

- `run_model.py`：打分链路，不因探索变量而改。
- `model_main.py`：薄封装，不改。
- `score_traceback.py`：打分回溯，不改。
- `feature_traceback.py`：通过 `run_func` 动态传入，通常不改。
- `main_traceback.ipynb`：入口不变，除非用户明确要求改 notebook。
