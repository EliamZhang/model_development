### 项目名称
aus_old_risk_bid_submodel_v1_2_20260327_txn

### 相关对接人员
* 模型开发及维护：张昱良
* 模型服务调用方：张昱良

##### 模型依赖

+ 无

##### 描述信息

澳洲老客申请风险银行流水子模型20260327

## 环境信息

##### Python 版本

3.7

##### 机器学习框架

lightgbm

##### 机器学习框架版本

2.3.1

##### 处理器

CPU

##### APT 依赖

```txt
```

##### PIP 依赖

```txt
pandas==1.1.5
numpy==1.18.2
lightgbm==2.3.1
joblib==0.16.0
```

##### CONDA 依赖

```txt
```

## 模型 API 信息

##### 模型文件存放目录路径

git_pickle

##### 模型推理对象路径

model_main.PredictMain

##### 入参 key 列表

```
[
  "userId",
  "applicationId",
  "flowTime",
  "bank_accounts",
  "illion_raw_transactions",
  "illion_day_end_balances"
]
```

##### 入参示例

```json
{
  "userId": 484132203,
  "applicationId": 2303527,
  "flowTime": "2026-03-01 12:00:02.0",
  "bank_accounts": [
    {
      "bank_account_id": 4429596,
      "account_type": ""
    }
  ],
  "illion_raw_transactions": [
    {
      "amount": 155.08,
      "balance": 591.41,
      "bank_account_id": 4429596,
      "category": "Non SACC Loans",
      "dr_cr": "debit",
      "illion_trx_uuid": "4d511597-8542-4257-b7f9-436f1dd1d570",
      "text": "PAYMENT BY AUTHORITY TO QuickCash DT.4wfpel 26840439",
      "third_party": "Moneyspot",
      "transaction_date": "2025-12-23",
      "transaction_id": 2604737994,
      "trx_type": "General Payment"
    }
  ],
  "illion_day_end_balances": [
    {
      "balance": -42560.72,
      "balance_date": "2025-09-28",
      "balance_id": "1.07723905e+09",
      "bank_account_id": "3948332"
    }
  ]
}


```

##### 出参示例

```json
{
  "aus_old_risk_bid_submodel_20260323_v1_2_txn_lgb": 0.04101206412286587,
  "aus_old_risk_bid_submodel_20260323_v1_2_txn_lgb_features": {
    "bank_txn_balance_std_168d": 294.16349,
    "bank_txn_balance_kurtosis_182d": -999999.99999,
    "bank_txn_balance_debit_gap_max_182d": -999999.99999,
    "bank_txn_balance_skewness_182d": -999999.99999,
    "bank_txn_category_cluster_debt_share_182d": 0.51559,
    "bank_txn_category_dishonours_credit_cnt_56d": 2,
    "bank_txn_category_internal_transfer_credit_amt_182d": 0.0,
    "bank_txn_category_non_sacc_loans_debit_cnt_28d": 1,
    "bank_txn_category_external_transfers_share_28d": 0.56284,
    "bank_txn_lender_disburse_advance_amount_min": 100.0,
    "bank_txn_lender_repay_jgs": 3,
    "bank_txn_lender_repay_bnpl_amount_min_l28d": -1,
    "bank_txn_lender_disburse_competitor_MyPayNow_count_l84d": -1,
    "bank_txn_expense_insurance_ratio_14d": 0,
    "bank_txn_expense_internaltransfer_daily_max_14d": -999999.99999,
    "bank_txn_expense_internaltransfer_max_consecutive_days": -999999.99999,
    "bank_txn_income_wages_cv_56d": -999999.99999,
    "bank_txn_income_wages_mean_consumption_rate_14d": 0.11633,
    "txn_raw_input_cnt": 63,
    "txn_balance_input_cnt": 2,
    "send_time": "2026-03-01 12:00:02",
    "transaction_date_max": "2026-02-27",
    "balance_date_max": "2025-09-28"
  }
}



```

##### CPU 数量

1

##### Memory数量（G）

1

##### 副本数

2

##### 启动方式

uWSGI