      
# -*- coding: utf-8 -*-
import os
import datetime
import logging
import joblib
import pandas as pd
from functools import reduce
from . import txn_tool
import numpy as np


from pytz import FixedOffset


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "git_pickle")

# ⭐0306_update：本次新增model_vars_income、model_vars_expense、model_vars_surplus，1_2版本新增特征模块所需
# model variables (UPDATED ON 1128)
# ================================

model_vars_balance = {
    'bank_txn_balance_std_168d': [-999999.99999],
    'bank_txn_balance_kurtosis_182d': [-999999.99999],
    'bank_txn_balance_debit_gap_max_182d': [-999999.99999],
    'bank_txn_balance_skewness_182d': [-999999.99999],
}


model_vars_category = {
    'bank_txn_category_cluster_debt_share_182d': [-999999.99999],
    'bank_txn_category_dishonours_credit_cnt_56d': [-999999.99999],
    'bank_txn_category_internal_transfer_credit_amt_182d': [-999999.99999],
    'bank_txn_category_non_sacc_loans_debit_cnt_28d': [-999999.99999],
    'bank_txn_category_external_transfers_share_28d': [-999999.99999],
}


model_vars_lender = {
    'bank_txn_lender_disburse_advance_amount_min': [-1],
    'bank_txn_lender_repay_jgs': [-1],
    'bank_txn_lender_repay_bnpl_amount_min_l28d': [-1],
    'bank_txn_lender_disburse_competitor_MyPayNow_count_l84d': [-1],
}


model_vars_income = {
    'bank_txn_income_wages_cv_56d': [-999999.99999],
    'bank_txn_income_wages_mean_consumption_rate_14d': [-999999.99999],
}


model_vars_expense = {
    'bank_txn_expense_insurance_ratio_14d': [-999999.99999],
    'bank_txn_expense_internaltransfer_daily_max_14d': [-999999.99999],
    'bank_txn_expense_internaltransfer_max_consecutive_days': [-999999.99999],
}


model_vars_surplus = {
# 当前版本无新增 surplus 特征，先留空
}


# ⭐0306_update2：

# model file
model_files = {
    "bank_model": "aus_old_risk_bid_submodel_20260323_v1_2_txn_lgb.pkl", 
}


def data_prepare(user_id, raw_data, eod_date, send_time, applicationid):
    """
    ⭐ 生产级兜底版 data_prepare（项目二）
    覆盖：
      1) 交易空 + 余额空
      2) 交易空 + 余额有
      3) 交易有 + 余额空
      4) 交易有 + 余额有

    保证：
      - 永不报错
      - 列齐
      - 可 merge
      - 不改变模型字段
    """

    # ======================================================
    # 0. 基础信息
    # ======================================================
    base_row = {
        "user_id": user_id,
        "application_id": applicationid,
        "sample_datetime": pd.to_datetime(send_time),
    }
    df_user_inf = pd.DataFrame([base_row])

    # ======================================================
    # 工具：构造「单行 default DataFrame」
    # ======================================================
    def build_default_df(var_dict, fill_value):
        row = {
            "user_id": user_id,
            "sample_datetime": pd.to_datetime(send_time),
        }
        for k, v in var_dict.items():
            row[k] = v[0] if isinstance(v, list) else v
        return pd.DataFrame([row])

    # ======================================================
    # 工具：仅数值列 fillna + clip
    # ======================================================

# ⭐修改：填充逻辑
    def fillna_clip_numeric(df, fill_value):
        if df is None or df.shape[0] == 0:
            return df
        #num_cols = df.select_dtypes(include="number").columns
        #if len(num_cols) > 0:
            #df[num_cols] = df[num_cols].fillna(fill_value).clip(lower=fill_value)
        df = df.replace(['', ' ', 'None', 'nan'], np.nan)
        # 除 user_id / sample_datetime 外全部强制转数值
        cols = df.columns.difference(["user_id", "sample_datetime"])
        df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
        df[cols] = df[cols].fillna(fill_value).clip(lower=fill_value)
        
        return df

    # ======================================================
    # 1. category / income / lender（是否有交易）
    # ======================================================
    if raw_data is None or raw_data.shape[0] == 0:
        # 👉 无交易：全部 default
        df_txn_cate   = build_default_df(model_vars_category, -999999.99999)
        df_txn_income = build_default_df(model_vars_income,   -99999)
        
        # ⭐新增
        df_txn_expense = build_default_df(model_vars_expense,  -999999.99999)   # ✅ 新增
        #df_txn_surplus = build_default_df(model_vars_surplus,  -999999.99999)   # ✅ 新增
        
        df_txn_lender = build_default_df(model_vars_lender,   -1)
    else:
        # ===== category =====
        df_txn_cate = txn_tool.txn_cate.compute_bank_features(
            df=raw_data,
            categories=["cluster_share","category_share"] #⭐ 修改["cluster_share"]→"cluster_share","category_share"
        )
        df_txn_cate["user_id"] = user_id
        df_txn_cate["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_cate = fillna_clip_numeric(df_txn_cate, -999999.99999)

        # ===== income （old，已停用）=====
        #df_txn_income = txn_tool.txn_income.calculate_inm(df=raw_data)
        #df_txn_income["user_id"] = user_id
        #df_txn_income["sample_datetime"] = pd.to_datetime(send_time)
        #df_txn_income = fillna_clip_numeric(df_txn_income, -99999)
        
        # ⭐===== income (v1_1) =====
        df_txn_income = txn_tool.txn_income_v1_1.generate_income_feature(df=raw_data,feature_groups=['amount_by_type','consumption_rate'])
        df_txn_income["user_id"] = user_id
        df_txn_income["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_income = fillna_clip_numeric(df_txn_income, -999999.99999)
        
         # ⭐===== expense (NEW) =====
        df_txn_expense = txn_tool.txn_expense.generate_expense_feature(df=raw_data,feature_groups=['ratio','amount_daily','max_consecutive_days'])
        df_txn_expense["user_id"] = user_id
        df_txn_expense["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_expense = fillna_clip_numeric(df_txn_expense, -999999.99999)
        

        # ⭐===== lender（新版调用方式，尽量对齐旧版口径） =====
        
        #alv = txn_tool.txn_lender.TXNSampleVar(
            #user_id=user_id,
            #send_time=send_time,
            #raw_data=raw_data
        #)
        
        alv = txn_tool.txn_lender.TxnListSampleVar(
            user_id = user_id, 
            send_time = send_time, 
            raw_data = raw_data)
        
        alv.clean_txn_step1()
        
        alv.create_lender_cate_list(
            txn_tool.load_refdata.load_third_category("third_party_cate")
        )
        alv.grouped_app_cate_list()
        
        alv.calc_vars_disburse_stat(cate_list=['cash/wage advance'], day_list=[None])
        alv.calc_vars_summary_repay_jgs_stat(day_list=[None])
        alv.calc_vars_repay_stat(cate_list=['BNPL'], day_list=[28])
        alv.calc_vars_disburse_competitor_stat(competitor_list=['MyPayNow'], day_list=[84])
        
        # ============================================================
        # 5) 输出结果：保持和旧版一致
        # ============================================================
        df_txn_lender = pd.json_normalize(alv.variables)
        df_txn_lender["user_id"] = user_id
        df_txn_lender["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_lender = fillna_clip_numeric(df_txn_lender, -1)

    # ======================================================
    # 2. balance（是否有余额）
    # ======================================================
    if eod_date is None or eod_date.shape[0] == 0:
        df_txn_eod = build_default_df(model_vars_balance, -999999.99999)
    else:
        df_txn_eod = txn_tool.txn_eod.compute_balance_timeseries_features(
            df_input=eod_date,
            categories=["basic","timeseries","gaps"]  #⭐ 由"basic", "direction", "gaps", "kgram"改为
        )
        df_txn_eod["user_id"] = user_id
        df_txn_eod["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_eod = fillna_clip_numeric(df_txn_eod, -999999.99999)

    # ======================================================
    # 3. merge（强兜底）
    # ======================================================
    df_master = df_user_inf
    #for df in [df_txn_cate, df_txn_eod, df_txn_income, df_txn_lender]:
    #⭐修改特征字典
    for df in [df_txn_cate, df_txn_eod, df_txn_income, df_txn_expense, df_txn_lender]:
        if df is not None and df.shape[0] > 0:
            df_master = df_master.merge(
                df,
                on=["user_id", "sample_datetime"],
                how="left"
            )

    # ======================================================
    # 4. float 精度统一
    # ======================================================
    for col in df_master.columns:
        if df_master[col].dtype == "float64":
            df_master[col] = df_master[col].round(5)

    return df_master



def load_model(model_files):
    clfs = {}
    for model in model_files.keys():
        model_file = model_files[model]
        clfs.update({model: joblib.load(os.path.join(PKL_DIR, model_file)),})

    return clfs


def model_scoring(df, clfs):

    result = {}

    for model in model_files.keys():
        score_name = "aus_old_risk_bid_submodel_v20260323_v1_2_txn_lgb_score"
        result[score_name] = clfs[model].predict(
            df[clfs[model].feature_name()].values,
            num_iteration=clfs[model].best_iteration
        )[0]
    
    return result




def raw_txn_from_slice(data, uid, tm):
    columns = ["user_id", "sample_datetime", "transaction_id", "bank_account_id", "illion_trx_uuid", "transaction_date", "text", "amount", "balance"
               , "dr_cr", "trx_type", "category", "third_party"]
    
    if len(data) == 0:
        return pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(data)
        df['user_id'] = uid
        df['sample_datetime'] = pd.to_datetime(tm)
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
        df['amount'] = df['amount'].astype('float')
        df['balance'] = df['balance'].astype('float')

    return df


def balance_from_slice(data, uid, tm):
    columns = ["user_id", "sample_datetime", "balance_id", "bank_account_id", "balance_date", "balance"]
    
    if len(data) == 0:
        return pd.DataFrame(columns=columns)
    else:
        df = pd.DataFrame(data)
        df['user_id'] = uid
        df['sample_datetime'] = pd.to_datetime(tm)
        df['balance_date'] = pd.to_datetime(df['balance_date'])
        df['balance'] = df['balance'].astype('float')

    return df


def generate_score(input_vars):
    user_id = input_vars["userId"]
    application_id = input_vars["applicationId"]
    send_time = str(pd.to_datetime(input_vars["flowTime"][:19]))

    # ========== 1. 读取交易流水与余额明细 ========== #
    txn_raw = raw_txn_from_slice(input_vars["illion_raw_transactions"], user_id, send_time)
    day_balance = balance_from_slice(input_vars["illion_day_end_balances"], user_id, send_time)
    
    # ✅ 从 bank_accounts 把 account_type merge 回 txn_raw，保证下游工具仍可用
    ba = pd.DataFrame(input_vars.get("bank_accounts", []))
    if txn_raw.shape[0] > 0:
        if ba.shape[0] > 0 and "bank_account_id" in ba.columns:
            # 避免 merge dtype 不一致
            txn_raw["bank_account_id"] = txn_raw["bank_account_id"].astype(str)
            ba["bank_account_id"] = ba["bank_account_id"].astype(str)
            if "account_type" not in ba.columns:
                ba["account_type"] = ""
            txn_raw = txn_raw.merge(
                ba[["bank_account_id", "account_type"]].drop_duplicates("bank_account_id"),
                on="bank_account_id",
                how="left"
            )
        else:
            txn_raw["account_type"] = ""

    # ======================================================
    # ⭐ 特殊规则：两个数组都为空 → 模型分 = None, 生成 feature = None
    # ======================================================
    if txn_raw.shape[0] == 0 and day_balance.shape[0] == 0:

        score_name = "aus_old_risk_bid_submodel_v20260323_v1_2_txn_lgb_score"
        features_name = f"{score_name}_features"

        result = {
            score_name: None,      # ⭐ 新调整，数据源为空，模型分赋值空
            features_name: None   # ⭐ 新增功能：feature 直接为 None
        }

        return result
    # ======================================================
    # ⭐ 特殊逻辑结束
    # ======================================================

    # ========== 2. 入模变量 ========== # 
    # ⭐ 修改1-2模型入模变量
    model_vars = [
        # ===== balance =====
        'bank_txn_balance_std_168d',
        'bank_txn_balance_kurtosis_182d',
        'bank_txn_balance_debit_gap_max_182d',
        'bank_txn_balance_skewness_182d',
        

        # ===== category =====
        'bank_txn_category_cluster_debt_share_182d',
        'bank_txn_category_dishonours_credit_cnt_56d',
        'bank_txn_category_internal_transfer_credit_amt_182d',
        'bank_txn_category_non_sacc_loans_debit_cnt_28d',
        'bank_txn_category_external_transfers_share_28d',
        

        # ===== lender =====
        'bank_txn_lender_disburse_advance_amount_min',
        'bank_txn_lender_repay_jgs',
        'bank_txn_lender_repay_bnpl_amount_min_l28d',
        'bank_txn_lender_disburse_competitor_MyPayNow_count_l84d',
        
        # ===== Expense =====
        'bank_txn_expense_insurance_ratio_14d',
        'bank_txn_expense_internaltransfer_daily_max_14d',
        'bank_txn_expense_internaltransfer_max_consecutive_days',
        
        # ===== Income =====
        'bank_txn_income_wages_cv_56d',
        'bank_txn_income_wages_mean_consumption_rate_14d',
    ]

    # ========== 3. 生成宽表 ========== #
    df_master = data_prepare(
        user_id=user_id,
        raw_data=txn_raw,
        eod_date=day_balance,
        send_time=send_time,
        applicationid=application_id
    )[model_vars]

    # ========== 4. 加载模型 ========== #
    clfs = load_model(model_files=model_files)

    # ========== 5. 打分 ========== #
    result = model_scoring(df=df_master, clfs=clfs)

    # ========== 6. 输出特征 ========== #
    # ========== 6. 输出特征 ========== #
    score_name = "aus_old_risk_bid_submodel_v20260323_v1_2_txn_lgb_score"
    features_name = f"{score_name}_features"

    fea_dict = df_master.loc[:, model_vars].to_dict(orient="records")[0]
    fea_dict["txn_raw_input_cnt"] = txn_raw.shape[0]
    fea_dict["txn_balance_input_cnt"] = day_balance.shape[0]
    fea_dict["send_time"] = send_time

    fea_dict["transaction_date_max"] = (
        None if txn_raw.shape[0] == 0
        else txn_raw['transaction_date'].dt.strftime('%Y-%m-%d').max()
    )
    fea_dict["balance_date_max"] = (
        None if day_balance.shape[0] == 0
        else day_balance['balance_date'].dt.strftime('%Y-%m-%d').max()
    )

    result[features_name] = fea_dict

    return result


'''
if __name__ == "__main__":
    input_data = {
        "userId": 10882774,
        "applicationId": 123123,
        "flowTime": "2025-12-24 13:41:11.0",
        "illion_raw_transactions": [{
            "amount": -100,
            "balance": 1900.85,
            "bank_account_id": "3952166",
            "category": "Internal",
            "dr_cr": "debit",
            "illion_trx_uuid": "6ecfc903-a420-46d6-8037-497047c70330",
            "text": "Transfer",
            "third_party": "Internal",
            "transaction_date": "2025-10-09",
            "transaction_id": "2316171453",
            "trx_type": "Internal"
            }, {
            "amount": -100,
            "balance": 2000.85,
            "bank_account_id": "3952166",
            "category": "Internal",
            "dr_cr": "debit",
            "illion_trx_uuid": "5ffe5bf7-94f8-480e-9a4c-9bb0b21e261f",
            "text": "Transfer",
            "third_party": "Internal",
            "transaction_date": "2025-10-09",
            "transaction_id": "2316171454",
            "trx_type": "Internal"
            }
                ],
        "illion_day_end_balances":[{
            # "job_id": "123",
            "balance": -42560.72,
            "balance_date": "2025-09-28",
            "balance_id": "1.07723905e+09",
            "bank_account_id": "3.948332e+06]"
            }, {
            # "job_id": "123",
            "balance": -42144.71,
            "balance_date": "2025-09-27",
            "balance_id": "1.077239051e+09",
            "bank_account_id": "3.948332e+06]"
            }
            ]
        }

    result = generate_score(input_vars=input_data)
    print(result)
'''

    
    # test_df = pd.read_csv(os.path.join(BASE_DIR, "tests/applist_test.csv"))
    # print(test_df.iloc)
    # sample_df = test_df[["userId", "sample_datetime"]].drop_duplicates().reset_index(drop=True)

    # results = []
    # u_list = []
    # for i in range(sample_df.shape[0]):

    #     userId = sample_df["userId"].iloc[i]
    #     listingId = -1
    #     flowTime = sample_df["sample_datetime"].iloc[i]
    #     print([userId, listingId, flowTime])
    #     applist_data = test_df.loc[(test_df["userId"] == userId) & (test_df["sample_datetime"] == flowTime), ].reset_index(drop=True)
    #     # applist_data = applist_data.to_json(orient="records", force_ascii=False)
        
    #     input_vars = {"userId": userId, "listingId": listingId, "flowTime": flowTime, "app_list": applist_data}
    #     result = generate_score(input_vars=input_vars)
    #     print(result)
    #     u_list.append(userId)
    #     results.append(result)
    
    # res = pd.DataFrame(data={'userid':u_list, 'score':results})

    # res.to_csv(os.path.join(BASE_DIR, "tests/appscore_tests.csv"))

    


    