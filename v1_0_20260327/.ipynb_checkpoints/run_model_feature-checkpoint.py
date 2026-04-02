      
# coding: utf-8
import os
import datetime
import logging
import joblib
import pandas as pd
from functools import reduce
from . import txn_tool
# from txn_tool import txn_eod, txn_cate, txn_income, txn_lender, load_refdata
import numpy as np

# import txn_tool
from pytz import FixedOffset
# MEX_TZ = FixedOffset(-360)


logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
PKL_DIR = os.path.join(BASE_DIR, "git_pickle")


# model variables (UPDATED ON 1128)
# ================================

model_vars_balance = {
    'bank_txn_balance_below_avg_ratio_182d': [-999999.99999],
    'bank_txn_balance_debit_gap_max_182d': [-999999.99999],
    'bank_txn_balance_kgram2-1_0_84d': [-999999.99999],
    'bank_txn_balance_streak_inc_max_14d': [-999999.99999],
}

model_vars_category = {
    'bank_txn_category_cluster_debt_share_14d': [-999999.99999],
    'bank_txn_category_cluster_memberships_subs_share_56d': [-999999.99999],
    'bank_txn_category_credit_card_repayments_debit_cnt_28d': [-999999.99999],
    'bank_txn_category_global_debit_cnt_14d': [-999999.99999],
    'bank_txn_category_insurance_cnt_84d': [-999999.99999],
    'bank_txn_category_insurance_debit_amt_7d': [-999999.99999],
}

model_vars_income = {
    # 当前 fea_cols 中没有 income 类变量，如有再补充
}

model_vars_lender = {
    'bank_txn_lender_disburse_advance_amount_min_l56d': [-1],
    'bank_txn_lender_disburse_loan_amount_min_l28d': [-1],
    'bank_txn_lender_disburse_single_advance_amount_max': [-1],
    'bank_txn_lender_repay_bnpl_amount_avg_l168d': [-1],
    'bank_txn_lender_repay_bnpl_amount_sum_l28d': [-1],
}

model_vars_expense = {
    # 如果你暂时不想写默认值映射，这里可以先留空
}

model_vars_surplus = {
    # 先留空也可以
    # 例如：
    # 'bank_txn_surplus_xxx_28d': [-999999.99999],
}

# model file
model_files = {
    "bank_model": "new_customer_risk_model_v1.0_lgb_1119.pkl",
}


def data_prepare(user_id, raw_data, eod_date, send_time, applicationid):
    """
    ⭐ 生产级兜底版 data_prepare
    - 覆盖 4 种场景：
        1) 交易空 + 余额空
        2) 交易空 + 余额有
        3) 交易有 + 余额空
        4) 交易有 + 余额有
    - 不修改原模型变量名
    - 所有 DataFrame 都保证：
        - 有 user_id / sample_datetime
        - 只有 1 行
        - 可安全 merge
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
    # 工具：构造「单行默认表」
    # ======================================================
    def build_default_df(var_dict, fill_value):
        row = {
            "user_id": user_id,
            "sample_datetime": pd.to_datetime(send_time),
        }
        for k, v in var_dict.items():
            # v 可能是 [-9999]，也可能是标量
            row[k] = v[0] if isinstance(v, list) else v
        return pd.DataFrame([row])

    # ======================================================
    # 工具：仅处理数值列
    # ======================================================
    def fillna_clip_numeric(df, fill_value):
        if df is None or df.shape[0] == 0:
            return df

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
        # 👉 无交易：全部走 default
        df_txn_cate   = build_default_df(model_vars_category, -999999.99999)
        df_txn_income = build_default_df(model_vars_income,   -999999.99999)
        df_txn_expense = build_default_df(model_vars_expense,  -999999.99999)   # ✅ 新增
        df_txn_surplus = build_default_df(model_vars_surplus,  -999999.99999)   # ✅ 新增
        df_txn_lender = build_default_df(model_vars_lender,   -1)
    else:
        # ===== category =====
        df_txn_cate = txn_tool.txn_cate.compute_bank_features(
            df=raw_data,
            categories=None
        )
        df_txn_cate["user_id"] = user_id
        df_txn_cate["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_cate = fillna_clip_numeric(df_txn_cate, -999999.99999)

        # ===== income（old，已停用） =====
        # df_txn_income = txn_tool.txn_income.calculate_inm(df=raw_data)
        # df_txn_income["user_id"] = user_id
        # df_txn_income["sample_datetime"] = pd.to_datetime(send_time)
        # df_txn_income = fillna_clip_numeric(df_txn_income, -999999.99999)
        
        # ===== income (v1_1) =====
        df_txn_income = txn_tool.txn_income_v1_1.generate_income_feature(df=raw_data,feature_groups=None)
        df_txn_income["user_id"] = user_id
        df_txn_income["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_income = fillna_clip_numeric(df_txn_income, -999999.99999)
        
        # ===== expense (NEW) =====
        df_txn_expense = txn_tool.txn_expense.generate_expense_feature(df=raw_data,feature_groups=None)
        df_txn_expense["user_id"] = user_id
        df_txn_expense["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_expense = fillna_clip_numeric(df_txn_expense, -999999.99999)
        
        # ===== surplus (NEW) =====
        df_txn_surplus = txn_tool.txn_surplus.generate_surplus_features(df = raw_data,feature_groups = None)
        df_txn_surplus["user_id"] = user_id
        df_txn_surplus["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_surplus = fillna_clip_numeric(df_txn_surplus, -999999.99999)

        # ============================================================
        # lender 变量计算（新版调用方式，尽量对齐旧版口径）
        # ============================================================

        # 1) 窗口和类别：沿用旧版
        day_list = [None, 1, 3, 7, 14, 28, 56, 84, 168, 182]
        cate_list = ['personal loan', 'BNPL', 'cash/wage advance', 'bank']

        # 2) 创建实例 + 数据准备
        alv = txn_tool.TxnListSampleVar(
            user_id=user_id,
            send_time=send_time,
            raw_data=raw_data
        )

        alv.clean_txn_step1()
        alv.create_lender_cate_list(
            lender_cate_mapping_df=txn_tool.load_refdata.load_third_category('third_party_cate')
        )
        alv.grouped_app_cate_list()

        # ============================================================
        # 3) 放款相关
        # ============================================================

        # 整体放款统计
        alv.calc_vars_summary_disburse_stat(day_list=day_list)
        alv.calc_vars_summary_disburse_jgs_stat(day_list=day_list)

        # 分类别放款统计
        alv.calc_vars_disburse_stat(
            cate_list=cate_list,
            day_list=day_list
        )

        # 单机构放款最大统计
        alv.calc_vars_disburse_single_stat(cate_list=cate_list)

        # 分类别机构数统计
        alv.calc_vars_disburse_jgs_stat(
            cate_list=cate_list,
            day_list=day_list
        )

        # ============================================================
        # 4) 还款相关
        # ============================================================

        # 整体还款统计
        alv.calc_vars_summary_repay_stat(day_list=day_list)
        alv.calc_vars_summary_repay_jgs_stat(day_list=day_list)

        # 分类别还款统计
        alv.calc_vars_repay_stat(
            cate_list=cate_list,
            day_list=day_list
        )

        # 单机构还款最大统计
        alv.calc_vars_repay_single_stat(cate_list=cate_list)

        # 分类别机构数统计
        alv.calc_vars_repay_jgs_stat(
            cate_list=cate_list,
            day_list=day_list
        )
        
        # --- 新增：指定具体机构的还款/放款统计（例如关注 ABC Lending 和 XYZ Finance）---
        competitor_info = txn_tool.load_refdata.load_third_category('competitor_list')
        competitor_list = competitor_info.loc[competitor_info["installed_sample_prop"]>=0.05, "third_party"].tolist()
        alv.calc_vars_repay_competitor_stat(
            competitor_list=competitor_list,
            day_list=day_list
        )
        alv.calc_vars_disburse_competitor_stat(
            competitor_list=competitor_list,
            day_list=day_list
        )
        
        

        # ============================================================
        # 5) 输出结果：保持和旧版一致
        # ============================================================
        df_txn_lender = pd.json_normalize(alv.variables)
        df_txn_lender["user_id"] = user_id
        df_txn_lender["sample_datetime"] = pd.to_datetime(send_time)

        # 和旧版一致：数值缺失填充 -1
        df_txn_lender = fillna_clip_numeric(df_txn_lender, -1)
        
        
    # ======================================================
    # 2. balance（是否有余额）
    # ======================================================
    if eod_date is None or eod_date.shape[0] == 0:
        df_txn_eod = build_default_df(model_vars_balance, -999999.99999)
    else:
        df_txn_eod = txn_tool.txn_eod.compute_balance_timeseries_features(
            df_input=eod_date,
            categories=None
        )
        df_txn_eod["user_id"] = user_id
        df_txn_eod["sample_datetime"] = pd.to_datetime(send_time)
        df_txn_eod = fillna_clip_numeric(df_txn_eod, -999999.99999)

    # ======================================================
    # 3. merge（强保证所有表都有 key）
    # ======================================================
    df_master = df_user_inf
    #for df in [df_txn_cate, df_txn_eod, df_txn_income, df_txn_lender]:
    for df in [df_txn_cate, df_txn_eod, df_txn_income, df_txn_expense, df_txn_surplus, df_txn_lender]:
        if df is not None and df.shape[0] > 0:
            df_master = df_master.merge(
                df,
                on=["user_id", "sample_datetime"],
                how="left"
            )

    # ======================================================
    # 4. 浮点统一精度
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
        score_name = "aus_new_risk_bid_3rdmodel_v1_0_20251201"
        result[score_name] = clfs[model].predict(df[clfs[model].feature_name()].values, num_iteration=clfs[model].best_iteration)[0]
    
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
            # 确保 key 都是 string，避免 merge dtype 报错
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
    # ⭐ 特殊规则：两个数组都为空 → 模型分 = -1, 生成 feature = None
    # ======================================================
    if txn_raw.shape[0] == 0 and day_balance.shape[0] == 0:

        score_name = "aus_new_risk_bid_3rdmodel_v1_0_20251201"
        features_name = f"{score_name}_features"

        result = {
            score_name: None,
            features_name: None   # ⭐ 新增功能：feature 直接为 None
        }

        return result
    # ======================================================
    # ⭐ 特殊逻辑结束
    # ======================================================

    # ========== 2. 入模变量 ========== #
    model_vars = [
        # ===== balance =====
        'bank_txn_balance_below_avg_ratio_182d',
        'bank_txn_balance_debit_gap_max_182d',
        'bank_txn_balance_kgram2-1_0_84d',
        'bank_txn_balance_streak_inc_max_14d',

        # ===== category =====
        'bank_txn_category_cluster_debt_share_14d',
        'bank_txn_category_cluster_memberships_subs_share_56d',
        'bank_txn_category_credit_card_repayments_debit_cnt_28d',
        'bank_txn_category_global_debit_cnt_14d',
        'bank_txn_category_insurance_cnt_84d',
        'bank_txn_category_insurance_debit_amt_7d',

        # ===== lender =====
        'bank_txn_lender_disburse_advance_amount_min_l56d',
        'bank_txn_lender_disburse_loan_amount_min_l28d',
        'bank_txn_lender_disburse_single_advance_amount_max',
        'bank_txn_lender_repay_bnpl_amount_avg_l168d',
        'bank_txn_lender_repay_bnpl_amount_sum_l28d',
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
    score_name = "aus_new_risk_bid_3rdmodel_v1_0_20251201"
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




import pandas as pd
from functools import lru_cache

@lru_cache()
def load_fixed_feature_config(dict_csv_path):
    """
    读取 dict.csv，返回：
    1) fixed_feature_cols: List[str]      # 严格顺序
    2) default_map: Dict[str, Any]        # feature -> default
    """
    df = pd.read_csv(dict_csv_path)

    df["feature"] = df["feature"].astype(str)
    df["default"] = df["default"]

    fixed_feature_cols = df["feature"].tolist()
    default_map = dict(zip(df["feature"], df["default"]))

    return fixed_feature_cols, default_map


def run_model_feature(input_vars, dict_csv_path="./dict.csv"):
    """
    ⭐ 工程安全版：仅生成特征（不打分）
    - 固定列 & 顺序完全由 dict.csv 控制
    - CSV / concat / 多进程 100% 不错位
    """

    # ======================================================
    # 0. 基础信息
    # ======================================================
    user_id = input_vars["userId"]
    application_id = input_vars["applicationId"]
    send_time = str(pd.to_datetime(input_vars["flowTime"][:19]))

    # ======================================================
    # 1. 读取固定列配置（唯一权威）
    # ======================================================
    FIXED_FEATURE_COLS, DEFAULT_MAP = load_fixed_feature_config(dict_csv_path)

    # 固定前置列（不放在 dict.csv 的那种）
    BASE_COLS = ["user_id", "application_id", "sample_datetime"]

    # ======================================================
    # 2. 读取交易流水与余额
    # ======================================================
    txn_raw = raw_txn_from_slice(
        input_vars.get("illion_raw_transactions", []),
        user_id,
        send_time
    )
    day_balance = balance_from_slice(
        input_vars.get("illion_day_end_balances", []),
        user_id,
        send_time
    )
    
    # ✅ 从 bank_accounts 把 account_type merge 回 txn_raw，保证下游工具仍可用
    ba = pd.DataFrame(input_vars.get("bank_accounts", []))
    if txn_raw.shape[0] > 0:
        if ba.shape[0] > 0 and "bank_account_id" in ba.columns:
            # 确保 key 都是 string，避免 merge dtype 报错
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
    # ⭐ case 1：两个数组都为空
    # ======================================================
    if txn_raw.shape[0] == 0 and day_balance.shape[0] == 0:

        empty_row = {
            "user_id": user_id,
            "application_id": application_id,
            "sample_datetime": send_time,
        }

        # 👉 所有固定特征列，严格按 dict.csv
        for col in FIXED_FEATURE_COLS:
            empty_row[col] = DEFAULT_MAP.get(col)

        # 元信息
        empty_row.update({
            "txn_raw_input_cnt": 0,
            "txn_balance_input_cnt": 0,
            "send_time": send_time,
            "transaction_date_max": None,
            "balance_date_max": None,
        })

        return empty_row

    # ======================================================
    # ⭐ case 2：正常生成特征
    # ======================================================
    df_master = data_prepare(
        user_id=user_id,
        raw_data=txn_raw,
        eod_date=day_balance,
        send_time=send_time,
        applicationid=application_id
    )

    fea_row = df_master.to_dict(orient="records")[0]

    # ======================================================
    # ⭐ 强制对齐（核心修复点）
    # ======================================================
    aligned_row = {
        "user_id": user_id,
        "application_id": application_id,
        "sample_datetime": send_time,
    }

    for col in FIXED_FEATURE_COLS:
        if col in fea_row:
            aligned_row[col] = fea_row[col]
        else:
            aligned_row[col] = DEFAULT_MAP.get(col)

    # ======================================================
    # ⭐ 元信息（稳定列）
    # ======================================================
    aligned_row.update({
        "txn_raw_input_cnt": txn_raw.shape[0],
        "txn_balance_input_cnt": day_balance.shape[0],
        "send_time": send_time,
        "transaction_date_max": (
            None if txn_raw.shape[0] == 0
            else txn_raw["transaction_date"].dt.strftime("%Y-%m-%d").max()
        ),
        "balance_date_max": (
            None if day_balance.shape[0] == 0
            else day_balance["balance_date"].dt.strftime("%Y-%m-%d").max()
        ),
    })

    return aligned_row

    