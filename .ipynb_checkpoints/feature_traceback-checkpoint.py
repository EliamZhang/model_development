      
# -*- coding: utf-8 -*-

import os, gc, glob, json, logging, csv
import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from collections import defaultdict

# =====================================================
# 1. 日志控制
# =====================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

CHUNK_SIZE = 200_000
WRITE_BATCH = 200

# =====================================================
# 2. 工具函数
# =====================================================
def is_valid_float(x):
    try:
        float(x)
        return True
    except Exception:
        return False

def is_valid_balance_record(rec: dict) -> bool:
    return is_valid_float(rec.get("balance"))

# =====================================================
# 3. 索引构建
# =====================================================
def build_csv_index(csv_dir):
    logger.info(f"📚 构建索引：{csv_dir}")
    index = defaultdict(set)

    fps = glob.glob(os.path.join(csv_dir, "*.csv"))
    for fp in fps:
        try:
            for chunk in pd.read_csv(
                fp,
                usecols=["user_id", "application_id"],
                chunksize=CHUNK_SIZE
            ):
                uniq = chunk.drop_duplicates(subset=["user_id", "application_id"])
                for uid, aid in uniq[["user_id", "application_id"]].to_numpy():
                    index[(uid, aid)].add(fp)
        except Exception as e:
            logger.warning(f"⚠️ 索引失败跳过：{fp} | {e}")

    logger.info(f"✅ 索引完成：{len(index)} keys")
    return index

# =====================================================
# 4. 匹配读取
# =====================================================
def load_match_rows_from_index(csv_index, user_id, application_id, usecols):
    fps = csv_index.get((user_id, application_id))
    if not fps:
        return pd.DataFrame(columns=usecols)

    matched = []
    for fp in fps:
        try:
            for chunk in pd.read_csv(fp, usecols=usecols, chunksize=CHUNK_SIZE):
                m = chunk[
                    (chunk["user_id"] == user_id) &
                    (chunk["application_id"] == application_id)
                ]
                if not m.empty:
                    matched.append(m)
        except Exception:
            continue

    if matched:
        return pd.concat(matched, ignore_index=True)
    return pd.DataFrame(columns=usecols)

# =====================================================
# 5. 模型输入构造
# =====================================================
TXN_USECOLS = [
    "user_id", "application_id",
    "amount", "balance", "bank_account_id", "account_type","category",
    "dr_cr", "illion_trx_uuid", "text", "third_party",
    "transaction_date", "transaction_id", "trx_type"
]

BAL_USECOLS = [
    "user_id", "application_id",
    "balance", "balance_date", "balance_id", "bank_account_id"
]

def build_input_data(r, txn_index, bal_index):
    uid, aid = r["user_id"], r["application_id"]
    ft = str(r["sample_datetime"])

    txn_df = load_match_rows_from_index(txn_index, uid, aid, TXN_USECOLS)
    bal_df = load_match_rows_from_index(bal_index, uid, aid, BAL_USECOLS)

    # =====================================================
    # 1) 交易明细：移除 account_type 字段（account_type 将上提到 bank_accounts）
    # =====================================================
    txn_records = []
    if not txn_df.empty:
        txn_records = (
            txn_df[
                [
                    "amount", "balance", "bank_account_id", "category",
                    "dr_cr", "illion_trx_uuid", "text", "third_party",
                    "transaction_date", "transaction_id", "trx_type"
                    # ✅ 注意：这里不再包含 "account_type"
                ]
            ]
            .fillna("")
            .to_dict("records")
        )

    # =====================================================
    # 2) 余额明细：保持不变
    # =====================================================
    bal_records = []
    if not bal_df.empty:
        raw_bal = (
            bal_df[
                ["balance", "balance_date", "balance_id", "bank_account_id"]
            ]
            .fillna("")
            .to_dict("records")
        )
        for rec in raw_bal:
            if is_valid_balance_record(rec):
                bal_records.append(rec)

    # =====================================================
    # 3) 新增 bank_accounts：按 bank_account_id 汇总 account_type
    #    - 优先从 txn_df 里拿 (bank_account_id, account_type)
    #    - 如果 bal_df 里出现了 txn_df 没有的 bank_account_id，则补一条 account_type=""
    # =====================================================
    bank_accounts = []

    if not txn_df.empty:
        ba_df = (
            txn_df[["bank_account_id", "account_type"]]
            .fillna("")
            .drop_duplicates(subset=["bank_account_id", "account_type"])
        )
        bank_accounts = ba_df.to_dict("records")

    # 补齐：余额里有但交易里没有的 bank_account_id
    if not bal_df.empty:
        existing_ids = {x.get("bank_account_id") for x in bank_accounts}
        for bid in bal_df["bank_account_id"].fillna("").astype(str).unique().tolist():
            if bid and bid not in existing_ids:
                bank_accounts.append({"bank_account_id": bid, "account_type": ""})
                existing_ids.add(bid)

    return {
        "userId": uid,
        "applicationId": aid,
        "flowTime": ft,

        # ✅ 新增字段：银行账户列表（你要求的结构）
        "bank_accounts": bank_accounts,

        # ✅ 交易明细结构不变，只是去掉 account_type
        "illion_raw_transactions": txn_records,

        # ✅ 余额明细结构不变
        "illion_day_end_balances": bal_records
    }

# =====================================================
# 6. 多进程共享索引
# =====================================================
_TXN_INDEX = None
_BAL_INDEX = None
_RUN_MODE = None
_RUN_FUNC = None

def _init_worker(txn_index, bal_index, run_mode, run_func):
    global _TXN_INDEX, _BAL_INDEX, _RUN_MODE, _RUN_FUNC
    _TXN_INDEX = txn_index
    _BAL_INDEX = bal_index
    _RUN_MODE = run_mode
    _RUN_FUNC = run_func

# =====================================================
# 7. 单样本处理
# =====================================================
def process_one_sample(r):
    global _TXN_INDEX, _BAL_INDEX, _RUN_MODE, _RUN_FUNC
    try:
        input_data = build_input_data(r, _TXN_INDEX, _BAL_INDEX)
        res = _RUN_FUNC(input_vars=input_data)

        out = dict(r)

        if isinstance(res, dict):
            if _RUN_MODE == "feature" and "features" in res:
                if isinstance(res["features"], dict):
                    out.update(res["features"])
            else:
                for k, v in res.items():
                    if isinstance(v, dict):
                        out.update(v)
                    else:
                        out[k] = v

        return out, None

    except Exception as e:
        out = dict(r)
        out["feature_error"] = str(e)

        err_detail = {
            "user_id": r.get("user_id"),
            "application_id": r.get("application_id"),
            "error_message": str(e),
            "input_data_json": json.dumps(
                build_input_data(r, _TXN_INDEX, _BAL_INDEX),
                ensure_ascii=False
            )
        }
        return out, err_detail

    finally:
        gc.collect()

# =====================================================
# 8. 对外唯一入口
# =====================================================
def run_feature_pipeline(
    run_mode,
    run_func,
    sample_path,
    txn_dir,
    bal_dir,
    out_path,
    error_path,
):
    logger.info("🚀 加载 samples")
    df = pd.read_csv(sample_path)
    df["sample_datetime"] = df["sample_datetime"].astype(str)

    done_keys = set()
    if os.path.exists(out_path):
        logger.info("🔁 启用断点续跑")
        done_df = pd.read_csv(out_path, usecols=["user_id", "application_id"])
        done_keys = set(zip(done_df.user_id, done_df.application_id))

    records = [
        r for r in df.to_dict("records")
        if (r["user_id"], r["application_id"]) not in done_keys
    ]

    if not records:
        logger.info("✅ 无待跑样本")
        return

    txn_index = build_csv_index(txn_dir)
    bal_index = build_csv_index(bal_dir)

    n_workers = max(cpu_count() - 4, 1)

    base_out_fields = list(df.columns)
    if "feature_error" not in base_out_fields:
        base_out_fields.append("feature_error")

    err_fields = ["user_id", "application_id", "error_message", "input_data_json"]
    out_fieldnames = base_out_fields[:]

    with open(out_path, "a", encoding="utf-8", newline="") as fout, \
         open(error_path, "a", encoding="utf-8", newline="") as ferr:

        out_writer = None
        err_writer = csv.DictWriter(ferr, fieldnames=err_fields, extrasaction="ignore")
        if os.path.getsize(error_path) == 0:
            err_writer.writeheader()

        with Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(txn_index, bal_index, run_mode, run_func)
        ) as pool:
            for out, err in tqdm(
                pool.imap_unordered(process_one_sample, records, chunksize=50),
                total=len(records),
                desc="Processing samples"
            ):
                for k in out.keys():
                    if k not in out_fieldnames:
                        out_fieldnames.append(k)

                if out_writer is None:
                    out_writer = csv.DictWriter(
                        fout,
                        fieldnames=out_fieldnames,
                        extrasaction="ignore"
                    )
                    if os.path.getsize(out_path) == 0:
                        out_writer.writeheader()

                out_writer.writerow(out)

                if err:
                    err_writer.writerow(err)

    logger.info(f"✅ 完成：{out_path}")
    logger.info(f"⚠️ 错误明细：{error_path}")

    