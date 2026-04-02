      
# coding: utf-8
import os
import gc
import glob
import json
import csv
import logging
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import pandas as pd
from tqdm import tqdm

from v1_0_20260327.run_model import generate_score

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

# =====================================================
# 2. 常量
# =====================================================
#CHUNK_SIZE = 500_000
#WRITE_BATCH = 200
#POOL_CHUNK = 50

CHUNK_SIZE = 200_000
WRITE_BATCH = 1000
POOL_CHUNK = 200

TXN_USECOLS = [
    "user_id", "application_id",
    "amount", "balance", "bank_account_id", "account_type","category",
    "dr_cr", "illion_trx_uuid", "text", "third_party",
    "transaction_date", "transaction_id", "trx_type"
]

BAL_USECOLS = [
    "user_id", "application_id",
    "job_id", "balance", "balance_date",
    "balance_id", "bank_account_id"
]

# =====================================================
# 3. 工具函数
# =====================================================
def is_valid_float(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def is_valid_balance_record(rec: dict) -> bool:
    return any(is_valid_float(rec.get(f)) for f in ["balance"])


# =====================================================
# 4. 索引构建
# =====================================================
def build_csv_index(csv_dir: str):
    logger.info(f"📚 构建索引：{csv_dir}")
    index = defaultdict(set)

    for fp in glob.glob(os.path.join(csv_dir, "*.csv")):
        try:
            for chunk in pd.read_csv(
                fp,
                usecols=["user_id", "application_id"],
                chunksize=CHUNK_SIZE
            ):
                uniq = chunk.drop_duplicates(subset=["user_id", "application_id"])
                for uid, aid in uniq.to_numpy():
                    index[(uid, aid)].add(fp)
        except Exception as e:
            logger.warning(f"⚠️ 索引失败跳过：{fp} | {e}")

    logger.info(f"✅ 索引完成：{len(index)} keys")
    return index


def load_match_rows_from_index(csv_index, user_id, application_id, usecols):
    fps = csv_index.get((user_id, application_id))
    if not fps:
        return pd.DataFrame(columns=usecols)

    matched = []
    for fp in fps:
        try:
            for chunk in pd.read_csv(fp, usecols=usecols, chunksize=CHUNK_SIZE):
                m = chunk[
                    (chunk["user_id"] == user_id)
                    & (chunk["application_id"] == application_id)
                ]
                if not m.empty:
                    matched.append(m)
        except Exception:
            continue

    return pd.concat(matched, ignore_index=True) if matched else pd.DataFrame(columns=usecols)


# =====================================================
# 5. 构造模型输入
# =====================================================
def build_input_data(r, txn_index, bal_index):
    uid, aid = r["user_id"], r["application_id"]
    ft = str(r["sample_datetime"])

    txn_df = load_match_rows_from_index(txn_index, uid, aid, TXN_USECOLS)
    bal_df = load_match_rows_from_index(bal_index, uid, aid, BAL_USECOLS)

    # =====================================================
    # 1) 交易明细：不再输出 account_type（account_type 上提到 bank_accounts）
    # =====================================================
    txn_records = []
    if not txn_df.empty:
        txn_records = (
            txn_df[
                [
                    "amount", "balance", "bank_account_id", "category",
                    "dr_cr", "illion_trx_uuid", "text", "third_party",
                    "transaction_date", "transaction_id", "trx_type"
                    # ✅ 注意：这里不包含 "account_type"
                ]
            ]
            .fillna("")
            .to_dict("records")
        )

    # =====================================================
    # 2) 余额明细：保持原结构不变
    # =====================================================
    bal_records = []
    if not bal_df.empty:
        raw_bal = (
            bal_df[
                ["job_id", "balance", "balance_date", "balance_id", "bank_account_id"]
            ]
            .fillna("")
            .to_dict("records")
        )
        for rec in raw_bal:
            if is_valid_balance_record(rec):
                bal_records.append(rec)

    # =====================================================
    # 3) 新增 bank_accounts：从 txn_df 汇总 (bank_account_id, account_type)
    #    并补齐：bal_df 中存在但 txn_df 中不存在的 bank_account_id -> account_type=""
    # =====================================================
    bank_accounts = []

    if not txn_df.empty:
        ba_df = (
            txn_df[["bank_account_id", "account_type"]]
            .fillna("")
            .drop_duplicates(subset=["bank_account_id", "account_type"])
        )
        bank_accounts = ba_df.to_dict("records")

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

        # ✅ 新结构：bank_accounts
        "bank_accounts": bank_accounts,

        # ✅ 交易明细：不含 account_type
        "illion_raw_transactions": txn_records,

        # ✅ 余额明细：原样
        "illion_day_end_balances": bal_records
    }


# =====================================================
# 6. 多进程 worker
# =====================================================
_TXN_INDEX = None
_BAL_INDEX = None


def _init_worker(txn_index, bal_index):
    global _TXN_INDEX, _BAL_INDEX
    _TXN_INDEX = txn_index
    _BAL_INDEX = bal_index


def process_one_sample(r):
    global _TXN_INDEX, _BAL_INDEX
    try:
        input_data = build_input_data(r, _TXN_INDEX, _BAL_INDEX)
        res = generate_score(input_vars=input_data)

        out = dict(r)
        if isinstance(res, dict):
            for k, v in res.items():
                if k.endswith("_features"):
                    if isinstance(v, dict):
                        out.update(v)
                    continue
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
# 7. Pipeline 主入口（写法二）
# =====================================================
def run_score_pipeline(
    sample_path: str,
    txn_dir: str,
    bal_dir: str,
    out_path: str,
    err_path: str,
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
    write_header = not os.path.exists(out_path)
    write_err_header = not os.path.exists(err_path)

    out_fields = list(df.columns)
    if "feature_error" not in out_fields:
        out_fields.append("feature_error")

    err_fields = [
        "user_id", "application_id",
        "error_message", "input_data_json"
    ]

    with open(out_path, "a", encoding="utf-8", newline="") as fout, \
         open(err_path, "a", encoding="utf-8", newline="") as ferr:

        out_writer = None
        err_writer = csv.DictWriter(ferr, fieldnames=err_fields)
        if write_err_header:
            err_writer.writeheader()

        with Pool(
            n_workers,
            initializer=_init_worker,
            initargs=(txn_index, bal_index)
        ) as pool:

            buffer_out, buffer_err = [], []

            for out, err in tqdm(
                pool.imap_unordered(process_one_sample, records, chunksize=POOL_CHUNK),
                total=len(records),
                desc="Processing samples"
            ):
                for k in out.keys():
                    if k not in out_fields:
                        out_fields.append(k)

                buffer_out.append(out)
                if err:
                    buffer_err.append(err)

                if len(buffer_out) >= WRITE_BATCH:
                    if out_writer is None:
                        out_writer = csv.DictWriter(
                            fout,
                            fieldnames=out_fields,
                            extrasaction="ignore"
                        )
                        if write_header:
                            out_writer.writeheader()
                            write_header = False

                    out_writer.writerows(buffer_out)
                    fout.flush()
                    buffer_out.clear()

                    if buffer_err:
                        err_writer.writerows(buffer_err)
                        ferr.flush()
                        buffer_err.clear()

            if buffer_out:
                if out_writer is None:
                    out_writer = csv.DictWriter(
                        fout,
                        fieldnames=out_fields,
                        extrasaction="ignore"
                    )
                    if write_header:
                        out_writer.writeheader()
                out_writer.writerows(buffer_out)

            if buffer_err:
                err_writer.writerows(buffer_err)

    logger.info(f"✅ 完成：{out_path}")
    logger.info(f"⚠️ 错误明细：{err_path}")

    