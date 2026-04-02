# coding: utf-8
from v1_0_20251201.run_model import generate_score   # ← 修改为你的模块名，例如 main、risk_model 等


# =====================================================
#  Test Case 1: 正常数据（流水 + 余额都不为空）
# =====================================================
input_data_normal = {
    "userId": 10882774,
    "applicationId": 123123,
    "flowTime": "2025-12-24 13:41:11.0",

    # ========== 示例：2 条 transaction 明细 ==========
    "illion_raw_transactions": [
        {
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
        },
        {
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

    # ========== 示例：2 条余额记录 ==========
    "illion_day_end_balances":[
        {
            "job_id": "123",
            "balance": -42560.72,
            "balance_date": "2025-09-28",
            "balance_id": "1.07723905e+09",
            "bank_account_id": "3.948332e+06]"
        },
        {
            "job_id": "123",
            "balance": -42144.71,
            "balance_date": "2025-09-27",
            "balance_id": "1.077239051e+09",
            "bank_account_id": "3.948332e+06]"
        }
    ]
}

# =====================================================
#  Test Case 2: 空数据输入（用于测试 score = -1）
# =====================================================
input_data_empty = {
    "userId": 999999,
    "applicationId": 777777,
    "flowTime": "2025-12-24 13:41:11.0",

    # ⭐ 两个数组都为空
    "illion_raw_transactions": [],
    "illion_day_end_balances": []
}


# =====================================================
# ======== 调用你的算法（Case 1：正常数据） ========
# =====================================================
print("\n" + "="*80)
print("🚀 Test Case 1：正常数据（应该返回模型实际分数）")
print("="*80)

result1 = generate_score(input_vars=input_data_normal)
print(result1)


# =====================================================
# ======== 调用你的算法（Case 2：空数据 -> 应返回 -1） ========
# =====================================================
print("\n" + "="*80)
print("🚀 Test Case 2：空数据输入（应该返回 -1 + features=None）")
print("="*80)

result2 = generate_score(input_vars=input_data_empty)
print(result2)


# =====================================================
# ======== Feature 打印逻辑（只对正常数据跑） ========
# =====================================================
def group_feature(key):
    if "balance" in key:
        return "Balance 余额特征"
    if "category" in key:
        return "Category 分类特征"
    if "income" in key:
        return "Income 收入特征"
    if "lender" in key:
        return "Lender 借贷特征"
    return "Other 其他特征"


# 正常数据的 feature 打印
if result1.get("aus_new_risk_bid_3rdmodel_v1_0_20251201_features") is not None:
    fea = result1["aus_new_risk_bid_3rdmodel_v1_0_20251201_features"]

    print("\n" + "="*60)
    print("📌 模型特征（Features）")
    print("="*60)

    groups = {}
    for k, v in fea.items():
        g = group_feature(k)
        groups.setdefault(g, []).append((k, v))

    for g in sorted(groups.keys()):
        print(f"\n🔸 {g}")
        print("-" * 60)
        for k, v in sorted(groups[g]):
            print(f"{k}: {v}")

print("\n🎉 运行成功！两个案例已验证完毕。\n")
