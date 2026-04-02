

import json

from v1_0_20251201.run_model_feature import (
    generate_score,
    run_model_feature
)

input_vars = {
  "userId": 10882774,
  "applicationId": -1,
  "flowTime": "2025-12-24 13:41:11.0",

  "bank_accounts": [
    {"bank_account_id": "3952166", "account_type": "Transaction"},
    {"bank_account_id": "3948332", "account_type": "CreditCard"}
  ],

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

  "illion_day_end_balances": [
    {
      "balance": -42560.72,
      "balance_date": "2025-09-28",
      "balance_id": "1.07723905e+09",
      "bank_account_id": "3948332"
    },
    {
      "balance": -42144.71,
      "balance_date": "2025-09-27",
      "balance_id": "1.077239051e+09",
      "bank_account_id": "3948332"
    }
  ]
}

print("========== run_model_feature ==========")

res_feature = run_model_feature(input_vars,dict_csv_path="./dict.csv")

print(json.dumps(res_feature, indent=2, default=str))


#print("\n========== generate_score ==========")

#res_score = generate_score(input_vars)

#print(json.dumps(res_score, indent=2, default=str))