import pandas as pd
import numpy as np

# 定义时间窗口和category类型
time_windows = [31, 62, 92, 123, 153, 184]
categories = {
    'wage': ['Wages'],
    'centrelink': ['Centrelink'], 
    'income': ['Wages', 'Centrelink']
}

# Part 1: 使用pivot_table和groupby进行高效聚合
def calculate_basic_metrics_efficient(df):
    # 创建结果DataFrame的基础结构
    unique_combinations = df[['user_id', 'sample_datetime']].drop_duplicates()
    result_df = unique_combinations.copy()
    
    # 为每个category和每个时间窗口计算指标
    for category_name, category_values in categories.items():
        # 筛选当前category的数据
        category_data = df[df['category'].isin(category_values)].copy()
        
        for window in time_windows:
            # 筛选当前时间窗口的数据
            window_data = category_data[category_data['trac_days'] <= window]
            
            # 使用groupby进行高效聚合
            grouped = window_data.groupby(['user_id', 'sample_datetime'])
            
            # 计算总额
            amount_sum = grouped['amount'].sum().reset_index()
            amount_sum = amount_sum.rename(columns={'amount': f'bank_txn_income_{category_name}_amount_{window}d'})
            
            # 计算条数
            count = grouped.size().reset_index(name=f'bank_txn_income_{category_name}_count_{window}d')
            
            # 计算最大金额
            max_amount = grouped['amount'].max().reset_index()
            max_amount = max_amount.rename(columns={'amount': f'bank_txn_income_{category_name}_maxamount_{window}d'})
            
            # 合并到结果DataFrame
            result_df = result_df.merge(amount_sum, on=['user_id', 'sample_datetime'], how='left')
            result_df = result_df.merge(count, on=['user_id', 'sample_datetime'], how='left')
            result_df = result_df.merge(max_amount, on=['user_id', 'sample_datetime'], how='left')
    
    return result_df



# Part 2: 衍生类变量计算，并在函数内部处理空值
def calculate_derived_metrics_efficient(basic_df):
    derived_df = basic_df.copy()
    
    for category in ['wage', 'centrelink', 'income']:
        # 1. 收入总额比例
        proportion_pairs = [
            (31, 62), (31, 92), (31, 184),
            (62, 123), (62, 184), (92, 184)
        ]
        
        for window1, window2 in proportion_pairs:
            col1 = f'bank_txn_income_{category}_amount_{window1}d'
            col2 = f'bank_txn_income_{category}_amount_{window2}d'
            proportion_col = f'bank_txn_income_{category}_amountproportion_{window1}d_{window2}d'
            
            derived_df[proportion_col] = (
                derived_df[col1] / derived_df[col2].replace(0, np.nan)
            )
            # 在函数内部处理空值
            derived_df[proportion_col] = derived_df[proportion_col].fillna(-99999)
        
        # 2. 计算单月金额
        monthly_windows = [
            (1, 31, '31d'),
            (32, 62, '32_62d'),
            (63, 92, '63_92d'),
            (93, 123, '93_123d'),
            (124, 153, '124_153d'),
            (154, 184, '154_184d')
        ]
        
        for min_days, max_days, window_name in monthly_windows:
            col_name = f'bank_txn_income_{category}_amount_{window_name}'
            if min_days == 1:
                derived_df[col_name] = derived_df[f'bank_txn_income_{category}_amount_{max_days}d']
            else:
                derived_df[col_name] = (
                    derived_df[f'bank_txn_income_{category}_amount_{max_days}d'] - 
                    derived_df[f'bank_txn_income_{category}_amount_{min_days-1}d']
                )
            # 处理空值
            derived_df[col_name] = derived_df[col_name].fillna(-99999)
        
        # 3. 变异系数计算
        monthly_cols = [f'bank_txn_income_{category}_amount_{window_name}' for _, _, window_name in monthly_windows]
        
        for total_days, month_name in [(92, '3'), (123, '4'), (153, '5'), (184, '6')]:
            relevant_cols = [col for col in monthly_cols if int(col.split('_')[-1].replace('d', '')) <= total_days]
            if relevant_cols:
                cv_col = f'bank_txn_income_{category}_amountcv_past{month_name}months'
                means = derived_df[relevant_cols].mean(axis=1)
                stds = derived_df[relevant_cols].std(axis=1)
                derived_df[cv_col] = stds / means.replace(0, np.nan)
                # 处理空值
                derived_df[cv_col] = derived_df[cv_col].fillna(-99999)
        
        # 4. 单月最大收入占比
        monthly_cols_6m = monthly_cols
        monthly_cols_3m = monthly_cols[:3]
        
        max_amount_6m = derived_df[monthly_cols_6m].max(axis=1)
        max_amount_3m = derived_df[monthly_cols_3m].max(axis=1)
        
        prop_6m_col = f'bank_txn_income_{category}_maxamount_singlemth_propotion_6mths'
        prop_3m_col = f'bank_txn_income_{category}_maxamount_singlemth_propotion_3mths'
        
        derived_df[prop_6m_col] = (
            max_amount_6m / derived_df[f'bank_txn_income_{category}_amount_184d'].replace(0, np.nan)
        )
        derived_df[prop_3m_col] = (
            max_amount_3m / derived_df[f'bank_txn_income_{category}_amount_92d'].replace(0, np.nan)
        )
        # 处理空值
        derived_df[prop_6m_col] = derived_df[prop_6m_col].fillna(-99999)
        derived_df[prop_3m_col] = derived_df[prop_3m_col].fillna(-99999)
        
        # 5. 环比增长次数
        monthly_cols_ordered = [f'bank_txn_income_{category}_amount_{window_name}' for _, _, window_name in monthly_windows]
        increment_col = f'bank_txn_income_last6mth_monm_{category}_amount_incrementnumber'
        
        increment_counts = pd.Series(0, index=derived_df.index)
        for i in range(len(monthly_cols_ordered) - 1):
            diff = derived_df[monthly_cols_ordered[i]] - derived_df[monthly_cols_ordered[i + 1]]
            increment_counts += (diff > 0).astype(int)
        
        derived_df[increment_col] = increment_counts
        
        # 6. 有发薪的月份数
        for total_days, month_name in [(92, '3'), (123, '4'), (153, '5'), (184, '6')]:
            mthcount_col = f'bank_txn_income_last{month_name}mth_{category}_mthcount'
            relevant_cols = monthly_cols_ordered[:int(month_name)]
            derived_df[mthcount_col] = (
                derived_df[relevant_cols] > 0
            ).sum(axis=1)
    
    # 最后确保所有数值列的空值都被处理
    numeric_columns = derived_df.select_dtypes(include=[np.number]).columns
    derived_df[numeric_columns] = derived_df[numeric_columns].fillna(-99999)
    
    return derived_df

# 变量计算
def calculate_inm(df):
    unique_combinations = df[['user_id', 'sample_datetime']].drop_duplicates()
    # 前置处理
    df['amount'] = abs(df['amount'])
    df['trac_days'] = (df['sample_datetime'] - df['transaction_date']).dt.days
    # df['trac_days'] = [(pd.to_datetime(df['sample_datetime']) - x).days for x in df["transaction_date"]]
    df = df.loc[(df['dr_cr']=='credit') & (df['category'].isin(['Wages', 'Centrelink', 'ALL Other Credits'])) & (df['amount']>0)]

    # 计算基础指标
    basic_metrics = calculate_basic_metrics_efficient(df)
    # 计算衍生指标（函数内部已处理空值）
    final_result = calculate_derived_metrics_efficient(basic_metrics)

    # 结果关联
    df_res = pd.merge(unique_combinations, final_result, on=['user_id', 'sample_datetime'], how='left').fillna(-99999)
    return df_res