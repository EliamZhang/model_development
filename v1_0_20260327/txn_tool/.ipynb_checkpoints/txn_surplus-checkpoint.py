import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class SingleApplicationSurplusFeatureEngineer:
    def __init__(self, 
                 df: pd.DataFrame,
                 time_windows: List[int] = None):
        """
        初始化特征工程类
        """
        
        self.time_windows = sorted(time_windows) if time_windows else [7, 14, 28, 56, 84, 168, 182]
        self.centrelink_list = [
        'Centrelink Pension', 'Disability Pension', 'JobSeeker', 'Youth Allowance',
        'Parenting Payment', 'Carers Benefits', 'National Disability Insurance Scheme', 'Vet Affairs',
        'Family Benefits', 'Child Support', 'Parental Leave Pay', 
        'Mobility Allowance'
        ]

        self.features = {}
        self.df = self._map(df)
        self._prepare_data()
        #self.df.to_csv('mapped_df.csv')
        self.raw_df = self.df.copy()
        
        # 预计算窗口数据索引
        self._precompute_window_indices()

    def _map(self, df: pd.DataFrame, mapping_file: str = None) -> pd.DataFrame:
        """根据交易映射表进行映射"""
        import os
        df = df.copy()


        # 统一空值
        cols = ['account_type','category','dr_cr','third_party']

        df[cols] = df[cols].astype(str).apply(lambda x: x.str.strip()).replace({'': None, 'nan': None, 'None': None})

        df.fillna({'account_type': 'Unlabeled', 'category': 'Unlabeled', 'dr_cr': 'Unlabeled'}, inplace=True)

        # 使用当前文件路径
        if mapping_file is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            mapping_file = os.path.join(current_dir,'reference','finv_level','finv交易类别映射矩阵_0303.csv')

        #print(f"映射文件路径: {mapping_file}")
        if not os.path.exists(mapping_file):
            print(f"警告: 映射文件 {mapping_file} 不存在")
            return df

        # # 批量填充
        # df.fillna({'account_type': 'Unlabeled','category': 'Unlabeled','dr_cr': 'Unlabeled'}, inplace=True)


        file_ext = os.path.splitext(mapping_file)[1].lower()

        try:
            mapping_df = pd.read_csv(mapping_file) if file_ext == '.csv' else pd.read_excel(mapping_file) if file_ext in ['.xlsx','.xls'] else None
            if mapping_df is None:
                print(f"警告: 不支持的文件格式 {file_ext}")
                return df
        except Exception as e:
            print(f"读取映射文件失败: {e}")
            return df

        need_cols = ['dr_cr','category','account_type','tag_level1','tag_level2']
        missing_cols = [c for c in need_cols if c not in mapping_df.columns]
        if missing_cols:
            print(f"警告: 映射文件缺少字段: {missing_cols}")
            return df

        df = df.merge(mapping_df[need_cols], on=['dr_cr','category','account_type'], how='left')
        return df

    def _prepare_data(self):
        """预处理数据"""
        
        if 'sample_datetime' in self.df.columns:
            self.df["sample_datetime"] = pd.to_datetime(self.df["sample_datetime"])
            self.sample_datetime = self.df["sample_datetime"].iloc[0]
        else:
            self.sample_datetime = None
        
        if 'user_id' in self.df.columns:
            self.user_id = self.df['user_id'].iloc[0]
        else:
            self.user_id = None
            
        if 'transaction_date' in self.df.columns:
            self.df['transaction_date'] = pd.to_datetime(self.df['transaction_date'])
            self.df['month_num'] = self.df['transaction_date'].dt.year * 12 + self.df['transaction_date'].dt.month
        
        if 'transaction_date' in self.df.columns and 'sample_datetime' in self.df.columns:
            self.df["trac_days"] = (self.df["sample_datetime"].dt.floor('D') - self.df["transaction_date"]).dt.days
        
        if 'amount' in self.df.columns:
            self.df['amount'] = self.df['amount'].abs()
        
        if 'trac_days' in self.df.columns:
            # 先过滤再排序
            self.df = self.df[self.df['trac_days'] >= 0].copy()
            self.df.sort_values(by=["trac_days"], ascending=True, inplace=True)
            self.df.reset_index(drop=True, inplace=True)

    def _precompute_window_indices(self):
        """预计算每个窗口的数据索引"""
        if 'trac_days' not in self.df.columns:
            self.window_masks = {}
            return
            
        self.window_masks = {}
        for window in self.time_windows:
            self.window_masks[window] = self.df['trac_days'] <= window

    def _clean_tp_name(self, tp: str) -> str:
        """清洗交易对手名称，用于特征键名"""
        if not isinstance(tp, str):
            tp = str(tp)
        
        # 一次性完成所有替换
        import re
        cleaned = tp.lower().replace(' ', '').replace('/', '').replace('-', '')
        cleaned = cleaned.replace('.', '').replace('&', '').replace('@', '')
        cleaned = cleaned.replace('$', '').replace('%', '').replace('_', '')
        cleaned = re.sub(r'[^\w]', '', cleaned)
        
        return cleaned
    
    def cal_income(self) -> Dict:
        '''收入识别规则'''
        features = {}

        df = self.df.copy()
        is_income = df['tag_level1'] == 'INCOME'
        df = df[is_income].copy()
        
        # 向量化计算income_factor
        df['income_factor'] = 0.0
        
        wage_mask = (df['tag_level2'] == 'Wages') & (df['category'] == 'Wages')
        centrelink_mask = (df['tag_level2'] == 'Centrelink') & (df['third_party'].isin(self.centrelink_list))
        otherincome_mask = (df['tag_level2'] == 'Other Income') & (df['category'] != 'Internal Transfer')
        
        df.loc[wage_mask | centrelink_mask, 'income_factor'] = 1.0
        df.loc[otherincome_mask, 'income_factor'] = 0.8
        
        df['cal_income'] = df['amount'] * df['income_factor']
        
        # 批量计算所有窗口
        for window in self.time_windows:
            window_mask = df['trac_days'] <= window
            features[f'bank_txn_surplus_income_sum_{window}d'] = df.loc[window_mask, 'cal_income'].sum() if window_mask.any() else 0.0

        self.features.update(features)
        return features

    def cal_expense(self) -> Dict:
        '''支出识别规则'''
        #剔除Internal Transfer
        features = {}
        df = self.df.copy()
        expense_mask = (df['tag_level1'] == 'EXPENSE') & (df['category'] != 'Internal Transfer')
        #is_expense = df['tag_level1'] == 'EXPENSE'
        df = df[expense_mask].copy()
        
        # 批量计算所有窗口
        for window in self.time_windows:
            window_mask = df['trac_days'] <= window
            features[f'bank_txn_surplus_expense_sum_{window}d'] = df.loc[window_mask, 'amount'].sum() if window_mask.any() else 0.0

        self.features.update(features)
        return features

    # def cal_liability(self) -> Dict:
    #     '''负债识别规则'''
    #     features = {}
    #     df = self.df.copy()
    #     is_liability = df['tag_level1'] == 'LIABILITY'
    #     df = df[is_liability].copy()

    #     # 向量化计算liability_factor
    #     df['liability_factor'] = 0.0
    #     liability_mask = (df['tag_level2'] == 'Liability_Repayment') | (df['tag_level2'] == 'Dishonours Fee')
    #     df.loc[liability_mask, 'liability_factor'] = 1.0
        
    #     df['cal_liability'] = df['amount'] * df['liability_factor']
        
    #     # 批量计算所有窗口
    #     for window in self.time_windows:
    #         window_mask = df['trac_days'] <= window
    #         features[f'bank_txn_surplus_liability_sum_{window}d'] = df.loc[window_mask, 'cal_liability'].sum() if window_mask.any() else 0.0
        
    #     self.features.update(features)
    #     return features

    def cal_liability_repayment(self) -> Dict:
        """计算负债还款总额"""
        features = {}
        df = self.df.copy()
        # 筛选tag_level2为'liability_repayment'的交易
        df = df[(df['tag_level1'] == 'LIABILITY') & 
            (df['tag_level2'] == 'Liability_Repayment')]
        
        if not df.empty:
            for window in self.time_windows:
                window_mask = df['trac_days'] <= window
                features[f'bank_txn_surplus_liabilityrepayment_sum_{window}d'] = df.loc[window_mask, 'amount'].sum()
        else:
            for window in self.time_windows:
                features[f'bank_txn_surplus_liabilityrepayment_sum_{window}d'] = 0.0
        
        self.features.update(features)
        return features

    def cal_dishonours_fee(self) -> Dict:
        """计算拒付费总额"""
        features = {}
        df = self.df.copy()
        # 筛选tag_level2为'dishonours Fee'的交易
        df = df[(df['tag_level1'] == 'LIABILITY') & 
            (df['tag_level2'] == 'Dishonours Fee')]
        
        if not df.empty:
            for window in self.time_windows:
                window_mask = df['trac_days'] <= window
                # 使用sum()直接求和，空Series会返回0.0
                features[f'bank_txn_surplus_dishonoursfee_sum_{window}d'] = df.loc[window_mask, 'amount'].sum()
        else:
            # 如果没有数据，所有窗口都返回0
            for window in self.time_windows:
                features[f'bank_txn_surplus_dishonoursfee_sum_{window}d'] = 0.0
        
        self.features.update(features)
        return features

    def cal_liability_drawdown(self) -> Dict:
        """计算负债提取总额"""
        features = {}
        df = self.df.copy()
        # 筛选tag_level2为'liability_drawdown'的交易
        df = df[(df['tag_level1'] == 'LIABILITY') & 
            (df['tag_level2'] == 'Liability_Drawdown')]
        
        if not df.empty:
            for window in self.time_windows:
                window_mask = df['trac_days'] <= window
                features[f'bank_txn_surplus_liabilitydrawdown_sum_{window}d'] = df.loc[window_mask, 'amount'].sum()
        else:
            for window in self.time_windows:
                features[f'bank_txn_surplus_liabilitydrawdown_sum_{window}d'] = 0.0
        
        self.features.update(features)
        return features
    
    def cal_refund_or_reversal(self) -> Dict:
        """计算refund总额"""
        features = {}
        df = self.df.copy()
        # 筛选tag_level2为'Refund_Or_Reversal'的交易
        df = df[(df['tag_level1'] == 'LIABILITY') & 
            (df['tag_level2'] == 'Refund_Or_Reversal')]
    
        if not df.empty:
            for window in self.time_windows:
                window_mask = df['trac_days'] <= window
                features[f'bank_txn_surplus_refundorreversal_sum_{window}d'] = df.loc[window_mask, 'amount'].sum()
        else:
            for window in self.time_windows:
                features[f'bank_txn_surplus_refundorreversal_sum_{window}d'] = 0.0
        
        self.features.update(features)
        return features

    def cal_surplus(self) -> Dict:
        '''盈余识别规则'''
        features = {}
        df = self.df.copy()
        # 确保基础特征已计算
        if not any(k.startswith('bank_txn_surplus_income_sum_') for k in self.features):
            self.cal_income()
        if not any(k.startswith('bank_txn_surplus_expense_sum_') for k in self.features):
            self.cal_expense()
        # if not any(k.startswith('bank_txn_surplus_liability_sum_') for k in self.features):
        #     self.cal_liability()
        if not any(k.startswith('bank_txn_surplus_liabilityrepayment_sum_') for k in self.features):
            self.cal_liability_repayment()
        if not any(k.startswith('bank_txn_surplus_dishonoursfee_sum_') for k in self.features):
            self.cal_dishonours_fee()
        if not any(k.startswith('bank_txn_surplus_liabilitydrawdown_sum_') for k in self.features):
            self.cal_liability_drawdown()
        if not any(k.startswith('bank_txn_surplus_refundorreversal_sum_') for k in self.features):
            self.cal_refund_or_reversal()
    
        # 批量计算盈余
        for window in self.time_windows:
            income = self.features.get(f'bank_txn_surplus_income_sum_{window}d', 0)
            expense = self.features.get(f'bank_txn_surplus_expense_sum_{window}d', 0)
            #liability = self.features.get(f'bank_txn_surplus_liability_sum_{window}d', 0)
            liability_repayment = self.features.get(f'bank_txn_surplus_liabilityrepayment_sum_{window}d', 0)
            dishonours_fee = self.features.get(f'bank_txn_surplus_dishonoursfee_sum_{window}d', 0)
            liability_drawdown = self.features.get(f'bank_txn_surplus_liabilitydrawdown_sum_{window}d', 0)
            
            if pd.isna(income) or pd.isna(expense) or pd.isna(liability_repayment) or pd.isna(dishonours_fee) or pd.isna(liability_drawdown):
                features[f'bank_txn_surplus_surplus_sum_{window}d'] = 0.0
            else:
                features[f'bank_txn_surplus_surplus_sum_{window}d'] = income - expense - liability_repayment - dishonours_fee + liability_drawdown

        self.features.update(features)
        return features

    def cal_fortnightly_metrics(self) -> Dict:
        """计算双周化口径的财务指标"""
        features = {}
        
        # 确保基础指标已计算
        if not any(k.startswith('bank_txn_surplus_income_sum_') for k in self.features):
            self.cal_income()
        if not any(k.startswith('bank_txn_surplus_expense_sum_') for k in self.features):
            self.cal_expense()
        # if not any(k.startswith('bank_txn_surplus_liability_sum_') for k in self.features):
        #     self.cal_liability()
        if not any(k.startswith('bank_txn_surplus_liabilityrepayment_sum_') for k in self.features):
            self.cal_liability_repayment()
        if not any(k.startswith('bank_txn_surplus_dishonoursfee_sum_') for k in self.features):
            self.cal_dishonours_fee()
        if not any(k.startswith('bank_txn_surplus_liabilitydrawdown_sum_') for k in self.features):
            self.cal_liability_drawdown()
        if not any(k.startswith('bank_txn_surplus_refundorreversal_sum_') for k in self.features):
            self.cal_refund_or_reversal()

        df = self.df.copy()

        # 预计算每个时间窗口的各类交易天数间隔
        for window in self.time_windows:
            window_mask = df['trac_days'] <= window
            window_df = df[window_mask]
            categories = {
            'income': (window_df['tag_level1']=='INCOME'),
            'expense': (window_df['tag_level1']=='EXPENSE'),
            'liabilityrepayment': (window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Liability_Repayment'),
            'liabilitydrawdown':(window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Liability_Drawdown'),
            'dishonoursfee':(window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Dishonours Fee'),
            'refundorreversal':(window_df['tag_level1'] == 'LIABILITY') & (window_df['tag_level2'] == 'Refund_Or_Reversal')
            }
            factors = {}
            for cat, mask in categories.items():
                cat_df = window_df[mask]
                if not cat_df.empty:
                    first_day = cat_df['trac_days'].min()
                    last_day = cat_df['trac_days'].max()
                    day_span = max(last_day - first_day, 14)  # 至少为14天(周期)
                    factors[cat] = 14.0 / day_span
                else:
                    factors[cat] = 0
            # 获取原始总和值
            income_val = self.features.get(f'bank_txn_surplus_income_sum_{window}d', 0)
            expense_val = self.features.get(f'bank_txn_surplus_expense_sum_{window}d', 0)
            liability_repayment_val = self.features.get(f'bank_txn_surplus_liabilityrepayment_sum_{window}d', 0)
            liability_drawdown_val = self.features.get(f'bank_txn_surplus_liabilitydrawdown_sum_{window}d', 0)
            dishonours_fee_val = self.features.get(f'bank_txn_surplus_dishonoursfee_sum_{window}d', 0)
            refund_or_reversal_val = self.features.get(f'bank_txn_surplus_refundorreversal_sum_{window}d', 0)

            for cat, val in [('income', income_val), ('expense', expense_val), ('liabilityrepayment', liability_repayment_val),('liabilitydrawdown',liability_drawdown_val),
                             ('dishonoursfee',dishonours_fee_val),('refundorreversal',refund_or_reversal_val)]:
                if not pd.isna(factors[cat]) and not pd.isna(val):
                    features[f'bank_txn_surplus_fortnightly_{cat}_{window}d'] = val * factors[cat]
                else:
                    features[f'bank_txn_surplus_fortnightly_{cat}_{window}d'] = 0

            # 计算双周化盈余
            factors_valid = all(not pd.isna(factors[cat]) for cat in ['income', 'expense', 'liabilityrepayment','liabilitydrawdown','dishonoursfee'])
            values_valid = all(not pd.isna(val) for val in [income_val, expense_val, liability_repayment_val, liability_drawdown_val, dishonours_fee_val])
            if factors_valid and values_valid:
                fortnightly_surplus =  (income_val * factors['income']) - (expense_val * factors['expense']) - (liability_repayment_val * factors['liabilityrepayment']) - (dishonours_fee_val * factors['dishonoursfee']) + (liability_drawdown_val * factors['liabilitydrawdown'])
                features[f'bank_txn_surplus_fortnightly_surplus_{window}d'] = fortnightly_surplus
            else:
                features[f'bank_txn_surplus_fortnightly_surplus_{window}d'] = 0



        
        # 向量化批量计算
        # windows_array = np.array(self.time_windows)
        # fortnightly_factor = 14.0 / windows_array
        
        # for i, window in enumerate(self.time_windows):
        #     income_val = self.features.get(f'bank_txn_surplus_income_sum_{window}d')
        #     expense_val = self.features.get(f'bank_txn_surplus_expense_sum_{window}d')
        #     liability_val = self.features.get(f'bank_txn_surplus_liability_sum_{window}d')
            
        #     factor = fortnightly_factor[i]
            
        #     features[f'bank_txn_surplus_fortnightly_income_{window}d'] = income_val * factor if not pd.isna(income_val) else np.nan
        #     features[f'bank_txn_surplus_fortnightly_expense_{window}d'] = expense_val * factor if not pd.isna(expense_val) else np.nan
        #     features[f'bank_txn_surplus_fortnightly_liability_{window}d'] = liability_val * factor if not pd.isna(liability_val) else np.nan
            
        #     if not any(pd.isna([income_val, expense_val, liability_val])):
        #         features[f'bank_txn_surplus_fortnightly_surplus_{window}d'] = (income_val - expense_val - liability_val) * factor
        #     else:
        #         features[f'bank_txn_surplus_fortnightly_surplus_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def amount_comparison(self, windows: List[int] = [7, 30, 90, 180]) -> Dict:
        """计算收入、支出、负债、盈余的折算后趋势特征"""
        features = {}
        
        if not hasattr(self, 'df') or self.df is None:
            return features
        
        df = self.df.copy()
        
        comparisons = [
            (7, 30, '1w_vs_1m'),
            (30, 90, '1m_vs_3m'),
            (30, 180, '1m_vs_6m'),
            (90, 180, '3m_vs_6m')
        ]
        
        all_windows = set()
        for w1, w2, _ in comparisons:
            all_windows.add(w1)
            all_windows.add(w2)
        
        # 预计算所有窗口的mask
        window_masks = {w: df['trac_days'] <= w for w in all_windows}
        
        # 预计算tag_level过滤
        income_mask = df['tag_level1'] == 'INCOME'
        expense_mask = (df['tag_level1'] == 'EXPENSE') & (df['category'] != 'Internal Transfer')
        #liability_mask = df['tag_level1'] == 'LIABILITY'
        
        # 预计算收入因子
        df['income_factor'] = 0.0
        wage_cond = (df['tag_level2'] == 'Wages') & (df['category'] == 'Wages')
        centrelink_cond = (df['tag_level2'] == 'Centrelink') & (df['third_party'].isin(self.centrelink_list))
        otherincome_cond = (df['tag_level2'] == 'Other Income') & (df['category'] != 'Internal Transfer')
        
        df.loc[income_mask & (wage_cond | centrelink_cond), 'income_factor'] = 1.0
        df.loc[income_mask & otherincome_cond, 'income_factor'] = 0.8

        # # 预计算支出因子
        # df['expense_factor'] = 0.0
        # expense_cond = (df['tag_level1'] == 'EXPENSE') & (df['category'] != 'Internal Transfer')
        
        # df.loc[expense_cond, 'expense_factor'] = 1.0
        
        # # 预计算负债因子
        # df['liability_factor'] = 0.0
        # liability_cond = (df['tag_level2'] == 'Liability_Repayment') | (df['tag_level2'] == 'Dishonours Fee')
        # df.loc[liability_mask & liability_cond, 'liability_factor'] = 1.0
        
        # 批量计算所有窗口数据
        window_data = {}
        for window in all_windows:
            mask = window_masks[window]
            window_df = df[mask]
            
            income = (window_df.loc[income_mask[mask], 'amount'] * window_df.loc[income_mask[mask], 'income_factor']).sum() if income_mask[mask].any() else 0.0
            expense = window_df.loc[expense_mask[mask], 'amount'].sum() if expense_mask[mask].any() else 0.0
            liability_repayment = window_df.loc[(window_df['tag_level2'] == 'Liability_Repayment'), 'amount'].sum()
            dishonours_fee = window_df.loc[(window_df['tag_level2'] == 'Dishonours Fee'), 'amount'].sum()
            liability_drawdown = window_df.loc[(window_df['tag_level2'] == 'Liability_Drawdown'), 'amount'].sum()
            refund_reversal = window_df.loc[(window_df['tag_level2'] == 'Refund_Or_Reversal'), 'amount'].sum()

            
            
            surplus = income - expense - liability_repayment - dishonours_fee + liability_drawdown
            
            window_data[window] = {
                'income': income,
                'expense': expense,
                'surplus': surplus,
                'liabilityrepayment': liability_repayment,
                'dishonoursfee': dishonours_fee,
                'liabilitydrawdown': liability_drawdown,
                'refundorreversal': refund_reversal
            }
        
        # 批量计算比值
        metrics = ['income', 'expense','liabilityrepayment','dishonoursfee',
                    'liabilitydrawdown','refundorreversal','surplus']
        
        for metric_name in metrics:
            for w1, w2, comp_name in comparisons:
                feature_key = f'bank_txn_surplus_{metric_name}_{comp_name}'
                
                val1 = window_data[w1][metric_name] / w1
                val2 = window_data[w2][metric_name] / w2
                
                if val2 != 0:
                    features[feature_key] = val1 / val2
                else:
                    features[feature_key] = np.nan
        
        self.features.update(features)
        return features

    def amount_by_type(self) -> Dict:
        """按负债类型的统计特征"""
        #修改特征变量名称 liability修改为surplus
        features = {}
        df = self.df.copy()
        df = df[df['tag_level1'] == 'LIABILITY'].copy()
        
        liability_types = ['Liability_Repayment', 'Liability_Drawdown', 'Refund_Or_Reversal', 'Dishonours Fee']
        
        # 预清理类型名称
        cleaned_types = {lt: self._clean_tp_name(lt) for lt in liability_types}
        
        for window in self.time_windows:
            window_data = df[df['trac_days'] <= window]
            
            if len(window_data) == 0:
                for liability_type in liability_types:
                    cleaned_type = cleaned_types[liability_type]
                    for stat in ['mean', 'median', 'max', 'min', 'std', 'cv']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = np.nan
                    for stat in ['sum','count']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = 0.0
                continue
            
            for liability_type in liability_types:
                cleaned_type = cleaned_types[liability_type]
                type_data = window_data[window_data['tag_level2'] == liability_type]
                
                if len(type_data) > 0:
                    amounts = type_data['amount'].values
                    
                    # 批量计算统计量
                    total_sum = np.sum(amounts)
                    total_count = len(amounts)
                    total_mean = np.mean(amounts)
                    total_median = np.median(amounts)
                    total_max = np.max(amounts)
                    total_min = np.min(amounts)
                    total_std = np.std(amounts, ddof=1) if total_count > 1 else np.nan
                    total_cv = total_std / total_mean if total_mean > 0 and not np.isnan(total_std) else np.nan
                    
                    # 批量赋值
                    stats = {
                        'sum': total_sum,
                        'count': total_count,
                        'mean': total_mean,
                        'median': total_median,
                        'max': total_max,
                        'min': total_min,
                        'std': total_std,
                        'cv': total_cv
                    }
                    
                    for stat_name, stat_val in stats.items():
                        features[f'bank_txn_surplus_{cleaned_type}_{stat_name}_{window}d'] = stat_val
                else:
                    for stat in [ 'mean', 'median', 'max', 'min', 'std', 'cv']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = np.nan
                    for stat in ['sum', 'count']:
                        features[f'bank_txn_surplus_{cleaned_type}_{stat}_{window}d'] = 0.0
        
        self.features.update(features)
        return features

    def generate_all_features(self, feature_groups: List[str] = None, include_metadata: bool = True) -> Dict:
        """生成所有特征"""
        
        if feature_groups is None:
            feature_groups = [
                'cal_income', 'cal_expense',  'cal_surplus',
                'cal_fortnightly_metrics', 'amount_comparison', 'amount_by_type'
            ]
        
        # 重置特征字典
        self.features = {}
        
        # 依次生成特征
        for feature in feature_groups:
            method = getattr(self, feature)
            method()
        
        # 添加基本信息
        self.features['user_id'] = self.user_id
        self.features['sample_datetime'] = self.sample_datetime

        df = pd.DataFrame([self.features])
        base_columns = ['user_id', 'sample_datetime']
        existing_base = [col for col in base_columns if col in df.columns]
        feature_cols = [col for col in df.columns if col not in existing_base]
        
        df = df[existing_base + feature_cols]
        df.columns = df.columns.str.strip().str.replace(' ', '').str.lower()
        
        return df

#==========================================================================================
def generate_surplus_features(df: pd.DataFrame = None, feature_groups: List[str] = None, include_metadata: bool = True):
    engineer = SingleApplicationSurplusFeatureEngineer(
        df=df,
        time_windows=[7, 14, 28, 56, 84, 168, 182])
    result_df = engineer.generate_all_features(feature_groups=feature_groups)
    return result_df