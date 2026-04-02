import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class SingleApplicationExpenseFeatureEngineer:
    
    
    def __init__(self, 
                 df: pd.DataFrame,
                 time_windows: List[int] = None,
                 expense_type_tp_pairs: List[List] = None,
                 tag_level1: str = "EXPENSE"):
        """
        初始化特征工程类
        """
        
        
        self.original_df = df.copy()
        
        #交易类别映射表
        self.df = self._map(df)
        
        if self.df is None:
            self.df = pd.DataFrame()  # 初始化为空DataFrame
        
        if 'transaction_date' in self.df.columns and 'date' not in self.df.columns:
            self.df.rename(columns={'transaction_date': 'date'}, inplace=True)

        self.raw_df = self.df.copy()
        
        
        self.time_windows = sorted(time_windows) if time_windows else [7, 14, 28, 56, 84, 168, 182]
        
        # 处理二维数组支出类型和交易对手
        self.expense_type_tp_pairs = expense_type_tp_pairs if expense_type_tp_pairs is not None else []
        
        self.expense_types = [pair[0] for pair in self.expense_type_tp_pairs]
        
        self.third_parties = list(set(tp for pair in self.expense_type_tp_pairs for tp in pair[1] if tp))
        
        self.third_party_col = 'third_party'
        self.tag_level1 = tag_level1
        self.features = {}
        #self.raw_df = df.copy()
        
        self._prepare_data()
        

    def __call__(self, 
                 df: pd.DataFrame,
                 feature_groups: List[str] = None,
                 include_metadata: bool = True) -> pd.DataFrame:
        """
        使实例可以像函数一样被调用，直接生成特征
        """
        return self.generate_all_features(
            df=df,
            feature_groups=feature_groups,
            include_metadata=include_metadata
        )
    

    
    
    def _map(self, df: pd.DataFrame, mapping_file: str = './reference/finv_level/finv交易类别映射矩阵_0303.csv') -> pd.DataFrame:
        ''' 根据交易映射表进行映射'''
        import os

        df = df.copy()
        df['account_type'] = df['account_type'].fillna('Unlabeled')
        df['category'] = df['category'].fillna('Unlabeled')
        df['dr_cr'] = df['dr_cr'].fillna('Unlabeled')

        # ⭐ 新增：把相对路径转成当前文件所在目录的绝对路径
        if not os.path.isabs(mapping_file):
            base_dir = os.path.dirname(os.path.realpath(__file__))
            mapping_file = os.path.join(base_dir, mapping_file.lstrip("./"))

        if not os.path.exists(mapping_file):
            print(f"警告: 映射文件 {mapping_file} 不存在")
            return df

        # 根据文件扩展名读取映射文件
        file_ext = os.path.splitext(mapping_file)[1].lower()

        try:
            if file_ext == '.csv':
                mapping_df = pd.read_csv(mapping_file)
            elif file_ext in ['.xlsx', '.xls']:
                mapping_df = pd.read_excel(mapping_file)
            else:
                print(f"警告: 不支持的文件格式 {file_ext}")
                return df
        except Exception as e:
            print(f"读取映射文件失败: {e}")
            return df

        df = df.merge(
            mapping_df[['dr_cr', 'category', 'account_type', 'tag_level1', 'tag_level2']],
            on=['dr_cr', 'category', 'account_type'],
            how='left'
        )

        return df

    # 统计匹配情况

    def _clean_tp_name(self, tp: str) -> str:
        """清洗交易对手名称，用于特征键名"""
        if not isinstance(tp, str):
            tp = str(tp)
        
        # 转换为小写
        cleaned = tp.lower()
        
        # 去除空格
        cleaned = cleaned.replace(' ', '')
        
        # 替换 '/' 为 '_'
        cleaned = cleaned.replace('/', '')
        
        # 替换其他可能引起问题的特殊字符
        cleaned = cleaned.replace('-', '')
        cleaned = cleaned.replace('.', '')
        cleaned = cleaned.replace('&', '')
        cleaned = cleaned.replace('@', '')
        cleaned = cleaned.replace('$', '')
        cleaned = cleaned.replace('%', '')
        
        # 去除其他标点符号
        import re
        cleaned = re.sub(r'[^\w_]', '', cleaned)
        
        return cleaned

    def _prepare_data(self):
        """预处理数据"""
         # ==================== [新增] 兼容 transaction_date -> date ====================
        if 'date' not in self.df.columns:
            if 'transaction_date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['transaction_date']).dt.floor('D')
            else:
                raise KeyError("缺少日期列：需要 'date' 或 'transaction_date' 其中之一")
        else:
            self.df['date'] = pd.to_datetime(self.df['date']).dt.floor('D')

        # raw_df 也需要同样的 date（后面 expense_data = self.raw_df[...] 会用到）
        if 'date' not in self.raw_df.columns:
            if 'transaction_date' in self.raw_df.columns:
                self.raw_df['date'] = pd.to_datetime(self.raw_df['transaction_date']).dt.floor('D')
            else:
                # 如果 raw_df 没有 transaction_date，至少保证有 date，不然后面会 KeyError
                self.raw_df['date'] = self.df['date']
        else:
            self.raw_df['date'] = pd.to_datetime(self.raw_df['date']).dt.floor('D')
        # ==================== [新增结束] ====================
        #if 'created_at' in self.df.columns and 'sample_datetime' not in self.df.columns:
            #self.df.rename(columns={'created_at': 'sample_datetime'}, inplace=True)
        
        if 'transaction_date' in self.df.columns and 'date' not in self.df.columns:
            self.df.rename(columns={'transaction_date': 'date'}, inplace=True)

        if 'sample_datetime' in self.df.columns:
            self.df["sample_datetime"] = pd.to_datetime(self.df["sample_datetime"])
            self.sample_datetime = self.df["sample_datetime"].iloc[0]
        else:
            self.sample_datetime = None
        
        if 'user_id' in self.df.columns:
            self.user_id = self.df['user_id'].iloc[0]
        else:
            self.user_id = None
        
        if 'date' in self.df.columns:
            self.df['date'] = pd.to_datetime(self.df['date'])
            self.df['month_num'] = self.df['date'].dt.year * 12 + self.df['date'].dt.month
        
        if 'date' in self.df.columns and 'sample_datetime' in self.df.columns:
            self.df["trac_days"] = (self.df["sample_datetime"].dt.floor('D') - self.df["date"]).dt.days
        
        # 根据tag_level1筛选支出数据
        if 'tag_level1' in self.df.columns:
            expense_mask = self.df['tag_level1'] == self.tag_level1
            self.df = self.df[expense_mask].copy()
        else:
            print(f"警告: 数据中不存在'tag_level1'列，跳过筛选")
        
        # 支出金额取绝对值
        if 'amount' in self.df.columns:
            self.df['amount'] = self.df['amount'].abs()
        
        # 按trac_days排序（从新到旧）
        if 'trac_days' in self.df.columns:
            self.df = self.df.sort_values(by=["trac_days"], ascending=True).reset_index(drop=True)

        if 'trac_days' in self.df.columns:
            # 过滤掉负数trac_days（未来日期的交易）
            self.df = self.df[self.df['trac_days'] >= 0].copy()
    
        
        # 按支出类型缓存数据
        self._type_data = {}
        for expense_type in self.expense_types:
            if 'tag_level2' in self.df.columns:
                type_data = self.df[self.df['tag_level2'] == expense_type].copy()
                if len(type_data) > 0:
                    self._type_data[expense_type] = type_data
        
        # 按时间窗口缓存数据
        self._window_data = {}
        for window in self.time_windows:
            if 'trac_days' in self.df.columns:
                self._window_data[window] = self.df[self.df['trac_days'] <= window].copy()
            else:
                self._window_data[window] = pd.DataFrame()
        
        # 按交易对手缓存数据
        self._tp_data = {}
        
        # 按支出类型-交易对手组合缓存数据
        self._type_tp_data = {}
        for expense_type, tps in self.expense_type_tp_pairs:
            self._type_tp_data[expense_type] = {}
            type_data = self._type_data.get(expense_type, pd.DataFrame())
            
            # 如果交易对手列表为空，存储空DataFrame
            if not tps:
                continue
                
            for tp in tps:
                if 'third_party' in type_data.columns:
                    tp_data = type_data[type_data['third_party'] == tp].copy()
                    self._type_tp_data[expense_type][tp] = tp_data
                    # 同时维护全局交易对手数据
                    if tp not in self._tp_data:
                        self._tp_data[tp] = pd.DataFrame()
                    self._tp_data[tp] = pd.concat([self._tp_data[tp], tp_data])

    # ==================== amount 相关特征 ====================
    def amount_global(self) -> Dict:
        """全局支出统计特征"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) > 0:
                amounts = window_data['amount'].values
                
                total_sum = np.sum(amounts)
                total_count = len(amounts)
                total_mean = np.mean(amounts)
                total_median = np.median(amounts)
                total_max = np.max(amounts)
                total_min = np.min(amounts)
                total_std = np.std(amounts) 
                total_cv = total_std / total_mean 
                
                features[f'bank_txn_expense_global_sum_{window}d'] = total_sum
                features[f'bank_txn_expense_global_count_{window}d'] = total_count
                features[f'bank_txn_expense_global_mean_{window}d'] = total_mean
                features[f'bank_txn_expense_global_median_{window}d'] = total_median
                features[f'bank_txn_expense_global_max_{window}d'] = total_max
                features[f'bank_txn_expense_global_min_{window}d'] = total_min
                features[f'bank_txn_expense_global_std_{window}d'] = total_std
                features[f'bank_txn_expense_global_cv_{window}d'] = total_cv
            else:
                features[f'bank_txn_expense_global_sum_{window}d'] = np.nan
                features[f'bank_txn_expense_global_count_{window}d'] = np.nan
                features[f'bank_txn_expense_global_mean_{window}d'] = np.nan
                features[f'bank_txn_expense_global_median_{window}d'] = np.nan
                features[f'bank_txn_expense_global_max_{window}d'] = np.nan
                features[f'bank_txn_expense_global_min_{window}d'] = np.nan
                features[f'bank_txn_expense_global_std_{window}d'] = np.nan
                features[f'bank_txn_expense_global_cv_{window}d'] = np.nan
            
        
        self.features.update(features)
        return features

    def amount_by_type(self) -> Dict:
        """按支出类型的统计特征"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for expense_type in self.expense_types:
                    for stat in ['sum', 'count', 'mean', 'median', 'max', 'min', 'std', 'cv']:
                        features[f'bank_txn_expense_{expense_type}_{stat}_{window}d'] = np.nan
                continue
            
            for expense_type in self.expense_types:
                type_data = window_data[window_data['tag_level2'] == expense_type]
                
                if len(type_data) > 0:
                    amounts = type_data['amount'].values
                    
                    total_sum = np.sum(amounts)
                    total_count = len(amounts)
                    total_mean = np.mean(amounts)
                    total_median = np.median(amounts)
                    total_max = np.max(amounts)
                    total_min = np.min(amounts)
                    total_std = np.std(amounts) if total_count > 1 else np.nan
                    total_cv = total_std / total_mean if total_mean > 0 else np.nan
                    
                    features[f'bank_txn_expense_{expense_type}_sum_{window}d'] = total_sum
                    features[f'bank_txn_expense_{expense_type}_count_{window}d'] = total_count
                    features[f'bank_txn_expense_{expense_type}_mean_{window}d'] = total_mean
                    features[f'bank_txn_expense_{expense_type}_median_{window}d'] = total_median
                    features[f'bank_txn_expense_{expense_type}_max_{window}d'] = total_max
                    features[f'bank_txn_expense_{expense_type}_min_{window}d'] = total_min
                    features[f'bank_txn_expense_{expense_type}_std_{window}d'] = total_std
                    features[f'bank_txn_expense_{expense_type}_cv_{window}d'] = total_cv
                else:
                    features[f'bank_txn_expense_{expense_type}_sum_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_count_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_mean_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_median_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_max_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_min_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_std_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_cv_{window}d'] = np.nan
                
        
        self.features.update(features)
        return features

    def amount_daily(self) -> Dict:
        """单日最大最小支出额度"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for expense_type in self.expense_types:
                    for stat in ['max', 'min']:
                        features[f'bank_txn_expense_{expense_type}_{stat}_{window}d'] = np.nan
                continue
            
            for expense_type in self.expense_types:
                type_data = window_data[window_data['tag_level2'] == expense_type]
                type_data = type_data.groupby('date')['amount'].sum().reset_index()
                
                if len(type_data) > 0:
                    amounts = type_data['amount'].values
                    
                    total_max = np.max(amounts)
                    total_min = np.min(amounts)
                    
                    features[f'bank_txn_expense_{expense_type}_daily_max_{window}d'] = total_max
                    features[f'bank_txn_expense_{expense_type}_daily_min_{window}d'] = total_min
                else:
                    features[f'bank_txn_expense_{expense_type}_daily_max_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_daily_min_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def amount_comparison(self, windows: List[int] = [7,30, 90, 180]) -> Dict:
        """支出金额对比特征"""
        features = {}
        
        window_map = {7:'1w',30: '1m', 90: '3m', 180: '6m'}
        available_windows = [w for w in window_map.keys() if w in windows]
        
        if not available_windows:
            return features
        
        for expense_type in self.expense_types:
            type_data = self._type_data.get(expense_type)
            
            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_expense_{expense_type}_1w_vs_1m'] = np.nan
                features[f'bank_txn_expense_{expense_type}_1m_vs_3m'] = np.nan
                features[f'bank_txn_expense_{expense_type}_1m_vs_6m'] = np.nan
                features[f'bank_txn_expense_{expense_type}_3m_vs_6m'] = np.nan
                continue
            
            window_avgs = {}
            for window in available_windows:
                window_data = type_data[type_data['trac_days'] <= window]
                if len(window_data) > 0:
                    window_avgs[window] = np.mean(window_data['amount'].values)
                else:
                    window_avgs[window] = 0
            
            #latest = type_data['amount'].iloc[0] if len(type_data) > 0 else 0

            if 7 in window_avgs and 30 in window_avgs and window_avgs[30] > 0:
                features[f'bank_txn_expense_{expense_type}_1w_vs_1m'] = window_avgs[7] / window_avgs[30]
            else:
                features[f'bank_txn_expense_{expense_type}_1w_vs_1m'] = np.nan
            
            if 30 in window_avgs and 90 in window_avgs and window_avgs[90] > 0:
                features[f'bank_txn_expense_{expense_type}_1m_vs_3m'] = window_avgs[30] / window_avgs[90]
            else:
                features[f'bank_txn_expense_{expense_type}_1m_vs_3m'] = np.nan

            if 30 in window_avgs and 180 in window_avgs and window_avgs[90] > 0:
                features[f'bank_txn_expense_{expense_type}_1m_vs_6m'] = window_avgs[30] / window_avgs[180]
            else:
                features[f'bank_txn_expense_{expense_type}_1m_vs_6m'] = np.nan
            
            if 90 in window_avgs and 180 in window_avgs and window_avgs[180] > 0:
                features[f'bank_txn_expense_{expense_type}_3m_vs_6m'] = window_avgs[90] / window_avgs[180]
            else:
                features[f'bank_txn_expense_{expense_type}_3m_vs_6m'] = np.nan
        
        self.features.update(features)
        return features

    def growth_rates(self) -> Dict:
        '''
        最大支出涨跌幅
        '''
        features = {}
        for expense_type in self.expense_types:
            type_data = self._type_data.get(expense_type, pd.DataFrame())

            if 'date' not in type_data.columns:
                if 'transaction_date' in type_data.columns:
                    type_data = type_data.copy()
                    type_data['date'] = pd.to_datetime(type_data['transaction_date']).dt.floor('D')
                else:
                    features[f'bank_txn_expense_{expense_type}_growth_rate_max'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_growth_rate_min'] = np.nan
                    continue
            #print("[DEBUG] type_data cols =", list(type_data.columns), "rows=", len(type_data))

            daily_data = type_data.groupby('date')['amount'].sum().reset_index()
            daily_data = daily_data.sort_values('date', ascending=True) 
            amounts = daily_data['amount'].values if len(type_data) > 0 else []
            if len(amounts) > 1:
                prev_amounts = amounts[:-1]
                curr_amounts = amounts[1:]
                growth_rates = np.where(prev_amounts != 0, 
                               (curr_amounts - prev_amounts) / prev_amounts, 
                               0)
                features[f'bank_txn_expense_{expense_type}_growth_rate_max'] = float(np.max(growth_rates))
                features[f'bank_txn_expense_{expense_type}_growth_rate_min'] = float(np.min(growth_rates))
            else:
                features[f'bank_txn_expense_{expense_type}_growth_rate_max'] = np.nan 
                features[f'bank_txn_expense_{expense_type}_growth_rate_min'] = np.nan 
        self.features.update(features)
        return features

    #############     ratio       #####################
    def ratio(self) -> Dict:
        """支出占比特征"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for expense_type in self.expense_types:
                    features[f'bank_txn_expense_{expense_type}_ratio_{window}d'] = np.nan
                continue
            
            total_amount = np.sum(window_data['amount'].values)
            
            if total_amount > 0:
                for expense_type in self.expense_types:
                    type_data = window_data[window_data['tag_level2'] == expense_type]
                    type_sum = np.sum(type_data['amount'].values)
                    features[f'bank_txn_expense_{expense_type}_ratio_{window}d'] = type_sum / total_amount
            else:
                for expense_type in self.expense_types:
                    features[f'bank_txn_expense_{expense_type}_ratio_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def ratio_by_type(self) -> Dict:
        '''
        最大单笔支出占支出类型总支出比
        '''
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for expense_type in self.expense_types:
                    features[f'bank_txn_expense_{expense_type}_max_ratio_{window}d'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_min_ratio_{window}d'] = np.nan
                continue
            
            for expense_type in self.expense_types:
                type_data = window_data[window_data['tag_level2'] == expense_type]
                type_sum = np.sum(type_data['amount'].values) if len(type_data) > 0 else np.nan
                if type_sum > 0:
                    min_value = np.min(type_data['amount'].values) if len(type_data) > 0 else np.nan
                    max_value = np.max(type_data['amount'].values) if len(type_data) > 0 else np.nan
                    features[f'bank_txn_expense_{expense_type}_max_ratio_{window}d'] = max_value / type_sum 
                    features[f'bank_txn_expense_{expense_type}_min_ratio_{window}d'] = min_value / type_sum 
                else:
                    features[f'bank_txn_expense_{expense_type}_max_ratio_{window}d'] = np.nan 
                    features[f'bank_txn_expense_{expense_type}_min_ratio_{window}d'] = np.nan
            
        self.features.update(features)
        return features

    #############     stability 相关特征      #####################
    def max_consecutive_months(self) -> Dict:
        '''
        连续支出月份
        '''
        features = {}
        for expense_type in self.expense_types:
            type_data = self._type_data.get(expense_type)
            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_expense_{expense_type}_max_consecutive_months'] = np.nan
                continue
            
            monthly_data = (type_data.groupby(type_data['date'].dt.to_period('M'))['amount'].sum()
                           .reset_index()
                           .rename(columns={'date': 'year_month'}))
            
            monthly_data = monthly_data[monthly_data['amount'] > 0]

            if len(monthly_data) == 0:
                features[f'bank_txn_expense_{expense_type}_max_consecutive_months'] = np.nan
                continue

            monthly_data['year_month'] = monthly_data['year_month'].dt.to_timestamp()
            monthly_data = monthly_data.sort_values('year_month').reset_index(drop=True)
            monthly_data['month_num'] = monthly_data['year_month'].dt.year * 12 + monthly_data['year_month'].dt.month
        
            months = monthly_data['month_num'].values
            
            max_len = 1
            current_len = 1
            for i in range(1, len(months)):
                if months[i] - months[i-1] == 1:  
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 1
            features[f'bank_txn_expense_{expense_type}_max_consecutive_months'] = max_len
        
        self.features.update(features)
        return features

    def max_consecutive_days(self) -> Dict:
        '''
        连续支出天数
        '''
        features = {}
        for expense_type in self.expense_types:
            type_data = self._type_data.get(expense_type)
            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_expense_{expense_type}_max_consecutive_days'] = np.nan
                continue
        
            # 按天汇总金额
            daily_data = type_data.groupby(type_data['date'].dt.date)['amount'].sum().index.tolist()
                     
            daily_data = sorted(daily_data) 

            if len(daily_data) == 0:
                features[f'bank_txn_expense_{expense_type}_max_consecutive_days'] = np.nan
                continue
                
        
            max_len = 1
            current_len = 1
            for i in range(1, len(daily_data)):
                if (daily_data[i] - daily_data[i-1]).days == 1:  
                    current_len += 1
                    max_len = max(max_len, current_len)
                else:
                    current_len = 1
            features[f'bank_txn_expense_{expense_type}_max_consecutive_days'] = max_len
    
        self.features.update(features)
        return features

    def trend_slope(self) -> Dict:
        """支出趋势斜率"""
        from sklearn.linear_model import LinearRegression
        
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for expense_type in self.expense_types:
                    features[f'bank_txn_expense_{expense_type}_trend_slope_{window}d'] = np.nan
                continue
            
            for expense_type in self.expense_types:
                type_data = window_data[window_data['tag_level2'] == expense_type]
                
                if len(type_data) >= 2:
                    daily = type_data.groupby('date')['amount'].sum().reset_index()
                    daily = daily.sort_values('date', ascending=True).reset_index(drop=True)
                    
                    if len(daily) >= 2:
                        min_date = daily['date'].min()
                        daily['date_numeric'] = (daily['date'] - min_date).dt.days
                        
                        X = daily['date_numeric'].values.reshape(-1, 1)  
                        y = daily['amount'].values
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        features[f'bank_txn_expense_{expense_type}_trend_slope_{window}d'] = model.coef_[0]
                    else:
                        features[f'bank_txn_expense_{expense_type}_trend_slope_{window}d'] = np.nan
                else:
                    features[f'bank_txn_expense_{expense_type}_trend_slope_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    #############     third_party       #####################
    def third_party_count(self) -> Dict:
        """交易对手特征"""
        features = {}
        
        if self.third_party_col not in self.df.columns:
            return features
        
        for expense_type, tps in self.expense_type_tp_pairs:
            type_data = self._type_data.get(expense_type)
            
            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_expense_{expense_type}_3rdparty_count'] = np.nan
                continue
            
            unique_counts = type_data['third_party'].nunique()
            features[f'bank_txn_expense_{expense_type}_3rdparty_count'] = unique_counts
            
        self.features.update(features)
        return features

    def third_party_consecutive(self) -> Dict:
        '''
        同一支出交易对手最长持续时间
        '''
        features = {}

        for expense_type in self.expense_types:
            type_data = self._type_data.get(expense_type)
            if type_data is not None and len(type_data) > 0:
                tps = type_data['third_party'].dropna().unique().tolist()
                max_lens = []
                
                for tp in tps:
                    tp_data = type_data[type_data['third_party'] == tp]
                    tp_data = tp_data.groupby('trac_days')['amount'].sum().reset_index()
                    tp_data = tp_data.sort_values('trac_days')
                    days = tp_data['trac_days'].values
                    if len(days) > 0:
                        current_len = 1
                        max_len = 1
                        for i in range(1, len(days)):
                            if days[i] - days[i-1] == 1:
                                current_len += 1
                                max_len = max(max_len, current_len)
                                max_lens.append(max_len)
                            else:
                                current_len = 1
                if max_lens:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_max_consecutive'] = np.max(max_lens)
                else:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_max_consecutive'] = 1
            
            else:
                features[f'bank_txn_expense_{expense_type}_3rdparty_max_consecutive'] = np.nan

        self.features.update(features)
        return features            

    def third_party_amount(self) -> Dict:
        """按支出类型-交易对手组合的金额统计特征"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())

            if len(window_data) == 0:
                for expense_type, tps in self.expense_type_tp_pairs:
                    # 对空交易对手列表，不生成对应的特征
                    if not tps:
                        continue
                    for tp in tps:
                        cleaned_tp = self._clean_tp_name(tp)
                        for stat in ['sum', 'count', 'mean', 'median', 'max', 'min', 'std', 'cv']:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_{stat}_{window}d'] = np.nan
                continue
            
            for expense_type, tps in self.expense_type_tp_pairs:
                # 跳过空交易对手列表
                if not tps:
                    continue
                    
                # 筛选该支出类型的窗口数据
                type_data = window_data[window_data['tag_level2'] == expense_type]
                
                for tp in tps:
                    # 筛选该交易对手的数据
                    tp_data = type_data[type_data['third_party'] == tp]
                    cleaned_tp = self._clean_tp_name(tp)
                    if len(tp_data) > 0:
                        amounts = tp_data['amount'].values

                        total_sum = np.sum(amounts)
                        total_count = len(amounts)
                        total_mean = np.mean(amounts)
                        total_median = np.median(amounts)
                        total_max = np.max(amounts)
                        total_min = np.min(amounts)
                        total_std = np.std(amounts) 
                        total_cv = total_std / total_mean if total_mean > 0 else np.nan
                    
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_sum_{window}d'] = total_sum
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_count_{window}d'] = total_count
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_mean_{window}d'] = total_mean
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_median_{window}d'] = total_median
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_{window}d'] = total_max
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_min_{window}d'] = total_min
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_std_{window}d'] = total_std
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_cv_{window}d'] = total_cv
                    else:
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_sum_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_count_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_mean_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_median_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_min_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_std_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_cv_{window}d'] = np.nan
        
        self.features.update(features)
        return features
            
    def third_party_ratio(self) -> Dict:
        """按支出类型-交易对手组合的占比特征"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for expense_type, tps in self.expense_type_tp_pairs:
                    # 跳过空交易对手列表
                    if not tps:
                        continue
                    for tp in tps:
                        cleaned_tp = self._clean_tp_name(tp)
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_total_{window}d'] = np.nan
                continue
            
            total_amount = np.sum(window_data['amount'].values)
            
            if total_amount > 0:
                for expense_type, tps in self.expense_type_tp_pairs:
                    # 跳过空交易对手列表
                    if not tps:
                        continue
                        
                    # 计算该支出类型的总金额
                    type_data = window_data[window_data['tag_level2'] == expense_type]
                    type_sum = np.sum(type_data['amount'].values) if len(type_data) > 0 else np.nan

                    if type_sum > 0:
                        for tp in tps:
                            cleaned_tp = self._clean_tp_name(tp)
                            tp_data = type_data[type_data['third_party'] == tp]
                            tp_sum = np.sum(tp_data['amount'].values) if len(tp_data) > 0 else np.nan
                            # 占该支出类型的比例
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_{window}d'] = tp_sum / type_sum
                            # 占全局的比例
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_total_{window}d'] = tp_sum / total_amount
                    else:
                        for tp in tps:
                            cleaned_tp = self._clean_tp_name(tp)
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_{window}d'] = np.nan
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_total_{window}d'] = np.nan
            else:
                for expense_type, tps in self.expense_type_tp_pairs:
                    # 跳过空交易对手列表
                    if not tps:
                        continue
                    for tp in tps:
                        cleaned_tp = self._clean_tp_name(tp)
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_ratio_total_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def third_party_amount_daily(self)-> Dict:
        """交易对手单日最大最小支出额度"""
        features = {}
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            if len(window_data) == 0:
                for expense_type, tps in self.expense_type_tp_pairs:
                    # 跳过空交易对手列表
                    if not tps:
                        continue
                    for tp in tps:
                        cleaned_tp = self._clean_tp_name(tp)
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_max_{window}d'] = np.nan
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_min_{window}d'] = np.nan
                continue
            for expense_type, tps in self.expense_type_tp_pairs:
                if not tps:
                    continue
                 # 计算该支出类型的总金额
                type_data = window_data[window_data['tag_level2'] == expense_type]
                if type_data is not None and len(type_data) > 0:
                    for tp in tps:
                        
                        tp_data = type_data[type_data['third_party'] == tp] 
                        if len(tp_data) > 0:
                            
                            daily_data = tp_data.groupby('date')['amount'].sum().reset_index()
                            tp_max = np.max(daily_data['amount'].values)
                            tp_min = np.min(daily_data['amount'].values)
                            cleaned_tp = self._clean_tp_name(tp)
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_max_{window}d'] = tp_max
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_min_{window}d'] = tp_min
                        else:
                            cleaned_tp = self._clean_tp_name(tp)
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_max_{window}d'] = np.nan
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_daily_min_{window}d'] = np.nan
        
        self.features.update(features)
        return features                 

    def third_party_amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        """按支出类型-交易对手组合的金额对比特征"""
        features = {}
        window_map = {30: '1m', 90: '3m', 180: '6m'}
        available_windows = [w for w in window_map.keys() if w in windows]

        if not available_windows:
            return features

        for expense_type, tps in self.expense_type_tp_pairs:
            if not tps:
                continue
            type_data = self._type_data.get(expense_type)
            
            for tp in tps:
                if type_data is None or len(type_data) == 0:
                    cleaned_tp = self._clean_tp_name(tp)
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_latest_vs_3m'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_1m_vs_3m'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_3m_vs_6m'] = np.nan
                    continue
                    
                tp_data = type_data[type_data['third_party'] == tp]
                cleaned_tp = self._clean_tp_name(tp)
                
                if tp_data is None or len(tp_data) == 0:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_latest_vs_3m'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_1m_vs_3m'] = np.nan
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_3m_vs_6m'] = np.nan
                    continue
                
                window_avgs = {}
                for window in available_windows:
                    window_data = tp_data[tp_data['trac_days'] <= window]
                    if len(window_data) > 0:
                        window_avgs[window] = np.mean(window_data['amount'].values)
                    else:
                        window_avgs[window] = 0

                latest = tp_data['amount'].iloc[0] if len(tp_data) > 0 else 0
                if 90 in window_avgs and window_avgs[90] > 0:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_latest_vs_3m'] = latest / window_avgs[90]
                else:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_latest_vs_3m'] = np.nan

                if 30 in window_avgs and 90 in window_avgs and window_avgs[90] > 0:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_1m_vs_3m'] = window_avgs[30] / window_avgs[90]
                else:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_1m_vs_3m'] = np.nan

                if 90 in window_avgs and 180 in window_avgs and window_avgs[180] > 0:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_3m_vs_6m'] = window_avgs[90] / window_avgs[180]
                else:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_3m_vs_6m'] = np.nan
        
        self.features.update(features)
        return features

                    
    def third_party_growth_rates(self) -> Dict:
        """按支出类型-交易对手组合的支出增长率特征"""
        features = {}
        
        for expense_type, tps in self.expense_type_tp_pairs:
            # 跳过空交易对手列表
            if not tps:
                continue
                
            type_data = self._type_data.get(expense_type, pd.DataFrame())
            
            for tp in tps:
                tp_data = type_data[type_data['third_party'] == tp] if len(type_data) > 0 else pd.DataFrame()
                cleaned_tp = self._clean_tp_name(tp)

                daily_data = tp_data.groupby('date')['amount'].sum().reset_index()
                daily_data = daily_data.sort_values('date', ascending=True)
                amounts = daily_data['amount'].values if len(daily_data) > 0 else []
                
                if len(amounts) > 1:
                    prev_amounts = amounts[:-1]
                    curr_amounts = amounts[1:]
                    # 计算增长率，避免除以0
                    growth_rates = np.where(prev_amounts != 0, 
                               (curr_amounts - prev_amounts) / prev_amounts, 
                               0)
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_growth_rate_max'] = float(np.max(growth_rates))
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_growth_rate_min'] = float(np.min(growth_rates))
                else:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_growth_rate_max'] = np.nan 
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_growth_rate_min'] = np.nan 

        self.features.update(features)
        return features

    def third_party_max_consecutive_months(self) -> Dict:
        '''
        支出交易对手的最长持续月份
        '''
        features = {}
        for expense_type, tps in self.expense_type_tp_pairs:
            
            if not tps:
                continue
            type_data = self._type_data.get(expense_type, pd.DataFrame())
            for tp in tps:
                tp_data = type_data[type_data['third_party'] == tp] if len(type_data) > 0 else pd.DataFrame()
                if tp_data is None or len(tp_data) == 0:
                    cleaned_tp = self._clean_tp_name(tp)
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_months'] = np.nan
                    continue
                monthly_data = (tp_data.groupby(tp_data['date'].dt.to_period('M'))['amount'].sum()
                                        .reset_index()
                                        .rename(columns={'date': 'year_month'}))
                monthly_data = monthly_data[monthly_data['amount'] > 0]
                
                if len(monthly_data) == 0:
                    cleaned_tp = self._clean_tp_name(tp)
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_months'] = np.nan
                    continue
                    
                monthly_data['year_month'] = monthly_data['year_month'].dt.to_timestamp()
                monthly_data = monthly_data.sort_values('year_month').reset_index(drop=True)

                monthly_data['month_num'] = monthly_data['year_month'].dt.year * 12 + monthly_data['year_month'].dt.month
                
                months = monthly_data['month_num'].values
                max_len = 1
                current_len = 1
                for i in range(1, len(months)):
                    if months[i] - months[i-1] == 1:
                        current_len += 1
                        max_len = max(max_len, current_len)
                    else:
                        current_len = 1
                cleaned_tp = self._clean_tp_name(tp)
                features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_months'] = max_len

        self.features.update(features)
        return features

    def third_party_max_consecutive_days(self) -> Dict:
        '''
        支出交易对手的最长持续天数
        '''
        features = {}
        for expense_type, tps in self.expense_type_tp_pairs:
            # 跳过空交易对手列表
            if not tps:
                continue
            type_data = self._type_data.get(expense_type, pd.DataFrame())
            for tp in tps:
                cleaned_tp = self._clean_tp_name(tp)
                tp_data = type_data[type_data['third_party'] == tp] if len(type_data) > 0 else pd.DataFrame()
                if tp_data is None or len(tp_data) == 0:
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_days']  = np.nan
                    continue
                daily_data = tp_data.groupby(tp_data['date'].dt.date)['amount'].sum().index.tolist()
                daily_data = sorted(daily_data)
                
                if len(daily_data) == 0:
                    cleaned_tp = self._clean_tp_name(tp)
                    features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_days'] = np.nan
                    continue
                
                max_len = 1
                current_len = 1
                for i in range(1, len(daily_data)):
                    if (daily_data[i] - daily_data[i-1]).days == 1:
                        current_len += 1
                        max_len = max(max_len, current_len)
                    else:
                        current_len = 1
                features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_max_consecutive_days']  = max_len

        self.features.update(features)
        return features

    def third_party_trend_slope(self) -> Dict:
        '''
        支出交易对手的趋势
        '''
        from sklearn.linear_model import LinearRegression
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            if len(window_data) == 0:
                for expense_type, tps in self.expense_type_tp_pairs:
                    if not tps:
                        continue
                    for tp in tps:
                        cleaned_tp = self._clean_tp_name(tp)
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_trend_slope_{window}d'] = np.nan
                continue
                
            for expense_type, tps in self.expense_type_tp_pairs:
                # 跳过空交易对手列表
                if not tps:
                    continue
                    
                type_data = window_data[window_data['tag_level2'] == expense_type]
                if len(type_data) == 0:
                    for tp in tps:
                        cleaned_tp = self._clean_tp_name(tp)
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_trend_slope_{window}d'] = np.nan
                    continue
                    
                for tp in tps:
                    tp_data = type_data[type_data['third_party'] == tp] 
                    cleaned_tp = self._clean_tp_name(tp)
                    
                    if len(tp_data) >= 2:
                        daily = tp_data.groupby('date')['amount'].sum().reset_index()
                        daily = daily.sort_values('date', ascending=True).reset_index(drop=True)
                        if len(daily) >= 2:
                            min_date = daily['date'].min()
                            daily['date_numeric'] = (daily['date'] - min_date).dt.days
                            X = daily['date_numeric'].values.reshape(-1, 1)
                            y = daily['amount'].values
                            model = LinearRegression()
                            model.fit(X, y)
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_trend_slope_{window}d'] = model.coef_[0]
                        else:
                            features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_trend_slope_{window}d'] = np.nan
                    else:
                        features[f'bank_txn_expense_{expense_type}_3rdparty_{cleaned_tp}_trend_slope_{window}d'] = np.nan
                                    
        self.features.update(features)
        return features

    #############     数据整合      #####################
    
    def get_metadata(self) -> Dict:
        """获取元数据信息"""
        metadata = {
            'user_id': self.user_id,
            'sample_datetime': self.sample_datetime
        }
        return metadata

    def generate_all_features(self,df:pd.DataFrame = None, feature_groups: List[str] = None, include_metadata: bool = True) -> Dict:
        """
        生成所有特征
        返回特征DataFrame
        """
        if df is not None:
            self.original_df = df.copy()
            self.df = self._map(df)
            self._prepare_data()
            self.raw_df = self.df.copy()
            #self._prepare_data()

            
        if feature_groups is None:
            feature_groups = [
                'amount_global', 'amount_by_type', 
                'ratio', 'ratio_by_type', 
                'amount_daily','amount_comparison','growth_rates', 'max_consecutive_months','max_consecutive_days','trend_slope',
                'third_party_count', 'third_party_consecutive',
                'third_party_amount',
                'third_party_ratio',  
                'third_party_amount_daily','third_party_amount_comparison','third_party_growth_rates', 'third_party_max_consecutive_months','third_party_max_consecutive_days', 'third_party_trend_slope'
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
        
        # 添加额外的元数据信息
        if include_metadata:
            metadata = self.get_metadata()
            for key, value in metadata.items():
                if key not in ['user_id', 'sample_datetime']:
                    self.features[f'metadata_{key}'] = value
        

        df = pd.DataFrame([self.features])
        base_columns = ['user_id', 'sample_datetime']
        existing_base = [col for col in base_columns if col in df.columns]
        feature_cols = [col for col in df.columns if col not in existing_base]
        
        df = df[existing_base + feature_cols]

        df.columns = df.columns.str.strip().str.replace(' ', '').str.lower()
        
        
        return df

expense_type_tp_pairs = [
    ['Dining Out', []],
    ['External Transfers', []],
    ['Internal Transfer', []],
    ['Groceries', []],
    ['Retail', []],
    ['Department Stores', []],
    ['Automotive', []],
    ['Rent', []],
    ['Health', []],
    ['Utilities', []],
    ['Education', []],
    ['Telecommunications', []],
    ['Donations', []],
    ['Centrelink', []],
    ['Fees', []],
    ['Entertainment', []],
    ['Home Improvement', []],
    ['Transport', []],
    ['Subscription TV', []]
]

def generate_expense_feature(df: pd.DataFrame,feature_groups:List[str] = None):
    expense_engineer = SingleApplicationExpenseFeatureEngineer(
        df=df,
        time_windows=[7, 14, 28, 56, 84, 168, 182],
        expense_type_tp_pairs=expense_type_tp_pairs)
    
    return expense_engineer.generate_all_features(df = df,feature_groups = feature_groups)