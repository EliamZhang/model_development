import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import re

class SingleApplicationIncomeFeatureEngineer:
    
    
    def __init__(self, 
                 df: pd.DataFrame,
                 time_windows: List[int] = None,
                 income_type_tp_pairs: List[List] = None,
                 income_type_category_pairs: List[List] = None,
                 tag_level1: str = "INCOME"):
        """
        初始化特征工程类
        """
        
        self.original_df = df.copy()
        

        #交易类别映射表
        self.df = self._map(df)
        self.raw_df = self.df.copy()
        
        
        
        self.time_windows = sorted(time_windows) if time_windows else [7, 14, 28, 56, 84, 168, 182]
        
        # 处理二维数组收入类型和交易对手
        self.income_type_tp_pairs = income_type_tp_pairs
        
        # 处理二维数组收入类型和类别
        self.income_type_category_pairs = income_type_category_pairs
        
        self.income_types = list(set([pair[0] for pair in self.income_type_tp_pairs] + 
                                     [pair[0] for pair in self.income_type_category_pairs]))
        
        self.third_parties = list(set(tp for pair in self.income_type_tp_pairs for tp in pair[1] if tp))
        self.categories = list(set(cat for pair in self.income_type_category_pairs for cat in pair[1] if cat))
        
        self.third_party_col = 'third_party'
        self.category_col = 'category'
        self.tag_level1 = tag_level1
        self.features = {}
        #self.raw_df = df.copy()
        
        self._prepare_data()
        

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

    # 统计匹配情况
        return df
        

    def _clean_entity_name(self, name: str) -> str:
        """
        清洗实体名称，用于特征键名
        
        Args:
            name: 原始实体名称
            
        Returns:
            清洗后的实体名称（小写、无空格、无特殊字符）
        """
        if not isinstance(name, str):
            name = str(name)
        
        # 转换为小写
        cleaned = name.lower()
        
        # 去除空格
        cleaned = cleaned.replace(' ', '')
        
        # 替换 '/' 为 '_'
        cleaned = cleaned.replace('/', '_')
        
        # 替换其他可能引起问题的特殊字符
        cleaned = cleaned.replace('-', '_')
        cleaned = cleaned.replace('.', '_')
        cleaned = cleaned.replace('&', 'and')
        cleaned = cleaned.replace('@', 'at')
        cleaned = cleaned.replace('$', 'dollar')
        cleaned = cleaned.replace('%', 'percent')
        
        # 去除其他标点符号
        cleaned = re.sub(r'[^\w_]', '', cleaned)
        
        return cleaned

    def _prepare_data(self):
        """预处理数据"""
        
        if 'transaction_date' in self.df.columns:
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
        
        # 根据tag_level1筛选收入数据
        if 'tag_level1' in self.df.columns:
            income_mask = self.df['tag_level1'] == self.tag_level1
            self.df = self.df[income_mask].copy()
        else:
            print(f"警告: 数据中不存在'tag_level1'列，跳过筛选")
        
        if 'amount' in self.df.columns:
            self.df['amount'] = self.df['amount'].abs()
        
        # 按trac_days排序（从新到旧）
        if 'trac_days' in self.df.columns:
            self.df = self.df.sort_values(by=["trac_days"], ascending=True).reset_index(drop=True)

        if 'trac_days' in self.df.columns:
            # 过滤掉负数trac_days（未来日期的交易）
            self.df = self.df[self.df['trac_days'] >= 0].copy()
            
        self._type_data = {}
        for income_type in self.income_types:
            if 'tag_level2' in self.df.columns:
                type_data = self.df[self.df['tag_level2'] == income_type].copy()
                if len(type_data) > 0:
                    self._type_data[income_type] = type_data
        
        self._window_data = {}
        for window in self.time_windows:
            if 'trac_days' in self.df.columns:
                self._window_data[window] = self.df[self.df['trac_days'] <= window].copy()
            else:
                self._window_data[window] = pd.DataFrame()

        # 初始化交易对手数据结构
        self._tp_data = {}
        self._type_tp_data = {}
        for income_type, tps in self.income_type_tp_pairs:
            self._type_tp_data[income_type] = {}
            type_data = self._type_data.get(income_type, pd.DataFrame())
            
            if not tps:
                continue
                
            for tp in tps:
                if 'third_party' in type_data.columns:
                    tp_data = type_data[type_data['third_party'] == tp].copy()
                    self._type_tp_data[income_type][tp] = tp_data
                    if tp not in self._tp_data:
                        self._tp_data[tp] = pd.DataFrame()
                    self._tp_data[tp] = pd.concat([self._tp_data[tp], tp_data])

        # 初始化类别数据结构
        self._category_data = {}
        self._type_category_data = {}
        for income_type, categories in self.income_type_category_pairs:
            self._type_category_data[income_type] = {}
            type_data = self._type_data.get(income_type, pd.DataFrame())
            
            if not categories:
                continue
                
            for category in categories:
                if 'category' in type_data.columns:
                    category_data = type_data[type_data['category'] == category].copy()
                    self._type_category_data[income_type][category] = category_data
                    if category not in self._category_data:
                        self._category_data[category] = pd.DataFrame()
                    self._category_data[category] = pd.concat([self._category_data[category], category_data])

    # ==================== 通用实体特征生成函数 ====================

    def _generate_entity_amount_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        """
        生成实体金额统计特征
        
        Args:
            entity_type: 实体类型 ('tp' 或 'category')
            entity_col: 实体列名 ('third_party' 或 'category')
            entity_pairs: 收入类型-实体配对列表
        """
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())

            if len(window_data) == 0:
                for income_type, entities in entity_pairs:
                    if not entities:
                        continue
                    for entity in entities:
                        cleaned_entity = self._clean_entity_name(entity)
                        for stat in ['sum', 'count', 'mean', 'median', 'max', 'min', 'std', 'cv']:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_{stat}_{window}d'] = np.nan
                continue
            
            for income_type, entities in entity_pairs:
                if not entities:
                    continue
                
                type_data = window_data[window_data['tag_level2'] == income_type]
                
                for entity in entities:
                    entity_data = type_data[type_data[entity_col] == entity]
                    cleaned_entity = self._clean_entity_name(entity)
                
                    if len(entity_data) > 0:
                        amounts = entity_data['amount'].values

                        total_sum = np.sum(amounts)
                        total_count = len(amounts)
                        total_mean = np.mean(amounts)
                        total_median = np.median(amounts)
                        total_max = np.max(amounts)
                        total_min = np.min(amounts)
                        total_std = np.std(amounts) if total_count > 1 else np.nan
                        total_cv = total_std / total_mean if total_mean > 0 else np.nan
                    
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_sum_{window}d'] = total_sum
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_count_{window}d'] = total_count
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_mean_{window}d'] = total_mean
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_median_{window}d'] = total_median
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_{window}d'] = total_max
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_min_{window}d'] = total_min
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_std_{window}d'] = total_std
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_cv_{window}d'] = total_cv
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_sum_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_count_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_mean_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_median_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_min_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_std_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_cv_{window}d'] = np.nan
        
        return features

    def _generate_entity_ratio_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        """
        生成实体占比特征
        """
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for income_type, entities in entity_pairs:
                    if not entities:
                        continue
                    for entity in entities:
                        cleaned_entity = self._clean_entity_name(entity)
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d'] = np.nan
                continue
            
            total_amount = np.sum(window_data['amount'].values)
            
            if total_amount > 0:
                for income_type, entities in entity_pairs:
                    if not entities:
                        continue
                    
                    type_data = window_data[window_data['tag_level2'] == income_type]
                    type_sum = np.sum(type_data['amount'].values) if len(type_data) > 0 else np.nan

                    if type_sum > 0:
                        for entity in entities:
                            entity_data = type_data[type_data[entity_col] == entity]
                            entity_sum = np.sum(entity_data['amount'].values) if len(entity_data) > 0 else np.nan
                            cleaned_entity = self._clean_entity_name(entity)
                            
                            # 占该收入类型的比例
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d'] = entity_sum / type_sum
                            # 占全局的比例
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d'] = entity_sum / total_amount
                    else:
                        for entity in entities:
                            cleaned_entity = self._clean_entity_name(entity)
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d'] = np.nan
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d'] = np.nan
            else:
                for income_type, entities in entity_pairs:
                    if not entities:
                        continue
                    for entity in entities:
                        cleaned_entity = self._clean_entity_name(entity)
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_ratio_total_{window}d'] = np.nan
        
        return features

    def _generate_entity_latest_amount_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        """
        生成实体最近收入金额特征
        """
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'
        
        for income_type, entities in entity_pairs:
            if not entities:
                continue
                
            type_data = self._type_data.get(income_type)
            
            if type_data is not None and len(type_data) > 0:
                for entity in entities:
                    entity_data = type_data[type_data[entity_col] == entity]
                    cleaned_entity = self._clean_entity_name(entity)
                    if len(entity_data) > 0:
                        latest = entity_data['amount'].iloc[0]
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_amount'] = latest
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_amount'] = np.nan
            else:
                for entity in entities:
                    cleaned_entity = self._clean_entity_name(entity)
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_amount'] = np.nan
        
        return features

    def _generate_entity_amount_comparison_features(self, entity_type: str, entity_col: str, entity_pairs: List[List], 
                                                   windows: List[int] = [30, 90, 180]) -> Dict:
        """
        生成实体金额对比特征
        """
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'

        window_map = {30: '1m', 90: '3m', 180: '6m'}
        available_windows = [w for w in window_map.keys() if w in windows]
        
        if not available_windows:
            return features
    
        for income_type, entities in entity_pairs:
            if not entities:
                continue
                
            type_data = self._type_data.get(income_type)
            if type_data is not None and len(type_data) > 0:
                for entity in entities:
                    entity_data = type_data[type_data[entity_col] == entity]
                    cleaned_entity = self._clean_entity_name(entity)
                    
                    if entity_data is None or len(entity_data) == 0:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_vs_3m'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_1m_vs_3m'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_3m_vs_6m'] = np.nan
                        continue
                    
                    window_avgs = {}
                    for window in available_windows:
                        window_data = entity_data[entity_data['trac_days'] <= window]
                        if len(window_data) > 0:
                            window_avgs[window] = np.mean(window_data['amount'].values)
                        else:
                            window_avgs[window] = np.nan

                    latest = entity_data['amount'].iloc[0] if len(entity_data) > 0 else np.nan
                    
                    # latest_vs_3m
                    if 90 in window_avgs and not np.isnan(window_avgs[90]) and window_avgs[90] > 0:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_vs_3m'] = latest / window_avgs[90]
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_latest_vs_3m'] = np.nan
                    
                    # 1m_vs_3m
                    if 30 in window_avgs and 90 in window_avgs:
                        if not np.isnan(window_avgs[30]) and not np.isnan(window_avgs[90]) and window_avgs[90] > 0:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_1m_vs_3m'] = window_avgs[30] / window_avgs[90]
                        else:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_1m_vs_3m'] = np.nan
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_1m_vs_3m'] = np.nan
                    
                    # 3m_vs_6m
                    if 90 in window_avgs and 180 in window_avgs:
                        if not np.isnan(window_avgs[90]) and not np.isnan(window_avgs[180]) and window_avgs[180] > 0:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_3m_vs_6m'] = window_avgs[90] / window_avgs[180]
                        else:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_3m_vs_6m'] = np.nan
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_3m_vs_6m'] = np.nan
        
        return features

    def _generate_entity_growth_rates_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        """
        生成实体增长率特征
        """
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'
        
        for income_type, entities in entity_pairs:
            if not entities:
                continue
                
            type_data = self._type_data.get(income_type, pd.DataFrame())

            if len(type_data) == 0:
                for entity in entities:
                    cleaned_entity = self._clean_entity_name(entity)
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max'] = np.nan
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_min'] = np.nan
                continue
            
            for entity in entities:
                entity_data = type_data[type_data[entity_col] == entity] if len(type_data) > 0 else pd.DataFrame()
                cleaned_entity = self._clean_entity_name(entity)
                
                
                if 'date' not in entity_data.columns:
                    if 'transaction_date' in entity_data.columns:
                        entity_data = entity_data.copy()
                        entity_data['date'] = pd.to_datetime(entity_data['transaction_date']).dt.floor('D')
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max'] = np.nan
                        continue
                #print("[DEBUG] type_data cols =", list(entity_data.columns), "rows=", len(entity_data))

                daily_data = entity_data.groupby('date')['amount'].sum().reset_index()
                daily_data = daily_data.sort_values('date', ascending=True)
                amounts = daily_data['amount'].values if len(daily_data) > 0 else []
                
                if len(amounts) > 1:
                    prev_amounts = amounts[:-1]
                    curr_amounts = amounts[1:]
                    growth_rates = np.where(prev_amounts != 0, 
                                          (curr_amounts - prev_amounts) / prev_amounts, 
                                          0)
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max'] = float(np.max(growth_rates))
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_min'] = float(np.min(growth_rates))
                else:
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_max'] = np.nan
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_growth_rate_min'] = np.nan

        return features

    def _generate_entity_max_consecutive_months_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        """
        生成实体最长持续月份特征
        """
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'
        
        for income_type, entities in entity_pairs:
            if not entities:
                continue
                
            type_data = self._type_data.get(income_type, pd.DataFrame())
            
            for entity in entities:
                entity_data = type_data[type_data[entity_col] == entity] if len(type_data) > 0 else pd.DataFrame()
                cleaned_entity = self._clean_entity_name(entity)
                
                if entity_data is None or len(entity_data) == 0:
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_consecutive_months'] = np.nan
                    continue
                    
                monthly_data = (entity_data.groupby(entity_data['date'].dt.to_period('M'))['amount'].sum()
                                        .reset_index()
                                        .rename(columns={'date': 'year_month'}))
                monthly_data = monthly_data[monthly_data['amount'] > 0]
                
                if len(monthly_data) == 0:
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_consecutive_months'] = np.nan
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
                
                features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_max_consecutive_months'] = max_len

        return features

    def _generate_entity_interval_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        """
        生成实体间隔特征
        """
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'
        
        for income_type, entities in entity_pairs:
            if not entities:
                continue
                
            type_data = self._type_data.get(income_type, pd.DataFrame())
            
            for entity in entities:
                entity_data = type_data[type_data[entity_col] == entity] if len(type_data) > 0 else pd.DataFrame()
                cleaned_entity = self._clean_entity_name(entity)
                
                if len(entity_data) > 1:
                    daily_data = entity_data.groupby('date')['amount'].sum().reset_index()
                    daily_data = daily_data.sort_values('date')
                    
                    if len(daily_data) > 1:
                        dates = daily_data['date'].values
                        intervals = np.diff(dates) / np.timedelta64(1, 'D')

                        if len(intervals) > 0:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_std'] = np.std(intervals) if len(intervals) > 1 else 0
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_max'] = np.max(intervals)
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_gap_30_count'] = np.sum(intervals > 30)
                        else:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_std'] = np.nan
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_max'] = np.nan
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_gap_30_count'] = np.nan
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_std'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_max'] = np.nan
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_gap_30_count'] = np.nan
                else:
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_std'] = np.nan
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_max'] = np.nan
                    features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_interval_gap_30_count'] = np.nan
        
        return features

    def _generate_entity_trend_slope_features(self, entity_type: str, entity_col: str, entity_pairs: List[List]) -> Dict:
        """
        生成实体趋势斜率特征
        """
        from sklearn.linear_model import LinearRegression
        features = {}
        entity_key = '3rdparty' if entity_type == 'tp' else 'category'
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            for income_type, entities in entity_pairs:
                if not entities:
                    continue
                    
                for entity in entities:
                    cleaned_entity = self._clean_entity_name(entity)
                    
                    if len(window_data) == 0:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d'] = np.nan
                        continue
                    
                    entity_data = window_data[(window_data['tag_level2'] == income_type) & 
                                             (window_data[entity_col] == entity)]
                    
                    if len(entity_data) >= 2:
                        daily = entity_data.groupby('date')['amount'].sum().reset_index()
                        daily = daily.sort_values('date', ascending=True).reset_index(drop=True)
                        
                        if len(daily) >= 2:
                            min_date = daily['date'].min()
                            daily['date_numeric'] = (daily['date'] - min_date).dt.days
                            X = daily['date_numeric'].values.reshape(-1, 1)
                            y = daily['amount'].values
                            model = LinearRegression()
                            model.fit(X, y)
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d'] = model.coef_[0]
                        else:
                            features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d'] = np.nan
                    else:
                        features[f'bank_txn_income_{income_type}_{entity_key}_{cleaned_entity}_trend_slope_{window}d'] = np.nan
                                    
        return features

    # ==================== 交易对手特征包装函数 ====================

    def third_party_consecutive(self) -> Dict:
        '''
        同一收入交易对手最长持续时间
        '''
        features = {}

        for income_type in self.income_types:
            type_data = self._type_data.get(income_type)
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
                            else:
                                current_len = 1
                        max_lens.append(max_len)
                    
                if max_lens:
                    features[f'bank_txn_income_{income_type}_3rdparty_max_consecutive_days'] = np.max(max_lens)
                else:
                    features[f'bank_txn_income_{income_type}_3rdparty_max_consecutive_days'] = np.nan
            else:
                features[f'bank_txn_income_{income_type}_3rdparty_max_consecutive_days'] = np.nan
                
        
        self.features.update(features)
        return features

    def third_party_consumption_rate(self) -> Dict:
        """
        计算指定交易对手的发薪后指定天数的资金消耗率&最大单日支出
        """
        features = {}
        consumption_windows = self.time_windows
        expense_data = self.raw_df[self.raw_df['tag_level1'] == 'EXPENSE'].copy()

        if 'date' not in expense_data.columns:
            if 'transaction_date' in expense_data.columns:
                expense_data = expense_data.copy()
                expense_data['date'] = pd.to_datetime(expense_data['transaction_date']).dt.floor('D')
            else:
                for tp in tps:
                    cleaned_tp = self._clean_entity_name(tp)
                    features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_consumption_rate_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_min_consumption_rate_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_mean_consumption_rate_{window}d'] = np.nan
                #continue
        #print("[DEBUG] type_data cols =", list(expense_data.columns), "rows=", len(expense_data))

        expense_data['date'] = pd.to_datetime(expense_data['date'])
        expense_data['amount'] = expense_data['amount'].abs()

        expense_daily = expense_data.groupby('date')['amount'].sum().reset_index()
        expense_daily['date'] = pd.to_datetime(expense_daily['date'])

        for income_type, tps in self.income_type_tp_pairs:
            if not tps:
                continue
                
            type_data = self._type_data.get(income_type, pd.DataFrame())
            
            for tp in tps:
                tp_data = type_data[type_data['third_party'] == tp] if len(type_data) > 0 else pd.DataFrame()
                cleaned_tp = self._clean_entity_name(tp)
                
                if tp_data is None or len(tp_data) == 0:
                    features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_daily_consumption_7d'] = np.nan
                    for window in consumption_windows:
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_min_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_mean_consumption_rate_{window}d'] = np.nan
                    continue
                
                daily_data = tp_data.groupby('date')['amount'].sum().reset_index()
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                daily_data = daily_data.sort_values('date', ascending=False).reset_index(drop=True)

                latest = daily_data.iloc[0]
                latest_date = latest['date']
                latest_amount = latest['amount']
                latest_end_date = latest_date + timedelta(days=7)
                window_expense = expense_daily[
                                (expense_daily['date'] >= latest_date) & 
                                (expense_daily['date'] <= latest_end_date)
                            ]['amount']
                max_daily_consumption = window_expense.max() if not window_expense.empty else 0.0
                features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_daily_consumption_7d'] = max_daily_consumption

                for window in consumption_windows:
                    consumption_rates = []
                    
                    for _,row in daily_data.iterrows():
                        income_date = row['date']
                        income_amount = row['amount']
                        if income_amount > 0:
                            end_date = income_date + timedelta(days=window)
                            window_expense = expense_daily[
                                (expense_daily['date'] >= income_date) & 
                                (expense_daily['date'] <= end_date)
                            ]['amount']
                            period_expense = window_expense.sum()
                            
                            consumption_rate = period_expense / income_amount
                            consumption_rates.append(consumption_rate)
                            
                    if len(consumption_rates)>0:
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_consumption_rate_{window}d'] = np.max(consumption_rates)
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_min_consumption_rate_{window}d'] = np.min(consumption_rates)
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_mean_consumption_rate_{window}d'] = np.mean(consumption_rates)
                    else:
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_max_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_min_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_3rdparty_{cleaned_tp}_mean_consumption_rate_{window}d'] = np.nan 
                
        self.features.update(features)
        return features

    def third_party_count(self) -> Dict:
        """交易对手数量特征"""
        features = {}
        
        if self.third_party_col not in self.df.columns:
            return features
        
        for income_type, tps in self.income_type_tp_pairs:
            type_data = self._type_data.get(income_type)
            if type_data is None:
                features[f'bank_txn_income_{income_type}_third_party_count'] = np.nan
                continue
            if len(type_data) > 0:
                unique_counts = type_data['third_party'].nunique()
                features[f'bank_txn_income_{income_type}_third_party_count'] = unique_counts
                
        self.features.update(features)
        return features

    def third_party_amount(self) -> Dict:
        """按收入类型-交易对手组合的金额统计特征"""
        features = self._generate_entity_amount_features('tp', 'third_party', self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_ratio(self) -> Dict:
        """按收入类型-交易对手组合的占比特征"""
        features = self._generate_entity_ratio_features('tp', 'third_party', self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_latest_amount(self) -> Dict:
        """按收入类型-交易对手组合的最近收入金额"""
        features = self._generate_entity_latest_amount_features('tp', 'third_party', self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        """按收入类型-交易对手组合的金额对比特征"""
        features = self._generate_entity_amount_comparison_features('tp', 'third_party', self.income_type_tp_pairs, windows)
        self.features.update(features)
        return features

    def third_party_growth_rates(self) -> Dict:
        """按收入类型-交易对手组合的收入增长率特征"""
        features = self._generate_entity_growth_rates_features('tp', 'third_party', self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_max_consecutive_months(self) -> Dict:
        """收入交易对手的最长持续月份"""
        features = self._generate_entity_max_consecutive_months_features('tp', 'third_party', self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_interval(self) -> Dict:
        """按收入类型-交易对手组合的收入间隔特征"""
        features = self._generate_entity_interval_features('tp', 'third_party', self.income_type_tp_pairs)
        self.features.update(features)
        return features

    def third_party_trend_slope(self) -> Dict:
        """收入交易对手的趋势"""
        features = self._generate_entity_trend_slope_features('tp', 'third_party', self.income_type_tp_pairs)
        self.features.update(features)
        return features

    # ==================== 类别特征包装函数 ====================

    def category_amount(self) -> Dict:
        """按收入类型-类别组合的金额统计特征"""
        features = self._generate_entity_amount_features('category', 'category', self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_ratio(self) -> Dict:
        """按收入类型-类别组合的占比特征"""
        features = self._generate_entity_ratio_features('category', 'category', self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_latest_amount(self) -> Dict:
        """按收入类型-类别组合的最近收入金额"""
        features = self._generate_entity_latest_amount_features('category', 'category', self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        """按收入类型-类别组合的金额对比特征"""
        features = self._generate_entity_amount_comparison_features('category', 'category', self.income_type_category_pairs, windows)
        self.features.update(features)
        return features

    def category_growth_rates(self) -> Dict:
        """按收入类型-类别组合的收入增长率特征"""
        features = self._generate_entity_growth_rates_features('category', 'category', self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_max_consecutive_months(self) -> Dict:
        """收入类别的最长持续月份"""
        features = self._generate_entity_max_consecutive_months_features('category', 'category', self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_interval(self) -> Dict:
        """按收入类型-类别组合的收入间隔特征"""
        features = self._generate_entity_interval_features('category', 'category', self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_trend_slope(self) -> Dict:
        """收入类别的趋势"""
        features = self._generate_entity_trend_slope_features('category', 'category', self.income_type_category_pairs)
        self.features.update(features)
        return features

    def category_consumption_rate(self) -> Dict:
        """
        计算指定收入类别的发薪后指定天数的资金消耗率&最大单日支出
        注意：这是针对收入类型（如Wage、Centrelink）的消费率，不是针对支出类别
        """
        features = {}
        consumption_windows = self.time_windows
        expense_data = self.raw_df[self.raw_df['tag_level1'] == 'EXPENSE'].copy()

        if 'date' not in expense_data.columns:
            if 'transaction_date' in expense_data.columns:
                expense_data = expense_data.copy()
                expense_data['date'] = pd.to_datetime(expense_data['transaction_date']).dt.floor('D')
            else:
                features[f'bank_txn_income_{income_type}_max_consumption_rate_{window}d'] = np.nan
                features[f'bank_txn_income_{income_type}_min_consumption_rate_{window}d'] = np.nan
                features[f'bank_txn_income_{income_type}_mean_consumption_rate_{window}d'] = np.nan
                #continue
        #print("[DEBUG] type_data cols =", list(expense_data.columns), "rows=", len(expense_data))

        expense_data['date'] = pd.to_datetime(expense_data['date'])
        expense_data['amount'] = expense_data['amount'].abs()

        expense_daily = expense_data.groupby('date')['amount'].sum().reset_index()
        expense_daily['date'] = pd.to_datetime(expense_daily['date'])

        for income_type, categories in self.income_type_category_pairs:
            if not categories:
                continue
                
            type_data = self._type_data.get(income_type, pd.DataFrame())
            
            for category in categories:
                category_data = type_data[type_data['category'] == category] if len(type_data) > 0 else pd.DataFrame()
                cleaned_category = self._clean_entity_name(category)
                
                if category_data is None or len(category_data) == 0:
                    features[f'bank_txn_income_{income_type}_category_{cleaned_category}_max_daily_consumption_7d'] = np.nan
                    for window in consumption_windows:
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_max_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_min_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_mean_consumption_rate_{window}d'] = np.nan
                    continue
                
                daily_data = category_data.groupby('date')['amount'].sum().reset_index()
                daily_data['date'] = pd.to_datetime(daily_data['date'])
                daily_data = daily_data.sort_values('date', ascending=False).reset_index(drop=True)

                # 计算7天最大单日消费
                latest = daily_data.iloc[0]
                latest_date = latest['date']
                latest_amount = latest['amount']
                latest_end_date = latest_date + timedelta(days=7)
                window_expense = expense_daily[
                                (expense_daily['date'] >= latest_date) & 
                                (expense_daily['date'] <= latest_end_date)
                            ]['amount']
                max_daily_consumption = window_expense.max() if not window_expense.empty else 0.0
                features[f'bank_txn_income_{income_type}_category_{cleaned_category}_max_daily_consumption_7d'] = max_daily_consumption

                # 计算各窗口的消费率
                for window in consumption_windows:
                    consumption_rates = []
                    
                    for _, row in daily_data.iterrows():
                        income_date = row['date']
                        income_amount = row['amount']
                        if income_amount > 0:
                            end_date = income_date + timedelta(days=window)
                            window_expense = expense_daily[
                                (expense_daily['date'] >= income_date) & 
                                (expense_daily['date'] <= end_date)
                            ]['amount']
                            period_expense = window_expense.sum()
                            
                            consumption_rate = period_expense / income_amount
                            consumption_rates.append(consumption_rate)
                            
                    if len(consumption_rates) > 0:
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_max_consumption_rate_{window}d'] = np.max(consumption_rates)
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_min_consumption_rate_{window}d'] = np.min(consumption_rates)
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_mean_consumption_rate_{window}d'] = np.mean(consumption_rates)
                    else:
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_max_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_min_consumption_rate_{window}d'] = np.nan
                        features[f'bank_txn_income_{income_type}_category_{cleaned_category}_mean_consumption_rate_{window}d'] = np.nan 
                
        self.features.update(features)
        return features

    
    # ==================== 原有特征函数 ====================

    def amount_global(self) -> Dict:
        """全局收入统计特征"""
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
                total_std = np.std(amounts) if total_count > 1 else 0
                total_cv = total_std / total_mean if total_mean > 0 else 0
                
                features[f'bank_txn_income_global_sum_{window}d'] = total_sum
                features[f'bank_txn_income_global_count_{window}d'] = total_count
                features[f'bank_txn_income_global_mean_{window}d'] = total_mean
                features[f'bank_txn_income_global_median_{window}d'] = total_median
                features[f'bank_txn_income_global_max_{window}d'] = total_max
                features[f'bank_txn_income_global_min_{window}d'] = total_min
                features[f'bank_txn_income_global_std_{window}d'] = total_std
                features[f'bank_txn_income_global_cv_{window}d'] = total_cv
            else:
                features[f'bank_txn_income_global_sum_{window}d'] = np.nan
                features[f'bank_txn_income_global_count_{window}d'] = np.nan
                features[f'bank_txn_income_global_mean_{window}d'] = np.nan
                features[f'bank_txn_income_global_median_{window}d'] = np.nan
                features[f'bank_txn_income_global_max_{window}d'] = np.nan
                features[f'bank_txn_income_global_min_{window}d'] = np.nan
                features[f'bank_txn_income_global_std_{window}d'] = np.nan
                features[f'bank_txn_income_global_cv_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def amount_by_type(self) -> Dict:
        """按收入类型的统计特征"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for income_type in self.income_types:
                    for stat in ['sum', 'count', 'mean', 'median', 'max', 'min', 'std', 'cv']:
                        features[f'bank_txn_income_{income_type}_{stat}_{window}d'] = np.nan
                continue
            
            for income_type in self.income_types:
                type_data = window_data[window_data['tag_level2'] == income_type]
                
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
                    
                    features[f'bank_txn_income_{income_type}_sum_{window}d'] = total_sum
                    features[f'bank_txn_income_{income_type}_count_{window}d'] = total_count
                    features[f'bank_txn_income_{income_type}_mean_{window}d'] = total_mean
                    features[f'bank_txn_income_{income_type}_median_{window}d'] = total_median
                    features[f'bank_txn_income_{income_type}_max_{window}d'] = total_max
                    features[f'bank_txn_income_{income_type}_min_{window}d'] = total_min
                    features[f'bank_txn_income_{income_type}_std_{window}d'] = total_std
                    features[f'bank_txn_income_{income_type}_cv_{window}d'] = total_cv
                else:
                    features[f'bank_txn_income_{income_type}_sum_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_count_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_mean_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_median_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_max_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_min_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_std_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_cv_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def latest_amount(self) -> Dict:
        """最近一次收入金额"""
        features = {}
        
        for income_type in self.income_types:
            type_data = self._type_data.get(income_type)
            
            if type_data is not None and len(type_data) > 0:
                latest = type_data['amount'].iloc[0]
                features[f'bank_txn_income_{income_type}_latest_amount'] = latest
            else:
                features[f'bank_txn_income_{income_type}_latest_amount'] = np.nan
        
        self.features.update(features)
        return features

    def amount_comparison(self, windows: List[int] = [30, 90, 180]) -> Dict:
        """收入金额对比特征"""
        features = {}
        
        window_map = {30: '1m', 90: '3m', 180: '6m'}
        available_windows = [w for w in window_map.keys() if w in windows]
        
        if not available_windows:
            return features
        
        for income_type in self.income_types:
            type_data = self._type_data.get(income_type)
            
            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_income_{income_type}_latest_vs_3m'] = np.nan
                features[f'bank_txn_income_{income_type}_1m_vs_3m'] = np.nan
                features[f'bank_txn_income_{income_type}_3m_vs_6m'] = np.nan
                continue
            
            window_avgs = {}
            for window in available_windows:
                window_data = type_data[type_data['trac_days'] <= window]
                if len(window_data) > 0:
                    window_avgs[window] = np.mean(window_data['amount'].values)
                else:
                    window_avgs[window] = np.nan
            
            latest = type_data['amount'].iloc[0] if len(type_data) > 0 else np.nan
            
            if 90 in window_avgs and not np.isnan(window_avgs[90]) and window_avgs[90] > 0:
                features[f'bank_txn_income_{income_type}_latest_vs_3m'] = latest / window_avgs[90]
            else:
                features[f'bank_txn_income_{income_type}_latest_vs_3m'] = np.nan
            
            if 30 in window_avgs and 90 in window_avgs:
                if not np.isnan(window_avgs[30]) and not np.isnan(window_avgs[90]) and window_avgs[90] > 0:
                    features[f'bank_txn_income_{income_type}_1m_vs_3m'] = window_avgs[30] / window_avgs[90]
                else:
                    features[f'bank_txn_income_{income_type}_1m_vs_3m'] = np.nan
            else:
                features[f'bank_txn_income_{income_type}_1m_vs_3m'] = np.nan
            
            if 90 in window_avgs and 180 in window_avgs:
                if not np.isnan(window_avgs[90]) and not np.isnan(window_avgs[180]) and window_avgs[180] > 0:
                    features[f'bank_txn_income_{income_type}_3m_vs_6m'] = window_avgs[90] / window_avgs[180]
                else:
                    features[f'bank_txn_income_{income_type}_3m_vs_6m'] = np.nan
            else:
                features[f'bank_txn_income_{income_type}_3m_vs_6m'] = np.nan
        
        self.features.update(features)
        return features

    def growth_rates(self) -> Dict:
        '''
        最大收入涨跌幅
        '''
        features = {}
        for income_type in self.income_types:
            type_data = self._type_data.get(income_type, pd.DataFrame())

            if len(type_data) == 0:
                features[f'bank_txn_income_{income_type}_growth_rate_max'] = np.nan
                features[f'bank_txn_income_{income_type}_growth_rate_min'] = np.nan
                continue
            
            daily_data = type_data.groupby('date')['amount'].sum().reset_index()
            daily_data = daily_data.sort_values('date', ascending=True) 
            
            amounts = daily_data['amount'].values if len(daily_data) > 0 else []
            if len(amounts) > 1:
                prev_amounts = amounts[:-1]
                curr_amounts = amounts[1:]
                growth_rates = np.where(prev_amounts != 0, 
                               (curr_amounts - prev_amounts) / prev_amounts, 
                               0)
                features[f'bank_txn_income_{income_type}_growth_rate_max'] = float(np.max(growth_rates))
                features[f'bank_txn_income_{income_type}_growth_rate_min'] = float(np.min(growth_rates))
            else:
                features[f'bank_txn_income_{income_type}_growth_rate_max'] = np.nan 
                features[f'bank_txn_income_{income_type}_growth_rate_min'] = np.nan 
        self.features.update(features)
        return features

    def ratio(self) -> Dict:
        """收入占比特征"""
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for income_type in self.income_types:
                    features[f'bank_txn_income_{income_type}_ratio_{window}d'] = np.nan
                continue
            
            total_amount = np.sum(window_data['amount'].values)
            
            if total_amount > 0:
                for income_type in self.income_types:
                    type_data = window_data[window_data['tag_level2'] == income_type]
                    type_sum = np.sum(type_data['amount'].values) if len(type_data) > 0 else np.nan
                    features[f'bank_txn_income_{income_type}_ratio_{window}d'] = type_sum / total_amount
            else:
                for income_type in self.income_types:
                    features[f'bank_txn_income_{income_type}_ratio_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def ratio_by_type(self) -> Dict:
        '''
        最大单笔收入占收入类型总收入比
        '''
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for income_type in self.income_types:
                    features[f'bank_txn_income_{income_type}_max_ratio_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_min_ratio_{window}d'] = np.nan
                continue
            
            for income_type in self.income_types:
                type_data = window_data[window_data['tag_level2'] == income_type]
                type_sum = np.sum(type_data['amount'].values) if len(type_data) > 0 else np.nan
                if type_sum > 0:
                    min_value = np.min(type_data['amount'].values) if len(type_data) > 0 else np.nan
                    max_value = np.max(type_data['amount'].values) if len(type_data) > 0 else np.nan
                    features[f'bank_txn_income_{income_type}_max_ratio_{window}d'] = max_value / type_sum 
                    features[f'bank_txn_income_{income_type}_min_ratio_{window}d'] = min_value / type_sum 
                else:
                    features[f'bank_txn_income_{income_type}_max_ratio_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_min_ratio_{window}d'] = np.nan
            
        self.features.update(features)
        return features

    def max_consecutive_months(self) -> Dict:
        '''
        连续获收入月份
        '''
        features = {}
        for income_type in self.income_types:
            type_data = self._type_data.get(income_type)
            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_income_{income_type}_max_consecutive_months'] = np.nan
                continue
            
            monthly_data = (type_data.groupby(type_data['date'].dt.to_period('M'))['amount'].sum()  
            .reset_index() 
            .rename(columns={'date': 'year_month'}))
            
            monthly_data = monthly_data[monthly_data['amount'] > 0]

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
            features[f'bank_txn_income_{income_type}_max_consecutive_months'] = max_len
        
        self.features.update(features)
        return features

    def interval(self) -> Dict:
        """收入间隔特征"""
        features = {}
        
        for income_type in self.income_types:
            type_data = self._type_data.get(income_type)
            
            if type_data is None or len(type_data) < 2:
                features[f'bank_txn_income_{income_type}_interval_std'] = np.nan
                features[f'bank_txn_income_{income_type}_interval_max'] = np.nan
                features[f'bank_txn_income_{income_type}_interval_gap_30_count'] = np.nan
                continue
            
            daily_data = type_data.groupby('date')['amount'].sum().reset_index()
            daily_data = daily_data.sort_values('date')
            
            dates = daily_data['date'].values
            intervals = np.diff(dates) / np.timedelta64(1, 'D')
            
            if len(intervals) > 0:
                features[f'bank_txn_income_{income_type}_interval_std'] = np.std(intervals) if len(intervals) > 1 else 0
                features[f'bank_txn_income_{income_type}_interval_max'] = np.max(intervals)
                features[f'bank_txn_income_{income_type}_interval_gap_30_count'] = np.sum(intervals > 30)
            else:
                features[f'bank_txn_income_{income_type}_interval_std'] = np.nan
                features[f'bank_txn_income_{income_type}_interval_max'] = np.nan
                features[f'bank_txn_income_{income_type}_interval_gap_30_count'] = np.nan
        
        self.features.update(features)
        return features

    def fluctuation(self) -> Dict:
        """收入波动特征"""
        features = {}
        
        for income_type in self.income_types:
            type_data = self._type_data.get(income_type)
            
            if type_data is None or len(type_data) < 2:
                features[f'bank_txn_income_{income_type}_decrease_count'] = np.nan
                continue
            
            daily_data = type_data.groupby('date')['amount'].sum().reset_index()
            daily_data = daily_data.sort_values('date')
            
            amounts = daily_data['amount'].values
            decreases = np.sum(amounts[1:] < amounts[:-1])
            
            features[f'bank_txn_income_{income_type}_decrease_count'] = decreases
        
        self.features.update(features)
        return features

    def trend_slope(self) -> Dict:
        """收入趋势斜率"""
        from sklearn.linear_model import LinearRegression
        
        features = {}
        
        for window in self.time_windows:
            window_data = self._window_data.get(window, pd.DataFrame())
            
            if len(window_data) == 0:
                for income_type in self.income_types:
                    features[f'bank_txn_income_{income_type}_trend_slope_{window}d'] = np.nan
                continue
            
            for income_type in self.income_types:
                type_data = window_data[window_data['tag_level2'] == income_type]
                
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
                        features[f'bank_txn_income_{income_type}_trend_slope_{window}d'] = model.coef_[0]
                    else:
                        features[f'bank_txn_income_{income_type}_trend_slope_{window}d'] = np.nan
                else:
                    features[f'bank_txn_income_{income_type}_trend_slope_{window}d'] = np.nan
        
        self.features.update(features)
        return features

    def consumption_rate(self) -> Dict:
        """
        计算发薪后指定天数的资金消耗率&最大单日支出
        """
        features = {}
        consumption_windows = self.time_windows
        expense_data = self.raw_df[self.raw_df['tag_level1'] == 'EXPENSE'].copy()

        if 'date' not in expense_data.columns:
            if 'transaction_date' in expense_data.columns:
                expense_data = expense_data.copy()
                expense_data['date'] = pd.to_datetime(expense_data['transaction_date']).dt.floor('D')
            else:
                features[f'bank_txn_income_{income_type}_max_consumption_rate_{window}d'] = np.nan
                features[f'bank_txn_income_{income_type}_min_consumption_rate_{window}d'] = np.nan
                features[f'bank_txn_income_{income_type}_mean_consumption_rate_{window}d'] = np.nan
                #continue
        #print("[DEBUG] type_data cols =", list(expense_data.columns), "rows=", len(expense_data))
        expense_data['date'] = pd.to_datetime(expense_data['date'])
        expense_data['amount'] = expense_data['amount'].abs()

        expense_daily = expense_data.groupby('date')['amount'].sum().reset_index()
        
        
        for income_type in self.income_types:
            type_data = self._type_data.get(income_type)

            if type_data is None or len(type_data) == 0:
                features[f'bank_txn_income_{income_type}_max_daily_consumption_7d'] = np.nan
                for window in consumption_windows:
                    features[f'bank_txn_income_{income_type}_max_consumption_rate_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_min_consumption_rate_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_mean_consumption_rate_{window}d'] = np.nan
                continue
            
            daily_data = type_data.groupby('date')['amount'].sum().reset_index()
            daily_data['date'] = pd.to_datetime(daily_data['date'])
            daily_data = daily_data.sort_values('date', ascending=False).reset_index(drop=True)

            latest = daily_data.iloc[0]
            latest_date = latest['date']
            latest_amount = latest['amount']
            end_date = latest_date + timedelta(days=7)
            window_expense = expense_daily[
                                (expense_daily['date'] >= latest_date) & 
                                (expense_daily['date'] <= end_date)
                            ]['amount']
            max_daily_consumption = window_expense.max() if not window_expense.empty else 0.0
            features[f'bank_txn_income_{income_type}_max_daily_consumption_7d'] = max_daily_consumption
            
            for window in consumption_windows:
                consumption_rates = []
                max_daily_consumptions = []
                
                for _,row in daily_data.iterrows():
                    income_date = row['date']
                    income_amount = row['amount']
                    if income_amount > 0:
                        end_date = income_date + timedelta(days=window)
                        window_expense = expense_daily[
                                (expense_daily['date'] >= income_date) & 
                                (expense_daily['date'] <= end_date)
                            ]['amount']
                        
                        period_expense = window_expense.sum()
                        max_daily_consumption = window_expense.max() if not window_expense.empty else 0.0
                        
                        consumption_rate = period_expense / income_amount
                        consumption_rates.append(consumption_rate)
                        max_daily_consumptions.append(max_daily_consumption)

                        
                if consumption_rates:
                    features[f'bank_txn_income_{income_type}_max_consumption_rate_{window}d'] = np.max(consumption_rates)
                    features[f'bank_txn_income_{income_type}_min_consumption_rate_{window}d'] = np.min(consumption_rates)
                    features[f'bank_txn_income_{income_type}_mean_consumption_rate_{window}d'] = np.mean(consumption_rates)
                else:
                    features[f'bank_txn_income_{income_type}_max_consumption_rate_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_min_consumption_rate_{window}d'] = np.nan
                    features[f'bank_txn_income_{income_type}_mean_consumption_rate_{window}d'] = np.nan
                
        self.features.update(features)
        return features

    def get_metadata(self) -> Dict:
        """获取元数据信息"""
        metadata = {
            'user_id': self.user_id,
            'sample_datetime': self.sample_datetime
        }
        return metadata

    def generate_all_features(self, df:pd.DataFrame = None, feature_groups: List[str] = None, include_metadata: bool = True) -> Dict:
        """
        生成所有特征,返回特征DataFrame
        """
        if df is not None:
            self.original_df = df.copy()
            self.df = self._map(df)
            self.raw_df = self.df.copy()
            self._prepare_data()
            
        if feature_groups is None:
            feature_groups = [
                'amount_global', 'amount_by_type', 'latest_amount', 'amount_comparison',
                'ratio', 'ratio_by_type', 
                'interval', 'growth_rates', 'fluctuation', 'max_consecutive_months', 
                'consumption_rate', 'trend_slope',
                'third_party_count', 'third_party_consecutive',
                'third_party_amount', 'third_party_ratio', 'third_party_latest_amount', 
                'third_party_amount_comparison', 'third_party_interval', 
                'third_party_growth_rates', 'third_party_max_consecutive_months',
                'third_party_consumption_rate', 'third_party_trend_slope',
                #
                'category_amount', 'category_ratio', 'category_latest_amount', 
                'category_amount_comparison', 'category_interval', 
                'category_growth_rates', 'category_max_consecutive_months',
                'category_trend_slope','category_consumption_rate'
            ]
        
        self.features = {}
        
        for feature in feature_groups:
            method = getattr(self, feature)
            method()
        
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

# input_data = pd.read_excel("./spark3-14146_rawdf.xlsx")
income_type_tp_pairs = [
        ['Wage', []],
        ['Centrelink', ['Centrelink Pension','Family Benefits','JobSeeker','Carers Benefits',
                       'Child Support','Youth Allowance','Parenting Payment','National Disability Insurance',
                       'Vet Affairs','Parental Leave Pay','Child Care Subsidy',
                       'Education Entry Payment','National Disability Insurance Scheme',
                       'Child Disability Assistance Payment','Child Disability Assistance Pa',
                       'Other Centrelink/Government Payments','Other Centrelink/Government Pa',
                       'COLC Concessions','Mobility Allowance','Stillborn Pay','Schoolkids Bonus',
                       'Disability Pension','Emergency Payment']],
        ['Other Income', []]
    ]

income_type_category_pairs = [
        ['Wage', []],
        ['Centrelink', []],
        ['Other Income', ['All Other Credits','Automotive','Department Stores','Dining Out',
                         'Donations','Education','Entertainment','Entertainment','Fees','Gambling',
                         'Groceries','Gyms and other memberships','Health','Home Improvement',
                         'Insurance','Internal Transfer','Overdrawn','Personal Care','Pet Care',
                         'Rent','Retail','Subscription TV','Telecommunications','Transport',
                         'Travel','Utilities']]
    ]


def generate_income_feature(df: pd.DataFrame,feature_groups: List[str] = None):
    # 初始化特征工程类
    income_engineer = SingleApplicationIncomeFeatureEngineer(
        df=df,
        time_windows=[7, 14, 28, 56, 84, 168, 182],
        income_type_tp_pairs=income_type_tp_pairs,
        income_type_category_pairs = income_type_category_pairs
        )
    return income_engineer.generate_all_features(df = df,feature_groups = feature_groups)