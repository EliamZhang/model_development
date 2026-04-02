import pandas as pd
import numpy as np		

class TXNSampleVar:
    """给定样本的txn lender相关变量
    """
    def __init__(self, user_id, send_time, raw_data):
        self.user_id = user_id
        self.send_time = pd.to_datetime(send_time)
        self.raw_data = raw_data
        # self.app_list_df = None
        self.variables = {"user_id": user_id, "sample_datetime": self.send_time}
        self.prod_dict = {
            'personal loan':'loan',
            'BNPL': 'bnpl',
            'cash/wage advance':'advance',
            'bank': 'bank'
        }
        self.app_cate_list_df = None
        self.app_cate_g_list_df = None
    
    def clean_txn_step1(self):
        """清洗txn lender数据第1步.
        选择和LOAN相关的数据
        """
        df = self.raw_data.copy()
        df = df.loc[df.category.str.lower().str.contains('loan'), ] # 筛选loan
        df['amount'] = df.amount.apply(lambda x: abs(x))  # 金额取绝对值
        # df['trac_diffdays'] = [(self.send_time - x).days for x in df["transaction_date"]]
        df['trac_diffdays'] = (self.send_time  - df['transaction_date']).dt.days
        # df = df.groupby(["user_id", "appcompany"]).agg({"inserttime": min, "updatetime": max, "app_install_time": max, "last_update_time":max}).reset_index()
        self.raw_data = df
    
    def create_lender_cate_list(self, lender_cate_mapping_df):
        """生成交易对手类别List数据.
        匹配类别标签，去重无类别APP

        Parameters
        ----------
        lender_cate_mapping_df : pandas.DataFrame
            交易对手类别数据
        """
        df = self.raw_data.copy()
        df = pd.merge(df, lender_cate_mapping_df, on=["third_party"], how="left")
        # df = df.loc[[not pd.isnull(x) for x in df["app_category"]]]

        self.raw_cate_list_df = df

    def grouped_app_cate_list(self):
        """聚合lender类别List数据.
        """
        df = self.raw_cate_list_df.copy()
        df = df.groupby(["user_id", "competitor", "product_type", "transaction_date", "dr_cr", "category"]).agg(
                amount = pd.NamedAgg(column="amount", aggfunc="sum"),
                transaction_cnt = pd.NamedAgg(column="transaction_id", aggfunc="count"),
                ).reset_index()
        self.raw_cate_g_list_df = df
    
    
    def calc_vars_repay_stat(self, cate_list, day_list=[None, ], calc_rto=True, calc_cnt=False, calc_max=False, calc_min=False):
        """计算近多段时间内指定类别还款次数和金额.

        Parameters
        ----------
        cate_list : List
            待计算APP类别列表
        day_list : list
            待计算时段列表
        calc_rto : bool
            是否计算占比，默认值为True
        """
        def __calc(df, df_cate, cate_list, day, calc_rto, calc_cnt, calc_max, calc_min):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_sum', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_sum']) for x in cate_list}
            variables = {varnames[x]: -1 for x in cate_list}        
            if calc_rto:
                varnames_rto = {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_avg', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_avg']) for x in cate_list}
                variables.update({varnames_rto[x]: -1 for x in cate_list})
            if calc_cnt:
                varnames_cnt = {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'count', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'count']) for x in cate_list}
                variables.update({varnames_cnt[x]: -1 for x in cate_list})
            if calc_max:
                varnames_max = {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_max', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_max']) for x in cate_list}
                variables.update({varnames_max[x]: -1 for x in cate_list})
            if calc_min:
                varnames_min = {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_min', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'amount_min']) for x in cate_list}
                variables.update({varnames_min[x]: -1 for x in cate_list})

            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day, ].copy()
                df_cate = df_cate.loc[df_cate["trac_diffdays"] <= day, ].copy()
    
            if df.shape[0] > 0:
                # variables.update({varnames[x]: 0 for x in cate_list})
                # if calc_rto:
                #     variables.update({varnames_rto[x]: 0 for x in cate_list})
                # if calc_cnt:
                #     variables.update({varnames_cnt[x]: 0 for x in cate_list})
                # if calc_max:
                #     variables.update({varnames_max[x]: 0 for x in cate_list})
                # if calc_min:
                #     variables.update({varnames_min[x]: 0 for x in cate_list})
                df_cate = df_cate.loc[df_cate["product_type"].isin(cate_list), ].copy()
                if df_cate.shape[0] > 0:
                    df_cate["repay_lender"] = [varnames[x] for x in df_cate["product_type"]]
                    variables.update(df_cate.groupby("repay_lender").agg({"amount": sum}).to_dict(orient="dict")["amount"])
                    if calc_rto:
                        df_cate["repay_lender_rto"] = [varnames_rto[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("repay_lender_rto").agg({"amount": np.mean}).to_dict(orient="dict")["amount"])
                    if calc_cnt:
                        df_cate["repay_lender_cnt"] = [varnames_cnt[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("repay_lender_cnt").agg({"transaction_id": 'count'}).to_dict(orient="dict")["transaction_id"])
                    if calc_max:
                        df_cate["repay_lender_max"] = [varnames_max[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("repay_lender_max").agg({"amount": np.max}).to_dict(orient="dict")["amount"])
                    if calc_min:
                        df_cate["repay_lender_min"] = [varnames_min[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("repay_lender_min").agg({"amount": np.min}).to_dict(orient="dict")["amount"])

            return variables

        df = self.raw_data.loc[self.raw_data['dr_cr']=='debit', ].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr']=='debit', ].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df, df_cate, cate_list, day, calc_rto, calc_cnt, calc_max, calc_min))
        self.variables.update(variables)
    
    def calc_vars_disburse_stat(self, cate_list, day_list=[None, ], calc_rto=True, calc_cnt=False, calc_max=False, calc_min=False):
        """计算近多段时间内指定类别放款次数和金额.

        Parameters
        ----------
        cate_list : List
            待计算APP类别列表
        day_list : list
            待计算时段列表
        calc_rto : bool
            是否计算占比，默认值为True
        """
        def __calc(df, df_cate, cate_list, day, calc_rto, calc_cnt, calc_max, calc_min):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_sum', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_sum']) for x in cate_list}
            variables = {varnames[x]: -1 for x in cate_list}        
            if calc_rto:
                varnames_rto = {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_avg', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_avg']) for x in cate_list}
                variables.update({varnames_rto[x]: -1 for x in cate_list})
            if calc_cnt:
                varnames_cnt = {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'count', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'count']) for x in cate_list}
                variables.update({varnames_cnt[x]: -1 for x in cate_list})
            if calc_max:
                varnames_max = {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_max', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_max']) for x in cate_list}
                variables.update({varnames_max[x]: -1 for x in cate_list})
            if calc_min:
                varnames_min = {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_min', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'amount_min']) for x in cate_list}
                variables.update({varnames_min[x]: -1 for x in cate_list})

            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day, ].copy()
                df_cate = df_cate.loc[df_cate["trac_diffdays"] <= day, ].copy()
    
            if df.shape[0] > 0:
                # variables.update({varnames[x]: 0 for x in cate_list})
                # if calc_rto:
                #     variables.update({varnames_rto[x]: 0 for x in cate_list})
                # if calc_cnt:
                #     variables.update({varnames_cnt[x]: 0 for x in cate_list})
                # if calc_max:
                #     variables.update({varnames_max[x]: 0 for x in cate_list})
                # if calc_min:
                #     variables.update({varnames_min[x]: 0 for x in cate_list})
                df_cate = df_cate.loc[df_cate["product_type"].isin(cate_list), ].copy()
                if df_cate.shape[0] > 0:
                    df_cate["disburse_lender"] = [varnames[x] for x in df_cate["product_type"]]
                    variables.update(df_cate.groupby("disburse_lender").agg({"amount": sum}).to_dict(orient="dict")["amount"])
                    if calc_rto:
                        df_cate["disburse_lender_rto"] = [varnames_rto[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("disburse_lender_rto").agg({"amount": np.mean}).to_dict(orient="dict")["amount"])
                    if calc_cnt:
                        df_cate["disburse_lender_cnt"] = [varnames_cnt[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("disburse_lender_cnt").agg({"transaction_id": 'count'}).to_dict(orient="dict")["transaction_id"])
                    if calc_max:
                        df_cate["disburse_lender_max"] = [varnames_max[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("disburse_lender_max").agg({"amount": np.max}).to_dict(orient="dict")["amount"])
                    if calc_min:
                        df_cate["disburse_lender_min"] = [varnames_min[x] for x in df_cate["product_type"]]
                        variables.update(df_cate.groupby("disburse_lender_min").agg({"amount": np.min}).to_dict(orient="dict")["amount"])

            return variables

        df = self.raw_data.loc[self.raw_data['dr_cr']=='credit', ].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr']=='credit', ].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df, df_cate, cate_list, day, calc_rto, calc_cnt, calc_max, calc_min))
        self.variables.update(variables)
    
    def calc_vars_summary_repay_stat(self, day_list=[None, 1, ]):
        """计算近多段时间内在装APP总数量.

        Parameters
        ----------
        day_list : list
            待计算时段列表，默认值为[None, 3, ]
        """
        def __calc(df, day):
            """计算近day日内在装APP总数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            day : int
                待计算时段，默认值为None，即不考虑时间

            Returns
            -------
            variables : dict
                变量结果字典
            """
            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day].copy()
            varname = "_".join(["bank_txn_lender_repay_amount_sum", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_repay_amount_sum"
            varnames_rto = "_".join(["bank_txn_lender_repay_amount_avg", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_repay_amount_avg"
            varname_cnt = "_".join(["bank_txn_lender_repay_count", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_repay_count"
            varname_min = "_".join(["bank_txn_lender_repay_amount_min", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_repay_amount_min"
            varname_max = "_".join(["bank_txn_lender_repay_amount_max", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_repay_amount_max"
            
            variables = {varname: -1, 
                         varnames_rto: -1,
                         varname_cnt: -1,
                         varname_min: -1,
                         varname_max: -1
                         }
            if df.shape[0] > 0:
                variables = {varname: df.amount.sum(),
                             varnames_rto: df.amount.mean(),
                             varname_cnt: df.shape[0],
                             varname_min: df.amount.min(),
                             varname_max: df.amount.max()
                             }

            return variables
        
        variables = {}
        df = self.raw_data.loc[self.raw_data['dr_cr']=='debit', ].copy()
        for day in day_list:
            variables.update(__calc(df=df, day=day))
        self.variables.update(variables)

    def calc_vars_summary_disburse_stat(self, day_list=[None, 1, ]):
        """计算近多段时间内在装APP总数量.

        Parameters
        ----------
        day_list : list
            待计算时段列表，默认值为[None, 3, ]
        """
        def __calc(df, day):
            """计算近day日内在装APP总数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            day : int
                待计算时段，默认值为None，即不考虑时间

            Returns
            -------
            variables : dict
                变量结果字典
            """
            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day].copy()
            varname = "_".join(["bank_txn_lender_disburse_amount_sum", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_disburse_amount_sum"
            varnames_rto = "_".join(["bank_txn_lender_disburse_amount_avg", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_disburse_amount_avg"
            varname_cnt = "_".join(["bank_txn_lender_disburse_count", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_disburse_count"
            varname_min = "_".join(["bank_txn_lender_disburse_amount_min", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_disburse_amount_min"
            varname_max = "_".join(["bank_txn_lender_disburse_amount_max", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_disburse_amount_max"
            
            variables = {varname: -1, 
                         varnames_rto: -1,
                         varname_cnt: -1,
                         varname_min: -1,
                         varname_max: -1
                         }
            if df.shape[0] > 0:
                variables = {varname: df.amount.sum(),
                             varnames_rto: df.amount.mean(),
                             varname_cnt: df.shape[0],
                             varname_min: df.amount.min(),
                             varname_max: df.amount.max()
                             }

            return variables
        
        variables = {}
        df = self.raw_data.loc[self.raw_data['dr_cr']=='credit', ].copy()
        for day in day_list:
            variables.update(__calc(df=df, day=day))
        self.variables.update(variables)
    
    def calc_vars_repay_single_stat(self, cate_list=['personal loan', 'BNPL', 'cash/wage advance', 'bank']):
        """
        计算历史单机构最大累计还款金额/次数
        """
        def __calc(df, df_cate, cate_list):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = {x: "_".join(["bank_txn_lender_repay_single", self.prod_dict[x], 'count_max']) for x in cate_list}
            variables = {varnames[x]: -1 for x in cate_list}
            varnames_amt = {x: "_".join(["bank_txn_lender_repay_single", self.prod_dict[x], 'amount_max']) for x in cate_list}
            variables.update({varnames_amt[x]: -1 for x in cate_list})
            # 汇总
            variables.update({"bank_txn_lender_repay_single_count_max": -1,
                              "bank_txn_lender_repay_single_amount_max": -1
                              }
                            )
    
            if df.shape[0] > 0:
                df_cate_sub = df_cate.loc[df_cate["product_type"].isin(cate_list), ].copy()
                if df_cate_sub.shape[0] > 0:
                    df_cate_sub["repay_lender_count_max"] = [varnames[x] for x in df_cate_sub["product_type"]]
                    variables.update(df_cate_sub.groupby("repay_lender_count_max").agg({"transaction_cnt": np.max}).to_dict(orient="dict")["transaction_cnt"])

                    df_cate_sub["repay_lender_amt_max"] = [varnames_amt[x] for x in df_cate_sub["product_type"]]
                    variables.update(df_cate_sub.groupby("repay_lender_amt_max").agg({"amount": np.max}).to_dict(orient="dict")["amount"])

                # 汇总
                variables.update({"bank_txn_lender_repay_single_count_max": df_cate.transaction_cnt.max(),
                                  "bank_txn_lender_repay_single_amount_max": df_cate.amount.max()
                                  }
                                )

            return variables

        df = self.raw_cate_g_list_df.loc[self.raw_cate_g_list_df['dr_cr']=='debit', ].copy()
        df_competitor = df.groupby(["user_id", "competitor", "product_type"]).agg(
                amount = pd.NamedAgg(column="amount", aggfunc="sum"),
                transaction_cnt = pd.NamedAgg(column="transaction_cnt", aggfunc="sum"),
                ).reset_index()
        variables = {}
        variables.update(__calc(df, df_competitor, cate_list))
        self.variables.update(variables)
    
    def calc_vars_disburse_single_stat(self, cate_list=['personal loan', 'BNPL', 'cash/wage advance', 'bank']):
        """
        计算历史单机构最大累计借款金额/次数
        """
        def __calc(df, df_cate, cate_list):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = {x: "_".join(["bank_txn_lender_disburse_single", self.prod_dict[x], 'count_max']) for x in cate_list}
            variables = {varnames[x]: -1 for x in cate_list}
            varnames_amt = {x: "_".join(["bank_txn_lender_disburse_single", self.prod_dict[x], 'amount_max']) for x in cate_list}
            variables.update({varnames_amt[x]: -1 for x in cate_list})
            # 汇总
            variables.update({"bank_txn_lender_disburse_single_count_max": -1,
                              "bank_txn_lender_disburse_single_amount_max": -1
                              }
                            )
    
            if df.shape[0] > 0:
                df_cate_sub = df_cate.loc[df_cate["product_type"].isin(cate_list), ].copy()
                if df_cate_sub.shape[0] > 0:
                    df_cate_sub["disburse_lender_count_max"] = [varnames[x] for x in df_cate_sub["product_type"]]
                    variables.update(df_cate_sub.groupby("disburse_lender_count_max").agg({"transaction_cnt": np.max}).to_dict(orient="dict")["transaction_cnt"])

                    df_cate_sub["disburse_lender_amt_max"] = [varnames_amt[x] for x in df_cate_sub["product_type"]]
                    variables.update(df_cate_sub.groupby("disburse_lender_amt_max").agg({"amount": np.max}).to_dict(orient="dict")["amount"])

                # 汇总
                variables.update({"bank_txn_lender_disburse_single_count_max": df_cate.transaction_cnt.max(),
                                  "bank_txn_lender_disburse_single_amount_max": df_cate.amount.max()
                                  }
                                )

            return variables

        df = self.raw_cate_g_list_df.loc[self.raw_cate_g_list_df['dr_cr']=='credit', ].copy()
        df_competitor = df.groupby(["user_id", "competitor", "product_type"]).agg(
                amount = pd.NamedAgg(column="amount", aggfunc="sum"),
                transaction_cnt = pd.NamedAgg(column="transaction_cnt", aggfunc="sum"),
                ).reset_index()
        variables = {}
        variables.update(__calc(df, df_competitor, cate_list))
        self.variables.update(variables)
    
    def calc_vars_summary_disburse_jgs_stat(self, day_list=[None, ]):
        """计算近多段时间内指定类别放款机构数.

        Parameters
        ----------
        cate_list : List
            待计算APP类别列表
        day_list : list
            待计算时段列表
        calc_rto : bool
            是否计算占比，默认值为True
        """
        def __calc(df, day):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = "_".join(["bank_txn_lender_disburse_jgs", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_disburse_jgs"
            variables = {varnames: -1}

            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day, ].copy()
    
            if df.shape[0] > 0:
                variables.update({varnames: df["competitor"].nunique()})
                                        
            return variables

        # df = self.raw_data.loc[self.raw_data['dr_cr']=='credit', ].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr']=='credit', ].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df_cate, day))
        self.variables.update(variables)
    
    def calc_vars_summary_repay_jgs_stat(self, day_list=[None, ]):
        """计算近多段时间内指定类别还款机构数.

        Parameters
        ----------
        cate_list : List
            待计算APP类别列表
        day_list : list
            待计算时段列表
        calc_rto : bool
            是否计算占比，默认值为True
        """
        def __calc(df, day):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = "_".join(["bank_txn_lender_repay_jgs", "l"+str(day)+"d"]) if bool(day) else "bank_txn_lender_repay_jgs"
            variables = {varnames: -1}

            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day, ].copy()
    
            if df.shape[0] > 0:
                variables.update({varnames: df["competitor"].nunique()})
                                        
            return variables

        # df = self.raw_data.loc[self.raw_data['dr_cr']=='debit', ].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr']=='debit', ].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df_cate, day))
        self.variables.update(variables)
    
    def calc_vars_disburse_jgs_stat(self, cate_list, day_list=[None, ]):
        """计算近多段时间内指定类别放款机构数.

        Parameters
        ----------
        cate_list : List
            待计算APP类别列表
        day_list : list
            待计算时段列表
        calc_rto : bool
            是否计算占比，默认值为True
        """
        def __calc(df, df_cate, cate_list, day):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'jgs', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_disburse", self.prod_dict[x], 'jgs']) for x in cate_list}
            variables = {varnames[x]: -1 for x in cate_list}

            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day, ].copy()
                df_cate = df_cate.loc[df_cate["trac_diffdays"] <= day, ].copy()
    
            if df.shape[0] > 0:
                df_cate = df_cate.loc[df_cate["product_type"].isin(cate_list), ].copy()
                if df_cate.shape[0] > 0:
                    df_cate["disburse_lender_jgs"] = [varnames[x] for x in df_cate["product_type"]]
                    variables.update(df_cate.groupby("disburse_lender_jgs")["competitor"].nunique().to_dict())
                                        
            return variables

        df = self.raw_data.loc[self.raw_data['dr_cr']=='credit', ].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr']=='credit', ].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df, df_cate, cate_list, day))
        self.variables.update(variables)
    
    def calc_vars_repay_jgs_stat(self, cate_list, day_list=[None, ]):
        """计算近多段时间内指定类别还款机构数.

        Parameters
        ----------
        cate_list : List
            待计算APP类别列表
        day_list : list
            待计算时段列表
        calc_rto : bool
            是否计算占比，默认值为True
        """
        def __calc(df, df_cate, cate_list, day):
            """计算近day日内安装某类别APP数量.

            Parameters
            ----------
            df : pandas.DataFrame
                APPList数据
            df_cate : pandas.DataFrame
                APP类别List数据
            cate_list : List
                待计算APP类别列表
            day : int
                待计算时段
            calc_rto : bool
                是否计算占比

            Returns
            -------
            variables : dict
                变量结果字典
            """
            varnames = {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'jgs', "l"+str(day)+"d"]) for x in cate_list} if bool(day) else {x: "_".join(["bank_txn_lender_repay", self.prod_dict[x], 'jgs']) for x in cate_list}
            variables = {varnames[x]: -1 for x in cate_list}

            if bool(day):
                df = df.loc[df["trac_diffdays"] <= day, ].copy()
                df_cate = df_cate.loc[df_cate["trac_diffdays"] <= day, ].copy()
    
            if df.shape[0] > 0:
                df_cate = df_cate.loc[df_cate["product_type"].isin(cate_list), ].copy()
                if df_cate.shape[0] > 0:
                    df_cate["repay_lender_jgs"] = [varnames[x] for x in df_cate["product_type"]]
                    variables.update(df_cate.groupby("repay_lender_jgs")["competitor"].nunique().to_dict())
                                        
            return variables

        df = self.raw_data.loc[self.raw_data['dr_cr']=='debit', ].copy()
        df_cate = self.raw_cate_list_df.loc[self.raw_cate_list_df['dr_cr']=='debit', ].copy()
        variables = {}
        for day in day_list:
            variables.update(__calc(df, df_cate, cate_list, day))
        self.variables.update(variables)
