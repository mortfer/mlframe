import pyspark.sql as pys
import pyspark.sql.functions as psf
import pandas as pd
from typing import Dict

def method_chaining(method: str, *initial_method_args, **initial_method_kwargs):
    def inner_func(df, *extra_args, **extra_kwargs):
        return getattr(df, method)(
            *(initial_method_args + extra_args),
            **{**initial_method_kwargs, **extra_kwargs},
        )

    return inner_func


def joiner(df1, df2, on, how, drop_keys: str = None, method: str = "join"):
    if isinstance(df1, pys.DataFrame) and isinstance(df2, pys.DataFrame):
        df1 = df1.alias("df1")
        df2 = df1.alias("df2")
        join_condition = []
        if all(isinstance(col, str) for col in on):
            join_condition = [
                psf.col(f"df1.{col}") == psf.col(f"df2.{col}") for col in on
            ]
        else:
            for cond in on:
                if isinstance(cond, str):
                    join_condition.append(
                        psf.col(f"df1.{cond}") == psf.col(f"df2.{cond}")
                    )
                elif isinstance(cond, tuple) and len(cond) == 2:
                    col_df1, col_df2_or_tuple = cond
                    if isinstance(col_df2_or_tuple, str):
                        join_condition.append(
                            psf.col(f"df1.{col_df1}")
                            == psf.col(f"df2.{col_df2_or_tuple}")
                        )
                    elif (
                        isinstance(col_df2_or_tuple, tuple)
                        and len(col_df2_or_tuple) == 2
                    ):
                        col_df2, operator = col_df2_or_tuple
                        if operator == ">":
                            join_condition.append(
                                psf.col(f"df1.{col_df1}") > psf.col(f"df2.{col_df2}")
                            )
                        elif operator == "<":
                            join_condition.append(
                                psf.col(f"df1.{col_df1}") < psf.col(f"df2.{col_df2}")
                            )
                        elif operator == ">=":
                            join_condition.append(
                                psf.col(f"df1.{col_df1}") >= psf.col(f"df2.{col_df2}")
                            )
                        elif operator == "<=":
                            join_condition.append(
                                psf.col(f"df1.{col_df1}") <= psf.col(f"df2.{col_df2}")
                            )
                        else:
                            raise ValueError(f"Unsupported operator: {operator}")
                    else:
                        raise ValueError("Invalid condition format in 'on' parameter.")
                else:
                    raise ValueError(
                        "Invalid 'on' format. Use List of column names or tuples"
                    )

        df_joined = getattr(df1, method)(df2, on=join_condition, how=how)

        if drop_keys is not None:
            if drop_keys == "right":
                df1_columns = [f"df1.{c}" for c in df1.columns]
                df2_columns = [f"df2.{c}" for c in df2.columns if c not in on]
            elif drop_keys == "left":
                df1_columns = [f"df1.{c}" for c in df1.columns if c not in on]
                df2_columns = [f"df2.{c}" for c in df2.columns]
            else:
                raise ValueError("Invalid drop_keys value.")
            df_joined = df_joined.select(*df1_columns, *df2_columns)
        return df_joined
    elif isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        raise NotImplementedError
    else:
        raise NotImplementedError

def df_filter(df, eq: Dict = None, ne: Dict = None, le: Dict = None, gt: Dict = None):
    def get_column(df, column):
        if isinstance(df, pd.DataFrame):
            return df[column]
        elif isinstance(df, pys.DataFrame):
            return getattr(df, column)
        else:
            raise Exception("what is this?")

    if eq is not None:
        for column, values in eq.items():
            df = df[get_column(df, column).isin(values)]
    if ne is not None:
        for column, values in ne.items():
            df = df[~get_column(df, column).isin(values)]
    if le is not None:
        for column, values in le.items():
            df = df[get_column(df, column)<=values]
    if gt is not None:
        for column, values in gt.items():
            df = df[get_column(df, column)>values]
    return df

def cast_df(casting_map: Dict):
    def inner_f(df):
        for col, col_type in casting_map.items():
            if col in df.columns:
                df = df.withColumn(col, psf.col(col).cast(col_type))
            else:
                # print(f"Column {col} not found to be cast")
                pass
        return df
    return inner_f


def drop_constant_columns(columns_to_eval):
    def func(df):
        if columns_to_eval is None:
            cols = df.columns
        else:
            cols = columns_to_eval
        
        if isinstance(df, pys.DataFrame):
            cnt = df.agg(*(psf.countDistinct(c).alias(c) for c in cols)).first()
            cols_to_drop = [c for c in cnt.asDict() if cnt[c] == 1]
            print(f"Dropping {cols_to_drop} because they are constant")
            df = df.drop(*cols_to_drop)
        elif isinstance(df, pd.DataFrame):
            cols_to_drop = df[cols].apply(pd.Series.nunique, axis=1) == 1
            cols_to_drop = [c for c, b in zip(cols, cols_to_drop) if b is True]
            print(f"Dropping {cols_to_drop} because they are constant")
            df = df.drop(cols_to_drop, axis=1)
        else:
            raise Exception("Unknown DataFrame type")
        return df
    return func

def pandas_expand_columns(df, column):
    # Expand columns
    embedding_df = pd.DataFrame(df[column].tolist(), index=df.index)
    embedding_df.columns = [f"{column}_{i}" for i in range(embedding_df.shape[1])]
    # Concatenate the original dataframe  excluding the column to expand with the expanded dataframe
    df_expanded = pd.concat([df.drop(columns=column), embedding_df], axis=1)
    return df_expanded