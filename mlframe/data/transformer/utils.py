import pyspark.sql as pys
import pyspark.sql.functions as psf
import pandas as pd


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
