from NegativeClassOptimization.preprocessing import onehot_encode_df, remove_duplicates_for_binary, sample_train_val
import numpy as np
import pandas as pd
import farmhash
import unittest

import NegativeClassOptimization.utils as utils
import NegativeClassOptimization.ml as ml


class GenericTests(unittest.TestCase):

    def test_deterministic_split_script_07(self):
        
        processed_dfs: dict = utils.load_processed_dataframes()
        ag_pos = ['3VRL']
        ag_neg = ['1FBI', '1WEJ', '5E94']
        sample_train = 73000

        df_train_val: pd.DataFrame = processed_dfs["train_val"]
        df_train_val = df_train_val.loc[df_train_val["Antigen"].isin([*ag_pos, *ag_neg])]
        
        print(f"df_train_val -- {df_train_val.shape}")

        print(f"random split: sample_train -- {sample_train} and df_train_val.size -- {df_train_val.sample(sample_train).shape}")
        
        if sample_train:
            if sample_train <= df_train_val.size:
                # deterministic split
                num_buckets = 16384
                df_train_val[f"Slide_farmhash_mod_{num_buckets}"] = list(map(
                    lambda s: farmhash.hash64(s) % num_buckets,
                    df_train_val["Slide"]
                ))
                sampling_frac = sample_train / df_train_val.shape[0]
                num_buckets_to_sample = np.round(sampling_frac * num_buckets)
                df_train_val = (
                    df_train_val
                    .loc[
                        df_train_val[f"Slide_farmhash_mod_{num_buckets}"] <= num_buckets_to_sample
                    ].copy()
                )
                print(f"deterministic split: sample_train -- {sample_train} and df_train_val.shape -- {df_train_val.shape}")
            else:
                raise OverflowError(f"sample_train {sample_train} > train_val size.")
    
    def test_remove_duplicates_for_binary(self):
        raise NotImplementedError()
        processed_dfs: dict = utils.load_processed_dataframes()
        df_train_val = processed_dfs["train_val"]
        df = remove_duplicates_for_binary(df_train_val, ag_pos=["1NSN", "1OB1"])

    def test_preprocessing_train_sampling(self):
        processed_dfs: dict = utils.load_processed_dataframes()
        df_train_val = processed_dfs["train_val"].sample(100)
        print(f"df_train_val.shape: {df_train_val.shape}")
        df = remove_duplicates_for_binary(df_train_val, ag_pos=["1NSN", "1OB1"])
        print(f"df_train_val.shape after dup removal: {df.shape}")        
        df = onehot_encode_df(df)
        print(f"df_train_val.shape after onehot_encode_df: {df.shape}")
        df = sample_train_val(df, 73)
        print(f"df_train_val.shape after sample_train_val 73000: {df.shape}")
    
    # def test_basic_MulticlassSN10(self):
    #     y_pred = model.forward(torch.Tensor(df["X"]))
    #     print(f"{y_pred.shape=}")

    #     y_true = torch.Tensor(df["y"]).type(torch.long)
    #     print(f"{y_true.shape=}")

    #     loss = nn.CrossEntropyLoss()(y_pred, y_true)
    #     print(f"{loss=}")


if __name__ == "__main__":
    unittest.main()