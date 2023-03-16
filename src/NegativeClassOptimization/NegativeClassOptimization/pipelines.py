from itertools import combinations
import logging
from pathlib import Path
import warnings
import mlflow
import numpy as np
from typing import List, Optional, Union
import pandas as pd
from sklearn import metrics
from NegativeClassOptimization import config, datasets, ml, utils, preprocessing, visualisations
import torch


class DataPipeline:
    """Fetch, process and handle data for ML tasks.
    """   
    
    def __init__(
        self, 
        log_mlflow: bool = False, 
        save_model_mlflow: bool = False,
        ):
        self.is_step_1_complete = False
        self.is_step_2_complete = False
        self.is_step_3_complete = False
        self.is_step_4_complete = False
        self.log_mlflow = log_mlflow
        self.save_model_mlflow = save_model_mlflow

    @staticmethod
    def load_processed_dataframes(
        dir_path = config.DATA_SLACK_1_PROCESSED_DIR,
        sample: Optional[int] = None,
        ) -> dict:
        return utils.load_processed_dataframes(
            dir_path, 
            sample,
        )
    
    @staticmethod
    def load_global_dataframe(
        dir_path = config.DATA_SLACK_1_GLOBAL,
        ) -> pd.DataFrame:
        return utils.load_global_dataframe(
            dir_path, 
        )


class BinaryclassPipeline(DataPipeline):
    """Organized workflow for binary classification. 
    """

    def loader(
        self,
        ag_pos,
        ag_neg,
        N,
        ):
        """Load data for binary classification.
        """
        df = utils.load_1v1_binary_dataset(
            ag_pos=ag_pos,
            ag_neg=ag_neg,
            num_samples=N,
            drop_duplicates=False,
            with_paratopes=False,
        )
        return df

    def step_1_process_data(
        self,
        ag_pos: str,
        ag_neg: Union[str, List[str]],
        N: Optional[int] = None,
        sample_train: Optional[int] = None,
        batch_size: int = 64,
        split_id: int = 0,
        shuffle_antigen_labels: bool = False,
        load_from_miniabsolut: bool = False,
        load_from_miniabsolut_split_seed: Optional[int] = None,
        ):
        """Process data for binary classification.
        """

        if load_from_miniabsolut:
            df_train_val, df_test_closed = self._load_from_miniabsolut(ag_pos, ag_neg, split_seed=load_from_miniabsolut_split_seed)
        
        else:
            df = self.loader(ag_pos, ag_neg, N)

            if "Slide_farmhash_mod_10" not in df.columns:
                warnings.warn("Slide_farmhash_mod_10 not in df.columns. Adding it now.")
                df["Slide_farmhash_mod_10"] = df["Slide"].apply(lambda x: preprocessing.farmhash_mod_10(x))

            df_train_val = df.loc[df["Slide_farmhash_mod_10"] != split_id].copy()
            df_test_closed = df.loc[df["Slide_farmhash_mod_10"] == split_id].copy()

        if shuffle_antigen_labels:

            df_train_val["Antigen"] = df_train_val["Antigen"].sample(frac=1).values
            df_test_closed["Antigen"] = df_test_closed["Antigen"].sample(frac=1).values

        train_data, test_data, train_loader, test_loader = (
            preprocessing.preprocess_data_for_pytorch_binary(
                df_train_val=df_train_val,
                df_test_closed=df_test_closed,
                ag_pos=[ag_pos],
                batch_size=batch_size,
                scale_X=False,
                sample_train=sample_train,
        ))

        if self.log_mlflow:
            mlflow.log_params({
                "ag_pos": ag_pos,
                "ag_neg": ag_neg,
                "sample_train": sample_train,
                "batch_size": batch_size,
                "split_id": split_id,
                "shuffle_antigen_labels": shuffle_antigen_labels,
                "load_from_miniabsolut": load_from_miniabsolut,
                "load_from_miniabsolut_split_seed": load_from_miniabsolut_split_seed,

                "N_train": len(train_loader.dataset),
                "N_closed": len(test_loader.dataset),
            })

            for i in range(3):
                uid = utils.get_uid()
                if len(list(config.TMP_DIR.glob(f"*{uid}*tsv"))) == 0:
                    break
                else:
                    if i == 2:
                        raise ValueError("Could not find unique uid.")
                    else:
                        continue

            train_data.df.to_csv(
                config.TMP_DIR / f"{uid}_train_dataset.tsv", 
                sep='\t',
                index=False)
            mlflow.log_artifact(config.TMP_DIR / f"{uid}_train_dataset.tsv", "dataset/train_dataset.tsv")

            test_data.df.to_csv(
                config.TMP_DIR / f"{uid}_test_dataset.tsv", 
                sep='\t',
                index=False)
            mlflow.log_artifact(config.TMP_DIR / f"{uid}_test_dataset.tsv", "dataset/test_dataset.tsv")

        self.ag_pos = ag_pos
        self.ag_neg = ag_neg
        self.N = N
        self.shuffle_antigen_labels = shuffle_antigen_labels
        self.load_from_miniabsolut = load_from_miniabsolut
        self.batch_size = batch_size
        self.split_id = split_id
        self.df_train_val = df_train_val
        self.df_test_closed = df_test_closed
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.is_step_1_complete = True


    def _miniabsolut_reader(self, ag, name, split_seed = None):

        if split_seed is None:
            path = config.DATA_MINIABSOLUT / ag / name
        else:
            path = config.DATA_MINIABSOLUT_SPLITS / f"MiniAbsolut_Seed{split_seed}" / ag / name

        return pd.read_csv(
            path,
            sep="\t",
            dtype={"Antigen": str}
        )


    def _load_from_miniabsolut(self, ag_pos, ag_neg, split_seed = None):

        # Load positive data
        df_train_val_pos = self._miniabsolut_reader(ag_pos, "high_train_15000.tsv", split_seed=split_seed)
        df_test_closed_pos = self._miniabsolut_reader(ag_pos, "high_test_5000.tsv", split_seed=split_seed)
            
        # Load negative data
        if isinstance(ag_neg, str):
            ag_neg = [ag_neg]
        N_neg = len(ag_neg)
        train_samples_per_ag_neg = 15000 // N_neg
        test_samples_per_ag_neg = 5000 // N_neg

        train_neg_dfs = []
        test_neg_dfs = []
        for ag_neg_i in ag_neg:
            df_train_val_neg_i = self._miniabsolut_reader(ag_neg_i, f"high_train_15000.tsv", split_seed=split_seed)
            df_train_val_neg_i = df_train_val_neg_i.iloc[:train_samples_per_ag_neg].copy()
            df_test_closed_neg_i = self._miniabsolut_reader(ag_neg_i, f"high_test_5000.tsv", split_seed=split_seed)
            df_test_closed_neg_i = df_test_closed_neg_i.iloc[:test_samples_per_ag_neg].copy()

            train_neg_dfs.append(df_train_val_neg_i)
            test_neg_dfs.append(df_test_closed_neg_i)

        df_train_val_neg = pd.concat(train_neg_dfs, axis=0).sample(frac=1).reset_index(drop=True)
        df_test_closed_neg = pd.concat(test_neg_dfs, axis=0).sample(frac=1).reset_index(drop=True)

        # Aggregate positive and negative dataframes and shuffle
        df_train_val = pd.concat(
                [df_train_val_pos, df_train_val_neg], axis=0
                ).sample(frac=1).reset_index(drop=True)
        df_test_closed = pd.concat(
                [df_test_closed_pos, df_test_closed_neg], axis=0
                ).sample(frac=1).reset_index(drop=True)
            
        return df_train_val, df_test_closed


    def step_2_train_model(
        self,
        input_dim = 220,
        num_hidden_units = 10,
        seed_id: int = 0,
        epochs: int = 50,
        learning_rate: float = 0.001,
        optimizer_type = "Adam",
        momentum = 0.9,
        weight_decay = 0,
        swa: bool = False,
        ):
        """Train model for binary classification.
        """
        # torch.manual_seed(seed_id)
        utils.nco_seed(seed_id)
        model = ml.SNN(num_hidden_units, input_dim)

        if self.save_model_mlflow:
            callback_on_model_end_epoch = lambda model, epoch: mlflow.pytorch.log_model(model, f"models/trained_model_epoch_{epoch}")
        else:
            callback_on_model_end_epoch = None

        train_output = ml.train_for_ndb1(
            epochs,
            learning_rate, 
            self.train_loader, 
            self.test_loader, 
            None,  # open_loader
            model,
            optimizer_type=optimizer_type,
            momentum=momentum,
            weight_decay=weight_decay,
            callback_on_model_end_epoch=callback_on_model_end_epoch,
            swa=swa,
            )
        
        if swa:
            swa_model, model, online_metrics = train_output
        else:
            online_metrics = train_output
        
        if self.log_mlflow:
            mlflow.log_params({
                "input_dim": input_dim,
                "num_hidden_units": num_hidden_units,
                "seed_id": seed_id,
                "epochs": epochs,
                "learning_rate": learning_rate,
                "optimizer_type": optimizer_type,
                "momentum": momentum,
                "weight_decay": weight_decay,
                "swa": swa,
            })
            utils.mlflow_log_params_online_metrics(online_metrics)

        if self.save_model_mlflow:
            mlflow.pytorch.log_model(model, "models/trained_model")
            mlflow.pytorch.log_model(swa_model, "models/swa_model")

        self.input_dim = input_dim
        self.num_hidden_units = num_hidden_units
        self.seed_id = seed_id
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.momentum = momentum
        self.weight_decay = weight_decay

        self.model = model
        self.swa_model = swa_model
        self.online_metrics = online_metrics

        self.is_step_2_complete = True


    def step_3_evaluate_model(self):
        """Evaluate model for binary classification.
        """
        eval_metrics = ml.evaluate_on_closed_and_open_testsets(
            None,  # open_loader 
            self.test_loader, 
            self.model)

        if self.log_mlflow:
            mlflow.log_dict(
                {
                    **{k1: v1.tolist() if type(v1) == np.ndarray else v1 for k1, v1 in eval_metrics["closed"].items()},
                    **{k2: v2.tolist() if type(v2) == np.ndarray else v2 for k2, v2 in eval_metrics["open"].items()},
                }, 
                "eval_metrics.json"
            )
        self.eval_metrics = eval_metrics


    def step_4_visualize(self):
        """Visualize model for binary classification.
        """
        warnings.warn("No visualization for binary classification is setup.")


class BinaryclassBindersPipeline(BinaryclassPipeline):

    def loader(
        self,
        ag_pos,
        ag_neg,
        N,
        ):
        """Load data for binary classification.
        """
        
        dataset_type = self.get_dataset_type(ag_pos, ag_neg)

        df = utils.build_binding_dataset_per_ag(
            ag_pos.split("_")[0],  # antigen name
            dataset_type,
            )
        return df

    
    def _load_from_miniabsolut(
        self,
        ag_pos,
        ag_neg,
        split_seed = None,
        ):
        """Load data for binary classification.
        """
        ag = ag_pos.split("_")[0]  # antigen name
        dataset_type = self.get_dataset_type(ag_pos, ag_neg)

        df_train_val_pos = self._miniabsolut_reader(ag, "high_train_15000.tsv", split_seed=split_seed)
        df_train_val_pos["Antigen"] = ag_pos
        df_test_closed_pos = self._miniabsolut_reader(ag, "high_test_5000.tsv", split_seed=split_seed)
        df_test_closed_pos["Antigen"] = ag_pos

        if dataset_type == "high_looser":
            df_train_val_neg = self._miniabsolut_reader(ag, "looserX_train_15000.tsv", split_seed=split_seed)
            df_test_closed_neg = self._miniabsolut_reader(ag, "looserX_test_5000.tsv", split_seed=split_seed)

        elif dataset_type == "high_95low":
            df_train_val_neg = self._miniabsolut_reader(ag, "95low_train_15000.tsv", split_seed=split_seed)
            df_test_closed_neg = self._miniabsolut_reader(ag, "95low_test_5000.tsv", split_seed=split_seed)
        
        df_train_val_neg["Antigen"] = ag_neg
        df_test_closed_neg["Antigen"] = ag_neg

        df_train_val = pd.concat([df_train_val_pos, df_train_val_neg], axis=0).sample(frac=1).reset_index(drop=True)
        df_test_closed = pd.concat([df_test_closed_pos, df_test_closed_neg], axis=0).sample(frac=1).reset_index(drop=True)

        return df_train_val, df_test_closed


    def get_dataset_type(self, ag_pos, ag_neg):
        ag = ag_pos.split("_")[0]  # antigen name
        assert ag_pos.split("_")[1] == "high", "ag_pos must be in format '{ag}_high'."
        assert ag == ag_neg.split("_")[0], "ag_pos and ag_neg must be from the same antigen."

        ag_neg_type = ag_neg.split("_")[1]
        if ag_neg_type == "looser":
            dataset_type = "high_looser"
        elif ag_neg_type == "95low":
            dataset_type = "high_95low"
        else:
            raise ValueError(f"ag_neg_type={ag_neg_type} not recognized.")
        return dataset_type


class MulticlassPipeline(DataPipeline):


    def step_1_process_data(
        self,
        input_data_dir: Path,
        ags: List[str],
        batch_size: int,
        sample_data_source: Optional[int] = None,
        sample_train_val: Optional[int] = None,
        sample_test: Optional[int] = None,
        sample_per_ag_train: Optional[int] = None,
        sample_per_ag_test: Optional[int] = None,
    ):

        dfs = MulticlassPipeline.load_processed_dataframes(
            dir_path = input_data_dir,
            sample = sample_data_source,
        )

        print(
            f'Dataframe sizes: {dfs["train_val"].shape=} | '
            f'{dfs["test_closed_exclusive"].shape=} | '
            f'{dfs["test_open_exclusive"].shape=}'
        )

        df_train = dfs["train_val"]
        df_train, scaler, encoder = preprocessing.preprocess_df_for_multiclass(
            df_train, 
            ags, 
            sample_per_ag=sample_per_ag_train,
            )
        
        self._log_encoder(encoder)
        
        df_test = dfs["test_closed_exclusive"]
        df_test, _, _ = preprocessing.preprocess_df_for_multiclass(
            df_test,
            ags,
            scaler,
            encoder,
            sample_per_ag=sample_per_ag_test,
            )

        _, train_loader = ml.construct_dataset_loader_multiclass(df_train, batch_size)
        _, test_loader = ml.construct_dataset_loader_multiclass(df_test, batch_size)
        _, open_loader = preprocessing.construct_open_dataset_loader(
            dfs["test_open_exclusive"],
            batch_size=batch_size,
            scaler=scaler
            )

        if self.log_mlflow:
            mlflow.log_params({
                "N_train": len(train_loader.dataset),
                "N_closed": len(test_loader.dataset),
                "N_open": len(open_loader.dataset),
            })
        
        self.input_data_dir = input_data_dir
        self.ags = ags
        self.sample_data_source = sample_data_source
        self.sample_train_val = sample_train_val
        self.sample_test = sample_test
        self.sample_per_ag_train = sample_per_ag_train
        self.sample_per_ag_test = sample_per_ag_test
        self.batch_size = batch_size
        self.dfs = dfs
        self.df_train = df_train
        self.df_test = df_test
        self.encoder = encoder
        self.scaler = scaler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.open_loader = open_loader

        self.is_step_1_complete = True


    def step_2_train_model(
        self, 
        model, 
        loss_fn, 
        optimizer,
        epochs, 
    ):
        assert self.is_step_1_complete  
        
        online_metrics = []
        for t in range(epochs):
        
            print(f"Epoch {t+1}\n-------------------------------")
            losses = ml.train_loop(self.train_loader, model, loss_fn, optimizer)
            test_metrics = ml.test_loop(self.test_loader, model, loss_fn)
            open_metrics = ml.openset_loop(self.open_loader, self.test_loader, model)
            
            online_metrics.append({
                    "train_losses": losses,
                    "test_metrics": test_metrics,
                    "open_metrics": open_metrics,
                })
            
            if self.save_model_mlflow:
                mlflow.pytorch.log_model(model, f"models/pytorch_model_epoch_{t+1}")

        if self.log_mlflow:
            mlflow.log_params({"model_num_params": sum(p.numel() for p in model.parameters())})
            utils.mlflow_log_params_online_metrics(online_metrics)
        
        if self.save_model_mlflow:
            mlflow.pytorch.log_model(model, "models/pytorch_model")

        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epochs = epochs
        self.online_metrics = online_metrics

        self.is_step_2_complete = True


    def step_3_evaluate_model(self):
        assert self.is_step_2_complete
        
        eval_metrics = ml.evaluate_on_closed_and_open_testsets(
            self.open_loader, 
            self.test_loader, 
            self.model
            )
        
        if self.log_mlflow:
            mlflow.log_dict(
                {
                    **{k1: v1.tolist() if type(v1) == np.ndarray else v1 for k1, v1 in eval_metrics["closed"].items()},
                    **{k2: v2.tolist() if type(v2) == np.ndarray else v2 for k2, v2 in eval_metrics["open"].items()},
                }, 
                "eval_metrics.json"
            )

            mlflow.log_metrics(
                {
                    k1: v1 for k1, v1 in eval_metrics["closed"].items() if type(v1) not in (np.ndarray, list)
                }
            )
            # Instead of mlflow.log_metrics(eval_metrics["open"])
            #  does some renaming.
            mlflow.log_metrics(
                {
                    'open_avg_precision': eval_metrics["open"]["avg_precision_open"],
                    'open_acc': eval_metrics["open"]["acc_open"],
                    'open_recall': eval_metrics["open"]["recall_open"],
                    'open_precision': eval_metrics["open"]["precision_open"],
                    'open_f1': eval_metrics["open"]["f1_open"],
                }
            )

            # Other artifacts
            x_test, y_test = ml.Xy_from_loader(self.test_loader)
            y_test_pred = self.model.predict(x_test)
            ## 
            # report: dict = metrics.classification_report(
            #     y_test, 
            #     y_test_pred,
            #     target_names=self.encoder.classes_,
            #     output_dict=True
            #     )
            # mlflow.log_dict(report, "classification_report.json")
        
        self.eval_metrics = eval_metrics
        self.is_step_3_complete = True
    

    def step_4_visualise(self):
        fig_abs_logit_distr, _ = visualisations.plot_abs_logit_distr(
            self.eval_metrics["open"], 
            metadata={
                "ag_pos": self.ags,
                "ag_neg": "",
                "N_train": len(self.train_loader.dataset),
                "N_closed": len(self.test_loader.dataset),
                "N_open": len(self.open_loader.dataset),
            },
            )
        
        try:
            fig_confusion_matrices, _ = visualisations.plot_confusion(
                cm=self.eval_metrics["closed"]["confusion_matrix_closed"],
                cm_normed=self.eval_metrics["closed"]["confusion_matrix_normed_closed"],
                class_names=self.encoder.classes_,
            )
        except KeyError as error:
            print(error)
            pass


        if self.log_mlflow:
            try:
                mlflow.log_figure(fig_abs_logit_distr, "fig_abs_logit_distr.png")
                mlflow.log_figure(fig_confusion_matrices, "fig_confusion_matrices.png")
            except:
                pass
        
        self.is_step_4_complete = True


    def _log_encoder(self, encoder):
        if self.log_mlflow:
            encoder_str = "__".join(encoder.classes_)
            if len(encoder_str) <= 500:
                mlflow.log_params({"encoder_classes": encoder_str})
            else:
                mlflow.log_params({"encoder_classes": "TOO_LONG"})


class MultilabelPipeline(MulticlassPipeline):

    def step_1_process_data(self,
        input_data_dir: Path,
        ags: List[str],
        batch_size: int,
        sample_data_source: Optional[int] = None,
        sample_train_val: Optional[int] = None,
        sample_test: Optional[int] = None,
        sample_per_ag_train: Optional[int] = None,
        sample_per_ag_test: Optional[int] = None,
        ):

        dfs = MulticlassPipeline.load_processed_dataframes(
            dir_path = input_data_dir,
            sample = sample_data_source,
        )

        df_train = dfs["train_val"]
        df_train, scaler, encoder = preprocessing.preprocess_df_for_multilabel(
            df_train, 
            ags, 
            sample_per_ag=sample_per_ag_train,
            )
        
        df_test = dfs["test_closed_exclusive"]
        df_test, _, _ = preprocessing.preprocess_df_for_multilabel(
            df_test,
            ags,
            scaler,
            encoder,
            sample_per_ag=sample_per_ag_test,
            )

        self._log_encoder(encoder)

        _, train_loader = ml.construct_dataset_loader(df_train, batch_size, dataset_class=datasets.MultilabelDataset)
        _, test_loader = ml.construct_dataset_loader(df_test, batch_size, dataset_class=datasets.MultilabelDataset)
        _, open_loader = preprocessing.construct_open_dataset_loader(
            dfs["test_open_exclusive"],
            batch_size=batch_size,
            scaler=scaler
            )

        if self.log_mlflow:
            mlflow.log_params({
                "N_train": len(train_loader.dataset),
                "N_closed": len(test_loader.dataset),
                "N_open": len(open_loader.dataset),
            })
        
        self.input_data_dir = input_data_dir
        self.ags = ags
        self.sample_data_source = sample_data_source
        self.sample_train_val = sample_train_val
        self.sample_test = sample_test
        self.sample_per_ag_train = sample_per_ag_train
        self.sample_per_ag_test = sample_per_ag_test
        self.batch_size = batch_size
        self.dfs = dfs
        self.df_train = df_train
        self.df_test = df_test
        self.encoder = encoder
        self.scaler = scaler
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.open_loader = open_loader

        self.is_step_1_complete = True


class NDB1_Assymetry_from_Absolut_Builder:
    
    def __init__(self, ags_closed, dataset = None) -> None:
        self.ags_c = ags_closed
        if dataset:
            self.dataset = dataset
        else:
            self.dataset = datasets.AbsolutDataset3()
        
        self.step_01_done = False


    def step_01_select_random_pairs(self, num_ag_pairs: int):
        rng = np.random.default_rng(seed=config.SEED)
        ag_pairs = list(map(tuple, rng.choice(
            list(combinations(self.ags_c, 2)), 
            size=num_ag_pairs,
            replace=False
        ).tolist()))
        self.ag_pairs = ag_pairs
        self.step_01_done = True


    def step_02_convert_to_global_format(self):
        assert self.step_01_done == True
        
        frames = {}
        for ag_pair in self.ag_pairs:
            df_pair = self.dataset.df_wide[list(ag_pair)].copy()
            
            # only keep slide which bind to exactly one of the two antigens
            df_pair = df_pair.loc[df_pair.sum(axis=1) == 1]

            df_pair = preprocessing.convert_wide_to_global(df_pair)
            frames[ag_pair] = df_pair

        self.frames = frames
        self.step_02_done = True