from pathlib import Path
import mlflow
import numpy as np
from typing import List, Optional
from sklearn import metrics
from NegativeClassOptimization import config, ml, utils, preprocessing, visualisations


class DataPipeline:
    """Fetch, process and handle data for ML tasks.
    """   
    
    @staticmethod
    def load_processed_dataframes(
        dir_path = config.DATA_SLACK_1_PROCESSED_DIR,
        sample: Optional[int] = None,
        ) -> dict:
        return utils.load_processed_dataframes(
            dir_path, 
            sample,
        )


class MulticlassPipeline(DataPipeline):


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


    def step_1_process_data(
        self,
        input_data_dir: Path,
        ags: List[str],
        batch_size: int,
        sample_data_source: Optional[int] = None,
        sample_train_val: Optional[int] = None,
    ):

        dfs = MulticlassPipeline.load_processed_dataframes(
            dir_path = input_data_dir,
            sample = sample_data_source,
        )

        df_train = dfs["train_val"]
        df_train, scaler, encoder = preprocessing.preprocess_df_for_multiclass(
            df_train, 
            ags, 
            sample_train=sample_train_val
            )
        
        if self.log_mlflow:
            mlflow.log_params({"encoder_classes": "_".join(encoder.classes_)})
        
        df_test = dfs["test_closed_exclusive"]
        df_test, _, _ = preprocessing.preprocess_df_for_multiclass(
            df_test,
            ags,
            scaler,
            encoder,
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

        if self.log_mlflow:
            utils.mlflow_log_params_online_metrics(online_metrics)
        
        if self.save_model_mlflow:
            mlflow.pytorch.log_model(model, "pytorch_model")

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
                    k1: v1.tolist() for k1, v1 in eval_metrics["closed"].items() if type(v1) != np.ndarray
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
            report: dict = metrics.classification_report(
                y_test, 
                y_test_pred,
                target_names=self.encoder.classes_,
                output_dict=True
                )
            mlflow.log_dict(report, "classification_report.json")
        
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
        
        fig_confusion_matrices, _ = visualisations.plot_confusion(
            cm=self.eval_metrics["closed"]["confusion_matrix_closed"],
            cm_normed=self.eval_metrics["closed"]["confusion_matrix_normed_closed"],
            class_names=self.encoder.classes_,
        )

        if self.log_mlflow:
            mlflow.log_figure(fig_abs_logit_distr, "fig_abs_logit_distr.png")
            mlflow.log_figure(fig_confusion_matrices, "fig_confusion_matrices.png")
        
        self.is_step_4_complete = True

