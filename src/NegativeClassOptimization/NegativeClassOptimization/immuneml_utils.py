"""
Generate yaml specification file for running immuneML analyses.
"""


import warnings
import yaml
import uuid
import NegativeClassOptimization.config as config


NUM_PROCESSES = 12

METRICS_LIST = [
    "accuracy",
    "balanced_accuracy",
    "confusion_matrix",
    "f1_micro",
    "f1_macro",
    "f1_weighted",
    "precision",
    "recall",
    "auc",
    "log_loss",
]


# Models
DEFAULT_MODELS = {
    "logistic_default_model": "LogisticRegression",
    "svm_default_model": "SVM",
    "randomforest_default_model": "RandomForestClassifier",
    "knn_default_model": "KNN"
}


# Encodings
DEFAULT_ENCODINGS = {
    "onehot_encoding": {
        "OneHot": {
            "use_positional_info": True,
            "distance_to_seq_middle": 6,
            "flatten": True,
            "sequence_type": "amino_acid",
        }
    },
    "kmerfreq_encoding": {
        "KmerFrequency": {
            "normalization_type": "RELATIVE_FREQUENCY",
            "reads": "UNIQUE",
            "sequence_encoding": "CONTINUOUS_KMER",
            "sequence_type": "AMINO_ACID",
            "k": 3,
            "scale_to_unit_variance": True,
            "scale_to_zero_mean": True,
        }
    },
    "word2vec_encoding": {
        "Word2Vec": {
            "vector_size": 16,
            "k": 3,
            "model_type": "SEQUENCE",
            "epochs": 100,
            "window": 8,
        }
    }
}


# Reports
DEFAULT_REPORTS = {
    "dataset_simple_dataset_overview": "SimpleDatasetOverview",
    "encoding_feature_value_barplot": "FeatureValueBarplot",
    "encoding_feature_distribution": "FeatureDistribution",
    "encoding_feature_comparison": {
        "FeatureComparison": {
            "comparison_label": "binder",
        }
    },
    "encoding_design_matrix_explorer": {"DesignMatrixExporter": {
        "file_format": "csv"
    }
    },
    "training_model_ml_settings_performance": "MLSettingsPerformance",
    "model_roccurve": "ROCCurve",
    "model_training_performance": {
        "TrainingPerformance": {
            "metrics": METRICS_LIST,
        }
    },
    "model_confounder_analysis": {
        "ConfounderAnalysis": {
            "metadata_labels": ["Antigen"]
        }
    },
}
TRAINING_REPORTS_LIST = [
    "training_model_ml_settings_performance",
]
MODEL_REPORTS_LIST = [
    "model_roccurve",
    "model_training_performance",
    "model_confounder_analysis",
]


class ImmuneMLSpecBuilder:
    """
    Build ImmuneML specifications.
    """

    def __init__(
        self,
        datasets: dict, encoding, ml_model, metric, selection, k_fold, refit_best, add_explore, metadata
    ):
        self.spec = ImmuneMLSpecBuilder._build_specification(datasets, encoding, ml_model, metric, selection, k_fold, refit_best, add_explore,metadata)

    @staticmethod
    def _build_specification(datasets, encoding, ml_model, metric, selection, k_fold, refit_best, add_explore, metadata=False) -> dict:
        instructions = ImmuneMLSpecBuilder._build_instructions(datasets, encoding, ml_model, metric, selection, k_fold, refit_best, add_explore, metadata)
        specification = {
            "definitions": {
                "datasets": datasets,
                "encodings": DEFAULT_ENCODINGS,
                "ml_methods": DEFAULT_MODELS,
                "reports": DEFAULT_REPORTS,
            },
            "instructions": instructions,
            "output": {
                "format": "HTML"
            }
        }
        return specification

    @staticmethod
    def _build_instructions(datasets, encoding, ml_model, metric, selection, k_fold, refit_best, add_explore, metadata) -> dict:
        
        dataset_names: list[str] = list(datasets.keys())

        explore_instructs = {}
        if add_explore:
            for name in dataset_names:
                explore_instructs[f"explore_{name}_instruction"] = {
                    "type": "ExploratoryAnalysis",
                    "analyses": {
                            "raw_overview": {
                                "dataset": name,
                                "report": "dataset_simple_dataset_overview",
                            },
                    },
                    "number_of_processes": NUM_PROCESSES,
                }

        fit_instructs = {}
        if metadata:
            metadata_path = metadata
        for name in dataset_names:
            fit_instructs[f"fit_{name}_instruction"] = {
                "type": "TrainMLModel",
                "labels": [
                    {"binder": {"positive_class": True}}
                ],
                "dataset": name,
                "settings": [
                    {
                        "encoding": encoding,
                        "ml_method": ml_model,
                    },
                ]}
            if metadata:
                fit_instructs[f"fit_{name}_instruction"]["assessment"] = {
                    "split_strategy": "manual",
                    "split_count": 1,
                    "manual_config":{
                        "train_metadata_path": f"{metadata_path}{name}_train_metadata.csv",#need to add a path
                        "test_metadata_path": f"{metadata_path}{name}_test_metadata.csv",
                                    },
                    "reports": {
                        "models": MODEL_REPORTS_LIST,
                    }
                }
            else:
                {
                        "split_strategy": "random",
                        "split_count": 1,
                        "training_percentage": 0.7,
                        "reports": {
                            "models": MODEL_REPORTS_LIST,
                        }
                    }

            if selection:
                fit_instructs[f"fit_{name}_instruction"]["selection"] = {
                     "split_strategy": "k_fold",
                     "split_count": k_fold,
                     "reports": {
                        "models": MODEL_REPORTS_LIST
                     },
                }

            
            fit_instructs[f"fit_{name}_instruction"].update({
                "strategy": "GridSearch",
                "metrics": METRICS_LIST,
                "reports": TRAINING_REPORTS_LIST,
                "number_of_processes": NUM_PROCESSES,
                "optimization_metric": metric,
                "refit_optimal_model": refit_best,
            })

        instructs = {
            **explore_instructs,
            **fit_instructs
        }
        return instructs


# class NoAliasDumper(yaml.SafeDumper):
#     def ignore_aliases(self, data):
#         return True


# if __name__ == "__main__":

#     name = "test"
#     uid = str(uuid.uuid4())[:8]
#     with open(config.IMMUNE_ML_BASE_PATH / f"{name}_{uid}.yml", 'w+') as f:
#         yaml.dump(
#             specification,
#             f,
#             default_flow_style=False,
#             sort_keys=False,
#             Dumper=NoAliasDumper
#         )
