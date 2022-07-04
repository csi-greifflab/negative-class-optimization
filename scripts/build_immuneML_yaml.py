from pathlib import Path
import yaml
import logging
from docopt import docopt
import NegativeClassOptimization.config as config
import NegativeClassOptimization.immuneml_utils as immuneml_utils

logging.basicConfig(level=logging.DEBUG)

docopt_doc = """
Usage: build_immuneMl_yaml.py <data_type> <encoding> <ml_model> <optimized_metric> (-o FILE) [options]

Arguments: 
    <data_type>         [pw_wo_dupl, pairwise, 1_vs_all]
    <encoding>          [oh, onehot, kmer, w2v]
    <ml_model>          [lr, logistic-regression, svm, rf, random-forest, knn]
    <optimized_metric>  [accuracy, balanced_accuracy, confusion_matrix, f1_micro, f1_macro, f1_weighted, precision, recall, auc, log_loss]
    

Options:
    -h --help       Show help.
    --selection     Add inner cross-validation for selecting model's hyperparameters [default: False]
    --k_fold=<int>  Number of cross-validation iterations [default: 3]
    --refit_best    Refits best model on all training data [default: False]
    --add_explore   [default: False]
    -o FILE         Specify output file.
    
    
"""

class NoAliasDumper(yaml.SafeDumper):
    """
    Yaml dumper without yaml references.
    """
    def ignore_aliases(self, data):
        return True


def basic_spec_from_dataset_paths(datasets_paths, encoding, ml_model, metric, selection, k_fold, refit_best, add_explore) -> dict:
    datasets = {}
    for dataset_path in datasets_paths:
        dataset_name = dataset_path.name.split(".")[0]
        dataset_path = str(dataset_path.absolute())
        datasets[dataset_name] = {
                "format": "Generic",
                "params": {
                    "path": dataset_path,
                    "is_repertoire": False,
                    "region_type": "FULL_SEQUENCE",
                    "column_mapping": {
                        "Slide": "sequence_aas",
                    },
                    "metadata_column_mapping": {
                        "UID": "UID",
                        "Antigen": "Antigen",
                        "binder": "binder",
                    }
                },
            }
    specification = immuneml_utils.ImmuneMLSpecBuilder(datasets, encoding, ml_model, metric, selection, k_fold, refit_best, add_explore).spec
    return specification


if __name__ == "__main__":
    
    arguments = docopt(docopt_doc, help=True, version='Naval Fate 2.0')
    
    #Parameters validation
    data_types_paths = {'pw_wo_dupl':"../data/pairwise_wo_dupl",
                       'pairwise':"../data/pairwise",
                       '1_vs_all':"../data/1_vs_all"}
    try:
        folder_path = data_types_paths[arguments['<data_type>']]
        logging.info(f"Building spec for {arguments['<data_type>']} analysis.")
        datasets_paths = list(Path(folder_path).glob("*.tsv"))
    except KeyError:
        print('Error! Use -h to see valid options for data_type')
        
    encodings = {'oh': 'onehot_encoding',
                 'onehot': 'onehot_encoding',
                 'kmer': 'kmerfreq_encoding',
                 'w2v': 'word2vec_encoding'
                }
    try:
        encoding = encodings[arguments['<encoding>']]
    except KeyError:
        print('Error! Use -h to see valid options for encoding')
    
    ml_models = {'lr': 'logistic_default_model',
                 'logistic-regression': 'logistic_default_model',
                 'svm': 'svm_default_model',
                 'rf': 'randomforest_default_model',
                 'random-forest':'randomforest_default_model',
                 'knn': 'knn_default_model'}
    try:
        ml_model = ml_models[arguments['<ml_model>']]
    except KeyError:
        print('Error! Use -h to see valid options for ml_models')
    
    metrics=['accuracy','balanced_accuracy','confusion_matrix','f1_micro','f1_macro','f1_weighted','precision','recall','auc','log_loss']
    if arguments['<optimized_metric>'] in metrics:
        metric = arguments['<optimized_metric>']
    else:
        raise KeyError('Error! Use -h to see valid options for optimized_metric')

        
    optional = {'selection':False,
                'refit_best':False,
                'add_explore':False}
    for option in optional.keys():
        if arguments[f'--{option}']:
            optional[option] = True    
    
    if arguments['--k_fold']:
        k_fold = int(arguments['--k_fold'])
    else:
        k_fold = 3
    
    spec_fp = arguments['-o']
    
    specification = basic_spec_from_dataset_paths(datasets_paths, encoding, ml_model, metric, selection = optional['selection'], k_fold = k_fold, refit_best=optional['refit_best'], add_explore=optional['add_explore'])
    with open(spec_fp, 'w+') as f:
        yaml.dump(
            specification,
            f,
            default_flow_style=False,
            sort_keys=False,
            Dumper=NoAliasDumper
        )
    
    
    
    