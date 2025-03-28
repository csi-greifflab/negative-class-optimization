library(jsonlite)
library(innsight)
library(parallel)

#function that creates empty SN10_list model
get_SN10_list_model <- function(weights_json_path) {
    pytorch_weights <- fromJSON(paste0('/bucket/KondrashovU/aygul/deep_lift_data/model_weights/',weights_json_path))
    SN10_list <- list( 
      input_dim = c(220),
      input_nodes = 1,
      output_nodes = 2,
      output_names = c("y"),
      layers = list(
        list(
          type = "Dense",
          input_layers = 0,
          output_layers = 2,
          weight = pytorch_weights$l1,
          bias = c(pytorch_weights$l1b),
          activation_name = "relu"
        ),
        list(
          type = "Dense",
          input_layers = 1,
          output_layers = -1,
          weight = pytorch_weights$l2,
          bias = c(pytorch_weights$l2b),
          activation_name = "sigmoid"
        )
      )
    )
    return(SN10_list)
}

#one-hot encoding function
one_hot_encode_sequence <- function(sequence) {
  # Define the 20 standard amino acids
  amino_acids <- c("A", "C", "D", "E", "F", "G", "H", "I", "K", "L", 
                   "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y")
  
  # Initialize an empty matrix to store the one-hot encoding
  encoding <- matrix(0, nrow = nchar(sequence), ncol = length(amino_acids))
  
  for (i in 1:nchar(sequence)) {
    aa <- substr(sequence, i, i)
    if (aa %in% amino_acids) {
      encoding[i, match(aa, amino_acids)] <- 1
    } else {
      stop(paste("Invalid amino acid:", aa))
    }
  }
    
  flattened <- c(t(encoding))
  flattened <- matrix(flattened, nrow=1)
  return(flattened)
}

set.seed(42)
shuffle_sequence <- function(sequence, n_shuffles=10) {
    shuffled_sequences <- list()
    for (i in 1:n_shuffles){ 
        shuffled_sequence <- paste(sample(strsplit(sequence, "")[[1]]), collapse = "")
        shuffled_sequences <- append(shuffled_sequences, shuffled_sequence)
    }
  return(shuffled_sequences)
}

get_deeplift_attr <- function(seq, converter,rule_name="reveal_cancel", n_shuffles=10) {
    seq_oh <- one_hot_encode_sequence(seq)
    shuffles <- shuffle_sequence(seq,  n_shuffles)
    attrs <- matrix(nrow=0, ncol=220)
    for (baseline in shuffles){
        baseline <- one_hot_encode_sequence(baseline)
        deeplift <- DeepLift$new(
            converter, 
            seq_oh,
            x_ref = baseline,
            rule_name = rule_name #reveal_cancel/rescale
        ) 
        attr <- get_result(deeplift)
        attrs <- rbind(attrs, attr)
    }
    mean_attr <- as.vector(colMeans(attrs))
    return(mean_attr)
}

get_attr_for_file <- function(weights_json)
{
    tryCatch({
        #logger$info(paste("Processing item:", weights_json))
        SN10_list <- get_SN10_list_model(weights_json)
        converter <- Converter$new(SN10_list)
        #get test dataset
        file_name <- sub("__weights.json$", "", weights_json)
        data_file_name <- paste0(file_name, '__test.tsv')
        data_path <- sprintf('/bucket/KondrashovU/aygul/deep_lift_data/test_datasets/%s', data_file_name)
        df <- read.delim(data_path)
        #calculate attributions
        attr_dict <- list()
        for (i in 1:nrow(df)){
            seq <- df$Slide[i]
            attr <- get_deeplift_attr(seq, converter)
            attr_dict[[seq]] <- attr
            }
        jsonlite::write_json(attr_dict, sprintf("/flash/KondrashovU/aygul/%s__attr.json", file_name))
    },
    error = function(e) {
        message(paste("Error processing file:", weights_json))
        message("Error message:", e$message)
        print('error')
  })
             
}

models <- list.files('/bucket/KondrashovU/aygul/deep_lift_data/model_weights')
cl <- makeCluster(detectCores())

clusterEvalQ(cl, {
    library(jsonlite)
    library(innsight)
    library(parallel)
})

clusterExport(cl, c("get_SN10_list_model", "one_hot_encode_sequence", "shuffle_sequence", "get_deeplift_attr", "get_attr_for_file"))

res <-parLapply(cl, models, get_attr_for_file)

stopCluster(cl)