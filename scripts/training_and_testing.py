from transformers import AutoTokenizer, Trainer, TrainingArguments, DataCollatorForTokenClassification, EvalPrediction
from datasets import Dataset, DatasetDict
from sklearn import metrics
import numpy as np
import evaluate
# import os
from utils import process_cnll_files
from transformers import AutoModelForTokenClassification
import argparse


parser = argparse.ArgumentParser(description="Train models for identifying reasons for and against vaccination")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--batchsize', type=int, default=16, help="Batch size for training")
parser.add_argument('--component', type=str, default="reason", help="Component to be trained: reason, stance, compressed or scientific_authority", choices=["reason", "stance", "compressed", "scientific_authority"])
parser.add_argument('--language', type=str, default="english", help="Language of the corpus to be used for training", choices=["english", "spanish", "both"])
parser.add_argument('--input_folder', type=str, default="../corpus", help="Path to the folder containing the .cnll files")
parser.add_argument('--lr', type=float, default=2e-5, help="Learning rate for training")
parser.add_argument('--epochs', type=int, default=20, help="Number of epochs for training")
args = parser.parse_args()

checkpoint = args.modelname # Specify the pre-trained model name
number_labels = 2 # Number of output labels for the classification task
if args.component == "stance":
    number_labels = 6
elif args.component == "compressed":
    number_labels = 4

# Define parameters:

BATCH_SIZE = args.batchsize
EPOCHS = args.epochs
REP=0
MODEL_NAME = args.modelname
MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL = 512 if MODEL_NAME == "roberta-base" else 4096
COMPONENT = args.component
#FOLDS=3

tokenizer = AutoTokenizer.from_pretrained(checkpoint, add_prefix_space=True)
MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL = tokenizer.model_max_length
# if "spanberta" in MODEL_NAME:
#     MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL = 512

languages = []
if args.language == "both" or args.language == "english":
    languages.append("english")
elif args.language == "both" or args.language == "spanish":
    languages.append("spanish")

vaccine_phrases_dev, vaccine_labels_dev, vaccine_phrases_train, vaccine_labels_train, vaccine_phrases_test, vaccine_labels_test = [], [], [], [], [], []
for lang in languages:
    #dev:
    directory_path_dev = f'{args.input_folder}/{lang}/dev'
    vac_phrases_dev, vac_labels_dev = process_cnll_files(directory_path_dev, COMPONENT)

    #train:
    directory_path_train = f'{args.input_folder}/{lang}/train'
    vac_phrases_train, vac_labels_train = process_cnll_files(directory_path_train, COMPONENT)

    #test:
    directory_path_test = f'{args.input_folder}/{lang}/test'
    vac_phrases_test, vac_labels_test = process_cnll_files(directory_path_test, COMPONENT)

    vaccine_phrases_dev.extend(vac_phrases_dev)
    vaccine_labels_dev.extend(vac_labels_dev)
    vaccine_phrases_train.extend(vac_phrases_train)
    vaccine_labels_train.extend(vac_labels_train)
    vaccine_phrases_test.extend(vac_phrases_test)
    vaccine_labels_test.extend(vac_labels_test)

def create_example_labels_dict_of_lists(example_lists, labels_lists):
    """
    Create a dictionary with "example" and "labels" as keys, where each value is a list.

    Parameters:
    - example_lists (list): List of example (strings).
    - labels_lists (list of lists): List of lists of corresponding labels.

    Returns:
    - dict: Dictionary with "example" and "labels" as keys, where each value is a list.
    """
    if len(example_lists) != len(labels_lists):
        raise ValueError("Input lists must have the same length.")

    result_dict = {"example": [], "labels": []}

    for example, labels in zip(example_lists, labels_lists):
        result_dict["example"].append(example)
        result_dict["labels"].append(labels)

    return result_dict

def partition_phrases_and_labels(strings, labels, tokenizer, max_length):
    """
    Splits a list of strings and corresponding labels into partitions, each containing phrases
    that are shorter than or equal to the specified maximum length. Phrases are identified by
    strings ending with punctuation marks such as '.', '!' or '?'.

    Parameters:
    - strings (List[str]): List of input strings representing phrases.
    - labels (List[str]): List of corresponding labels for each input string.
    - max_length (int): Maximum length of each partition.

    Returns:
    Tuple[List[List[str]], List[List[str]]]: A tuple containing lists of partitions and labels,
    where each partition represents a list of phrases and each label partition corresponds to
    the labels of the phrases.
    """

    current_max_partition = 0
    index = 0
    max_partition_length = 0

    # Find the largest phrase that is shorter than max_lenght
    for word in strings:
        # Check if the current partition length is less than the specified maximum length
        current_max_partition += (len(tokenizer(word)["input_ids"]) - 2)
        if current_max_partition < (max_length - 2):
          index += 1

          # Check if the current string marks the end of a phrase
          if word.endswith(".") or word.endswith("?") or word.endswith("!"):   #TODO: check if there could be more ways of ending a phrase.
              # Update the maximum partition length
              max_partition_length = index
        else:
             # Break the loop if the maximum partition length is reached
             break


    # Determine the length of the current partition
    if max_partition_length > 0:
      partition_lenght = max_partition_length
    else:
      # If no phrase shorter than the maximum length is found, use the specified maximum length
      partition_lenght = index if index != 0 else max_length

    current_partition = strings[:partition_lenght]
    current_labels = labels[:partition_lenght]

    assert len(tokenizer(current_partition, is_split_into_words=True)["input_ids"]) <= max_length, "Error: The tokenized examples exceed the maximum supported sequence length."

    # Extract the current partition of strings and labels
    remaining_strings = strings[partition_lenght:]
    remaining_labels = labels[partition_lenght:]

    assert partition_lenght != 0, "Error: The partition length is 0."

    # Recursively call the function for the remaining strings and labels
    if len(remaining_strings) > 0 and len(remaining_labels) > 0:
        remaining_strings_partition, remaining_labels_partition = partition_phrases_and_labels(
            remaining_strings, remaining_labels, tokenizer, max_length
        )
        # Concatenate the current partition with the partition from the recursive call
        # Concatenate the current labels with the labels from the recursive call
        return [current_partition] + remaining_strings_partition, [current_labels] + remaining_labels_partition

    # Return the current partition and labels when there are no more remaining strings
    return [current_partition], [current_labels]

def split_examples(non_tokenized_examples, non_tokenized_examples_labels, tokenizer, initial_max_length):
    """
    Split the examples in chunks that are smaller than initial_max_length.

    Parameters:
    - non_tokenized_examples: List of non-tokenized examples.
    - non_tokenized_examples_labels: List of labels corresponding to the non-tokenized examples.
    - max_length: Int indicating the maximum sequence length supported by the model.

    Returns:
    - Tokenized examples that do not exeed the maximum supported sequence lenght.
    """

    all_partitioned_non_tokenized_examples, all_partitioned_labels = [], []
    for example, labels in zip(non_tokenized_examples, non_tokenized_examples_labels):
        partitioned_element_i, partitioned_element_i_labels = partition_phrases_and_labels(example, labels, tokenizer, initial_max_length)
        for wrds, label in zip(partitioned_element_i, partitioned_element_i_labels):
            assert len(wrds) == len(label)
            assert len(tokenizer(wrds, is_split_into_words=True)["input_ids"]) <= initial_max_length, "Error: The tokenized examples exceed the maximum supported sequence length."
        all_partitioned_non_tokenized_examples.extend(partitioned_element_i)
        all_partitioned_labels.extend(partitioned_element_i_labels)
    return all_partitioned_non_tokenized_examples, all_partitioned_labels

# Create dev, train and test dictonaries with "example" and "labels" as keys, where each value is a list.

split_input_train, split_labels_train = split_examples(vaccine_phrases_train, vaccine_labels_train, tokenizer, MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL)
split_input_test, split_labels_test = split_examples(vaccine_phrases_test, vaccine_labels_test, tokenizer, MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL)
split_input_dev, split_labels_dev = split_examples(vaccine_phrases_dev, vaccine_labels_dev, tokenizer, MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL)

dicts_list_dev = create_example_labels_dict_of_lists(split_input_dev, split_labels_dev)
dicts_list_train = create_example_labels_dict_of_lists(split_input_train, split_labels_train)
dicts_list_test = create_example_labels_dict_of_lists(split_input_test, split_labels_test)

# Create datasets (train, dev and test) from dictionaries.
dataset_dev = Dataset.from_dict(dicts_list_dev)
dataset_train = Dataset.from_dict(dicts_list_train)
dataset_test = Dataset.from_dict(dicts_list_test)

    # Tokenize ():


dataset_dic_train_dev_test = DatasetDict({
        'train': dataset_train,
        'validation': dataset_dev,
        'test': dataset_test
    })



def tokenize_and_align_labels(examples):
    non_tokenized_examples = examples["example"]
    non_tokenized_examples_labels = examples["labels"]
    
    # Alingn labels:
    # for short_example, short_labels in zip(split_input, split_labels):
    tokenized_examples = tokenizer(non_tokenized_examples, is_split_into_words=True, padding="max_length", max_length=MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL, return_tensors="pt")
    for exmpl in tokenized_examples["input_ids"]:
        assert len(exmpl) <= MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL, "Error: The tokenized examples exceed the maximum supported sequence length."

    labels = []
    for i, label in enumerate(non_tokenized_examples_labels):
        word_ids = tokenized_examples.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_examples["labels"] = labels

    return tokenized_examples


tokenized_dataset_dic_train_dev_test = dataset_dic_train_dev_test.map(tokenize_and_align_labels, batched=True)

#Initializing a data collator
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")

def compute_metrics_f1(p: EvalPrediction):
    preds = p.predictions.argmax(-1)
    labels = p.label_ids

    true_labels = [[str(l) for l in label if l != -100] for label in labels]
    true_predictions = [
        [str(p) for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    all_true_labels = [l for label in true_labels for l in label] #fixme: this could be replaced by a flatten.
    all_true_preds = [p for preed in true_predictions for p in preed] #fixme: this could be replaced by a flatten.

    avrge = "macro" if (COMPONENT == 'stance' or COMPONENT == 'compressed') else 'binary'
    lbls = ['0', '1', '2', '3', '4', '5'] if COMPONENT == "stance" else (['0', '1', '2', '3'] if COMPONENT == "compressed" else ['0', '1'])
    f1_all = metrics.f1_score(all_true_labels, all_true_preds, average=None, labels=lbls)
    precision_all = metrics.precision_score(all_true_labels, all_true_preds, average=None, labels=lbls)
    recall_all = metrics.recall_score(all_true_labels, all_true_preds, average=None, labels=lbls)


    f1 = metrics.f1_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    acc = metrics.accuracy_score(all_true_labels, all_true_preds)

    recall = metrics.recall_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    precision = metrics.precision_score(all_true_labels, all_true_preds, average=avrge, pos_label='1')

    confusion_matrix = metrics.confusion_matrix(all_true_labels, all_true_preds)

    ans = {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': str(confusion_matrix),
    }

    ans['f1_all'] = str(f1_all)
    ans['precision_all'] = str(precision_all)
    ans['recall_all'] = str(recall_all)

    return ans

# Before training your model, create a map of the expected ids to their labels with id2label and label2id:
id2label = {
    0: "0",
    1: "1",
    2: "2",
    3: "3",
    4: "4",
    5: "5"
}
label2id = {
    "O": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5
}

# Load the model and specify the number of expected labels.
model = AutoModelForTokenClassification.from_pretrained(checkpoint, num_labels=number_labels, label2id=label2id)

results_suffix = MODEL_NAME.split("/")[-1] + "_" + COMPONENT
# Define the training hyperparameters in TrainingArguments. The only required parameter is output_dir which specifies where to save the model.
training_args = TrainingArguments(
    output_dir=f"results_{results_suffix}/",
    evaluation_strategy="steps",
    save_steps = 500,
    eval_steps = 500,
    save_total_limit=2,

    learning_rate=args.lr,
    per_device_train_batch_size= BATCH_SIZE,
    per_device_eval_batch_size= BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    eval_accumulation_steps = 512, #fixme: esto hace que funciona más lento pero permite que el sistema no se quede sin ram.
    report_to="none",
    load_best_model_at_end=True,
)

# Pass the training arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset_dic_train_dev_test["train"],
    eval_dataset=tokenized_dataset_dic_train_dev_test["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics_f1,
)

# Call train() to finetune the model.
trainer.train()

results = trainer.predict(tokenized_dataset_dic_train_dev_test["test"])

#Save results to results_test:

model_name_id = MODEL_NAME.split("/")[-1]

suffix = ""
if "augmented" in args.input_folder:
    suffix = "_augmented"
    if "extra" in args.input_folder:
        suffix += "_extra"

results_path_test = f"results_test_{model_name_id}-{COMPONENT}-{args.lr}{suffix}.txt"

# trainer.model.save_pretrained(f"results/{MODEL_NAME}-{COMPONENT}")
model.push_to_hub(f"argmining-vaccines/{model_name_id}-{COMPONENT}-{args.lr}{suffix}")
# Open the file in write mode
with open(results_path_test, 'w') as file:
    # Write the results to the file
    for mtric in results.metrics:
        file.write(str(mtric) + ':' + str(results.metrics[mtric]) + '\n')
