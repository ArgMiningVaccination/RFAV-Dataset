from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification, EvalPrediction
import argparse
from sklearn import metrics
from datasets import Dataset, DatasetDict

parser = argparse.ArgumentParser(description="Label text with reason, stance, and scientific authority labels.")
parser.add_argument('--modelname', type=str, default="roberta-base", help="Name of the language model to be downloaded from huggingface")
parser.add_argument('--component', type=str, default="reason", help="Component to be trained: reason, stance, compressed or scientific_authority", choices=["reason", "stance", "compressed", "scientific_authority"])
parser.add_argument('--example', type=str, help="Example to be labeled")
parser.add_argument('--from_file', type=bool, default=False, help="True if the example is a file path, False if the example is a string")
parser.add_argument('--output', type=str, default="", help="Output file to save the labeled example. If empty will print to the console.")
parser.add_argument('--is_extension', type=int, default=0, choices = [0,1,2], help="0 for models trained only with human annotations, 1 for models augmented using GPT4 annotations and 2 for models augmented with GPT4 and GPT3.5 annotations")
args = parser.parse_args()

MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL = 512

component_columns = {
    "reason": 1,
    "stance": 2,
    "compressed": 2,
    "scientific_authority": 3
}


tokenizer_prefix = {
    "xlm-roberta-base": "FacebookAI/xlm-roberta-base",
    "spanbert-base-cased": "SpanBERT/spanbert-base-cased",
    "longformer-base-4096": "allenai/longformer-base-4096",
    "roberta-base": "roberta-base",
    "bert-base-spanish-wwm-cased": "dccuchile/bert-base-spanish-wwm-cased",
}



tokenizer = AutoTokenizer.from_pretrained(tokenizer_prefix[args.modelname], add_prefix_space=True)

number_labels = 2 # Number of output labels for the classification task
if args.component == "stance":
    number_labels = 6
elif args.component == "compressed":
    number_labels = 4

model_suffix = "" if args.is_extension == 0 else "_augmented" if args.is_extension == 1 else "_augmented_extra"
model = AutoModelForTokenClassification.from_pretrained(f"argmining-vaccines/{args.modelname}-{args.component}{model_suffix}", num_labels=number_labels)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
training_args = TrainingArguments(
    output_dir="results/"
)

# Pass the training arguments to Trainer along with the model, dataset, tokenizer, data collator, and compute_metrics function.
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

if args.from_file:
    with open(args.example, "r") as f:
        lines = f.readlines()
    sentence = " ".join([line.split()[0] for line in lines if line.strip() != ""])
else:
    sentence = args.example

tokenized_sentence = tokenizer(sentence, padding="max_length", max_length=MAXIMUM_INPUT_SEQUENCE_LENGTH_SUPPORTED_BY_THE_MODEL, return_tensors="pt", truncation=True)

test_dataset = Dataset.from_dict(tokenized_sentence)

results = trainer.predict(test_dataset)

lbls = []
words = []
prev_wrd = None
for lbl, wrd in zip(results.predictions.argmax(-1)[0], tokenized_sentence.word_ids()):
    if wrd is not None and wrd != prev_wrd:
        lbls.append(lbl)
        word = tokenized_sentence.word_to_chars(wrd)
        word_str = sentence[word.start:word.end]
        words.append(word_str)
    prev_wrd = wrd

if args.output != "":
    with open(args.output, "w") as f:
        for word, lbl in zip(words, lbls):
            line_to_write = f"{word}\t{lbl}\n"
            f.write(line_to_write)

for pair in zip(words, lbls):
    word = pair[0]
    lbl = pair[1]
    line_to_print = f"{word}: {lbl}"
    print(line_to_print)