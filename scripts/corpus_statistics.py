from utils import process_cnll_files
import argparse

parser = argparse.ArgumentParser(description="Train models for identifying reasons for and against vaccination")
parser.add_argument('--input_folder', type=str, default="../corpus", help="Path to the folder containing the .cnll files")
parser.add_argument('--is_split', type=bool, default=False, help="Whether the input folder contains train, dev, and test folders")
args = parser.parse_args()

DIR = args.input_folder
COMPONENT = "stance"


if args.is_split:
    examples_tr, tags_tr = process_cnll_files(f"{DIR}/train", COMPONENT)
    examples_dv, tags_dv = process_cnll_files(f"{DIR}/dev", COMPONENT)
    examples_tt, tags_tt = process_cnll_files(f"{DIR}/test", COMPONENT)
    tags = tags_tr + tags_dv + tags_tt

examples, tags = process_cnll_files(DIR, COMPONENT)
total_tags = sum([len(tag) for tag in tags])
positive_tags = sum([len([t for t in tag if t != 0]) for tag in tags])
examples_without_label = sum([1 if len([t for t in tag if t > 0]) == 0 else 0 for tag in tags])


print(total_tags, positive_tags, positive_tags/total_tags)

print("Total words: ", total_tags)
print("Labelled words: ", positive_tags)
print("Proportion of labeled words: ", positive_tags/total_tags)
print("Examples without label: ", examples_without_label)

if COMPONENT == "stance":
    for i in range(1, 6):
        print(f"{i}: {sum([len([t for t in tag if t == i]) for tag in tags]) / (sum([len([t for t in tag if t > 0]) for tag in tags]))}")
    
    print("Proportion against: ", sum([len([t for t in tag if t == 1 or t == 2]) for tag in tags]) / (sum([len([t for t in tag if t > 0]) for tag in tags])))
    print("Proportion favor: ", sum([len([t for t in tag if t == 4 or t == 5]) for tag in tags]) / (sum([len([t for t in tag if t > 0]) for tag in tags])))
