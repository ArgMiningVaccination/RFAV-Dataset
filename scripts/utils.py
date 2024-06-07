import os


col2val = {
    "reason": 1,
    "stance": 2,
    "compressed": 2,
    "scientific_authority": 3
}

compress_values = {
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 3
}


def process_label(label, filename, line_number, component):
    if label == 'O':
        return 0
    elif (label == 'Reason' and component == "reason") or (label == "ScientificAuthority" and component == "scientific_authority"):
        return 1
    elif label.isnumeric():
        if component == 'stance':
            return int(label)
        elif component == "compressed":
            return compress_values[int(label)]
    else:
        raise ValueError(f"Invalid label '{label}' in file '{filename}', line {line_number}. Label must be 'O' or 'Reason'.")

def process_cnll_files(directory_path, component):
    # Initialize result variables
    list_of_examples = []
    list_of_tags = []

    # Iterate through each file in the specified directory
    files = [f for f in os.listdir(directory_path) if f.endswith('.cnll')]
    files.sort()
    for filename in files:
        if filename.endswith(".cnll"):  # Assuming the files have a .cnll extension
            file_path = os.path.join(directory_path, filename)

            # Initialize lists for each file
            col1_elements = []
            col2_elements = []

            # Read and process the file
            with open(file_path, 'r') as file:
                for line_number, line in enumerate(file, start=1):
                    # Split space-separated values
                    elements = line.strip().split(" ")
                    # Assuming the first column is at index 0 and the second column is at index 1
                    words_to_add = elements[0].split('/')
                    labels_to_add = [process_label(elements[col2val[component]], filename, line_number, component)] * len(words_to_add)
                    assert len(words_to_add) == len(labels_to_add)
                    col1_elements.extend(words_to_add)
                    col2_elements.extend(labels_to_add)

            # Ensure that the word count of col1_elements matches the length of col2_elements
            assert len(col1_elements) == len(col2_elements), "Error: Word count does not match list length"

            # Concatenate elements in the first column
            list_of_examples.append(col1_elements)

            # Append 0 for each "0" and 1 for each "Reason" in the second column
            list_of_tags.append(col2_elements)

    return list_of_examples, list_of_tags
