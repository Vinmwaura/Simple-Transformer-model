import json
import pathlib
import argparse

# Type function for argparse - a float within some predefined bounds.
def range_limited_float_type(arg):
    MIN_VAL = 0.1
    MAX_VAL = 0.9
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < MIN_VAL or f > MAX_VAL:
        raise argparse.ArgumentTypeError("Argument must be < " + str(MAX_VAL) + "and > " + str(MIN_VAL))
    return f

def main():
    parser = argparse.ArgumentParser(
        description="Generates JSON DB file.")

    parser.add_argument(
        "--txt-path",
        help="File path to .txt dataset.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-path",
        help="Destination output path for json output.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--dataset-ratio",
        help="Ratio of train dataset, for splitting dataset.",
        type=range_limited_float_type,
        default=0.8)

    args = vars(parser.parse_args())

    txt_dataset_path = args["txt_path"]  # File path to dataset.
    out_path = args["out_path"]  # Output Json file e.g ./<folder_path>/train.json.
    dataset_ratio = args["dataset_ratio"]  # Ratio of train:test dataset.

    with open(txt_dataset_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Create a set of unique characters.
    unique_characters = set(text)

    # Convert the set to a sorted list.
    unique_characters_list = sorted(unique_characters)

    # Split words by spaces, and then get index from vocab.
    word_index = []
    space_words = text.split(" ")
    for word in space_words:
        character_index = []
        for character in word:
            character_index.append(
                unique_characters_list.index(character))
        word_index.append(character_index)
        word_index.append([unique_characters_list.index(" ")])  # Add index for space character.

    train_dataset_length = round(len(word_index) * dataset_ratio)

    data_json = {
        "vocab": unique_characters_list,
        "all": word_index,
        "train": word_index[:train_dataset_length],
        "test": word_index[train_dataset_length:]}
    
    with open(out_path, "w") as json_f:
        json.dump(data_json, json_f)

if __name__ == "__main__":
    main()
