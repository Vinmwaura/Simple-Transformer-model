import json
import pathlib
import argparse


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

    args = vars(parser.parse_args())

    txt_dataset_path = args["txt_path"]  # File path to dataset.
    out_path = args["out_path"]  # Output Json file e.g ./<folder_path>/train.json.

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
        
    data_json = {
        "vocab": unique_characters_list,
        "all": word_index}
    
    with open(out_path, "w") as json_f:
        json.dump(data_json, json_f)

if __name__ == "__main__":
    main()
