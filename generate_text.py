import click
import codecs
import pathlib
import argparse

import torch

from models.DecoderTransformer import DecoderTransformer

from utils.model_utils import load_model
from utils.token_utils import generate_text

def validate_string(value_str, allowed_chars, context_window_length):
    value_str = codecs.decode(value_str, 'unicode_escape')

    list_chars = list(value_str)
    if len(list_chars) >= context_window_length:
        raise click.BadParameter(
            f"Input length: {len(list_chars):,}, must be less than {context_window_length:,}.")

    if set(list_chars).issubset(set(allowed_chars)):
        return value_str
    else:
        raise click.BadParameter(
            f"Input contains invalid characters.")

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.1:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def main():
    parser = argparse.ArgumentParser(
        description="Generate Text.")
    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--temperature",
        help="Temperature parameter for softmax sampling.",
        type=restricted_float,
        default=1.0)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint to load from (if any).",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "--seed",
        help="Seed value.",
        type=int,
        default=None)
    parser.add_argument(
        "--print-instant",
        action='store_true',
        help="Print text all at once, no character-by-character printing.")

    args = vars(parser.parse_args())

    """
    Lower Temperature (T < 1):
    Prioritizes the most probable next token and effectively reduces
    randomness in the generated token.

    Higher Temperature (T > 1):
    Less probable tokens become more likely to be chosen, therefore more
    diversity in generated token.

    Can't be 0, OBVIOUSLY!!
    """
    temperature = args["temperature"]

    seed = args["seed"]
    print_instant = args["print_instant"]
    device = args["device"]  # Device to run model on.
    model_checkpoint = args["model_checkpoint"]  # Filepath to models saved.

    if seed is not None:
        torch.manual_seed(seed)

    classifier_status, classifier_dict = load_model(model_checkpoint)
    if not classifier_status:
        raise Exception("An error occured while loading model checkpoint!")

    vocab = classifier_dict["vocab"]
    vocab_size = len(vocab)
    hidden_dim = classifier_dict["hidden_dim"]
    embedding_dim = classifier_dict["embedding_dim"]
    num_heads = classifier_dict["num_heads"]
    num_blocks = classifier_dict["num_blocks"]
    activation_type = classifier_dict["activation_type"]
    context_window = classifier_dict["context_window"]

    model_transformer = DecoderTransformer(
        padding_idx=vocab_size,  # Padding index is length of Vocabulary.
        num_embeddings=vocab_size + 1,  #  Number of tokens including padding token.
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        out_classes=vocab_size,
        num_blocks=num_blocks,
        activation_type=activation_type)
    model_transformer.custom_load_state_dict(classifier_dict["model"])
    model_transformer = model_transformer.to(device)

    # Starting characters.
    while True:
        click.echo(f"Allowed characters tokens in sentence:\n{vocab}")
        user_input = click.prompt("Enter start of sentence?", type=str)
        try:
            validated_input = validate_string(
                value_str=user_input,
                allowed_chars=vocab,
                context_window_length=context_window)
            break
        except click.BadParameter as e:
            click.echo(e)

    generate_text(
        model=model_transformer,
        vocab=vocab,
        init_text=validated_input,
        context_window=context_window,
        print_slowly=not print_instant,
        device=device,
        temperature=temperature)

if __name__ == "__main__":
    main()
