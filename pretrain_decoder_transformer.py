import os
import json
import math
import random
import pathlib
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DecoderTransformer import DecoderTransformer

from dataset_loader.character_dataset import CharacterEmbeddingDataset

from utils.model_utils import (
    save_model,
    load_model)
from utils.token_utils import generate_text
from utils.classification_utils import compute_classification

def restricted_float(x):
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError("%r not a floating-point literal" % (x,))

    if x < 0.1:
        raise argparse.ArgumentTypeError("%r not in range > 0.1"%(x,))
    return x

def classification_metric(
        model,
        dataloader,
        logging,
        device):
    # Compute Classification Metric.
    accuracy = compute_classification(
        model=model,
        dataloader=dataloader,
        device=device)
    message = "Test Accuracy: {:.5f}".format(accuracy)
    logging.info(message)

def checkpoint_model(
        data_dict,
        out_dir,
        model,
        model_optim,
        logging):
    global_steps = data_dict["global_steps"]

    # Save model that has achieved max TPR with the dataset.
    model_dict = {
        **data_dict,
        "model": model.state_dict(),
        "optimizer": model_optim.state_dict()}

    save_status = save_model(
        model_dict=model_dict,
        dest_path=out_dir,
        file_name=f"{global_steps}_model.pt",
        logging=logging.info)
    if save_status is True:
        logging.info("Successfully saved model.")
    else:
        logging.info("Error occured saving model.")

def main():
    project_name = "Decoder-Only Transformer"

    parser = argparse.ArgumentParser(
        description=f"Pre-train {project_name} model.")

    parser.add_argument(
        "--device",
        help="Which hardware device will model run on.",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--use-activation-checkpoint",
        action='store_true',
        help="Use Activation Checkpointing; trade-off memory footprint and compute.")
    parser.add_argument(
        "--dataset-path",
        help="File path to json dataset file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--test-model",
        action='store_true',
        help="Test model's accuracy using testing dataset during checkpointing.")
    parser.add_argument(
        "--tr-batch-size",
        help="Batch size of training dataset.",
        type=int,
        default=64)
    parser.add_argument(
        "--tst-batch-size",
        help="Batch size of testing dataset.",
        type=int,
        default=128)
    parser.add_argument(
        "--temperature",
        help="Temperature parameter for softmax sampling.",
        type=restricted_float,
        default=1.0)
    parser.add_argument(
        "--checkpoint-steps",
        help="Steps for checkpointing and/or testing model.",
        type=int,
        default=1_000)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint to load from (if any).",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "--resume-training",
        action='store_true',
        help="Resuming training using previously stored parameters.")
    parser.add_argument(
        "--load-optim",
        action='store_true',
        help="Load model's optimizer's weights and parameters, if loading model.")
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to JSON config file.",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="Folder path of output directory.",
        required=True)

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

    device = args["device"]  # Device to run model on.
    use_activation_checkpoint = args["use_activation_checkpoint"]
    load_optim = args["load_optim"]  # Reload saved optimizer weights.
    resume_training = args["resume_training"]
    dataset_path = args["dataset_path"]  # JSON file path (*.json).
    tr_batch_size = args["tr_batch_size"]  # Batch size of training dataset.
    tst_batch_size = args["tst_batch_size"]  # Batch size of testing dataset.
    model_checkpoint = args["model_checkpoint"]  # Filepath to models saved.
    checkpoint_steps = args["checkpoint_steps"]  # Steps to checkpoint model.
    test_model = args["test_model"]  # Test model flag.
    out_dir = args["out_dir"]  # Destination path for model.
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        raise e

    config_json = args["config_path"]  # Load and Parse config JSON.
    with open(config_json, 'r') as json_file:
        json_data = json_file.read()
    config_dict = json.loads(json_data)

    # Model Params (From config file).
    model_lr = config_dict["model_lr"]
    num_heads = config_dict["num_heads"]
    num_blocks = config_dict["num_blocks"]
    hidden_dim = config_dict["hidden_dim"]
    embedding_dim = config_dict["embedding_dim"]
    context_window = config_dict["context_window"]
    activation_type = config_dict["activation_type"]

    # Max global steps.
    max_global_steps = config_dict["max_global_steps"]

    # Load JSON dataset.
    with open(dataset_path, "r") as json_f:
        json_dataset = json.load(json_f)

    # Vocabulary / Vocabulary size of NLP dataset.
    vocab = json_dataset["vocab"]
    vocab_size = len(vocab)

    # Train Dataset.
    train_dataset = CharacterEmbeddingDataset(
        dataset=json_dataset["train"],
        padding_idx=vocab_size,
        context_window=context_window)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=tr_batch_size,
        num_workers=4,
        shuffle=True)

    # Test Dataset.
    test_dataset = CharacterEmbeddingDataset(
        dataset=json_dataset["test"],
        padding_idx=vocab_size,
        context_window=context_window)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=tst_batch_size,
        num_workers=4,
        shuffle=True)

    model_transformer = DecoderTransformer(
        padding_idx=vocab_size,  # Padding index is length of Vocabulary.
        num_embeddings=vocab_size + 1,  #  Number of tokens including padding token.
        hidden_dim=hidden_dim,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        out_classes=vocab_size,
        num_blocks=num_blocks,
        activation_type=activation_type,
        use_activation_checkpoint=use_activation_checkpoint)

    global_steps = 0

    # Load Transformer Model checkpoints if any.
    if model_checkpoint is not None:
        logging.info("Loading Model...")
        classifier_status, classifier_dict = load_model(model_checkpoint)
        if not classifier_status:
            raise Exception("An error occured while loading model checkpoint!")

        model_transformer.custom_load_state_dict(classifier_dict["model"])
        model_transformer = model_transformer.to(device)

        model_transformer_optim = torch.optim.Adam(
            model_transformer.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

        # Load Optimizer params and global steps params.
        if load_optim:
            logging.info("Resuming Training using saved optimizer weights and global_steps...")
            model_transformer_optim.load_state_dict(classifier_dict["optimizer"])

        # Use previous global_steps.
        if resume_training:
            global_steps = classifier_dict["global_steps"]

    else:
        model_transformer = model_transformer.to(device)

        model_transformer_optim = torch.optim.Adam(
            model_transformer.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

    # Model Params size.
    model_params_size = sum(param.numel() for param in model_transformer.parameters())

    # Cross Entropy Loss, ignores pad token.
    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab_size)

    # Update learning rate in case of changes.
    for model_transformer_optim_ in model_transformer_optim.param_groups:
        model_transformer_optim_["lr"] = model_lr

    # https://pytorch.org/docs/stable/amp.html
    scaler = torch.cuda.amp.GradScaler()

    # Log file path.
    log_path = os.path.join(
        out_dir,
        f"{project_name}.log")

    # Logs Info to parent directory.
    logging.basicConfig(
        # filename=log_path,
        format="%(asctime)s %(message)s",
        encoding='utf-8',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ],
        level=logging.DEBUG,
        force=True)

    logging.info(f"{project_name}")
    logging.info(f"Output Directory: {out_dir}")
    logging.info("#" * 100)
    logging.info("Model Parameters.")
    logging.info(f"Model Param size: {model_params_size:,}")
    logging.info(f"Number of heads: {num_heads:,}")
    logging.info(f"Context window: {context_window:,}")
    logging.info(f"Number of blocks: {num_blocks:,}")
    logging.info(f"Embedding Dimension: {embedding_dim:,}")
    logging.info(f"Vocab Size: {vocab_size:,}")
    logging.info(f"Activation Type: {activation_type}")
    logging.info(f"Using activation checkpoint: {use_activation_checkpoint}")
    logging.info(f"Model Learning Rate: {model_transformer_optim.param_groups[0]['lr']:,}")
    logging.info("#" * 100)
    logging.info("Training Parameters.")
    logging.info(f"Step: {global_steps:,}")
    logging.info(f"Max Global step: {max_global_steps:,}")
    logging.info(f"Checkpoint Steps: {checkpoint_steps:,}")
    logging.info("#" * 100)
    logging.info(f"Total Train Dataset: {len(train_dataset):,}")
    logging.info(f"Total Test Dataset (Excluded from Training): {len(test_dataset):,}")
    logging.info("#" * 100)
    logging.info(f"Sampling Parameters.")
    logging.info(f"Temperature: {temperature:,}")
    logging.info("#" * 100)

    # Training starts here.
    stop_training = False
    while not stop_training:
        total_loss = 0
        training_count = 0

        for index, (in_seq, target_seq) in enumerate(train_dataloader):
            # Checkpoint and test model.
            if global_steps % checkpoint_steps == 0:
                if test_model:
                    classification_metric(
                        model=model_transformer,
                        dataloader=test_dataloader,
                        logging=logging,
                        device=device)

                # Checkpoint model weights.
                checkpoint_model(
                    data_dict={
                        "vocab": vocab,
                        "num_heads": num_heads,
                        "num_blocks": num_blocks,
                        "hidden_dim": hidden_dim,
                        "embedding_dim": embedding_dim,
                        "context_window": context_window,
                        "activation_type": activation_type,
                        "global_steps": global_steps},
                    out_dir=out_dir,
                    model=model_transformer,
                    model_optim=model_transformer_optim,
                    logging=logging)

                # Test model generative capabilities.

                # Randomly pick one character from vocab to test model.
                init_char = random.choice(vocab)
                generate_text(
                    model=model_transformer,
                    vocab=vocab,
                    init_text=init_char,
                    context_window=context_window,
                    logging=logging.info,
                    device=device,
                    temperature=1)

            # Training Data.
            in_seq = in_seq.to(device)  # (N, Seq)
            target_seq = target_seq.to(device)  # (N, Seq)

            model_transformer.train(mode=True)

            # Runs the forward pass under ``autocast``.
            with torch.autocast(device_type=device, dtype=torch.float16):
                out_seq = model_transformer(in_seq)  # (N, Seq, Class)

                target_seq_flat = target_seq.flatten()  # (N*Seq)

                N, Seq, C = out_seq.shape            
                out_seq_flat = out_seq.view(N*Seq, C)  # (N*Seq, Class)

                loss = ce_loss(
                    out_seq_flat,
                    target_seq_flat)

                if torch.isnan(loss):
                    raise Exception("NaN encountered during training.")

            # Scales loss. Calls ``backward()`` on scaled loss to create scaled gradients.
            scaler.scale(loss).backward()

            scaler.step(model_transformer_optim)

            # Updates the scale for next iteration.
            scaler.update()

            model_transformer_optim.zero_grad()

            total_loss = total_loss + loss.item()
            training_count = training_count + 1

            temp_avg_loss = total_loss / training_count

            correct = torch.eq(
                torch.argmax(out_seq_flat, dim=1),
                target_seq_flat
            ).long().sum().item()

            message = "Cum. Steps: {:,} | Steps: {:,} / {:,} | Loss: {:,.5f} | Correct: {:,} | Total: {:,}".format(
                global_steps + 1,
                index + 1,
                len(train_dataloader),
                temp_avg_loss,
                correct,
                N*Seq)

            logging.info(message)

            global_steps = global_steps + 1

            # Stop training when stopping criteria is met.
            if global_steps >= max_global_steps:
                stop_training = True
                break

    # Final Test, Checkpoint, and Token generation capability.
    if test_model:
        classification_metric(
            model=model_transformer,
            dataloader=test_dataloader,
            logging=logging,
            device=device)

    # Checkpoint model weights.
    checkpoint_model(
        data_dict={
            "vocab": vocab,
            "num_heads": num_heads,
            "num_blocks": num_blocks,
            "hidden_dim": hidden_dim,
            "embedding_dim": embedding_dim,
            "context_window": context_window,
            "activation_type": activation_type,
            "global_steps": global_steps},
        out_dir=out_dir,
        model=model_transformer,
        model_optim=model_transformer_optim,
        logging=logging)

    # Test model generative capabilities.
    init_char = random.choice(vocab)
    generate_text(
        model=model_transformer,
        vocab=vocab,
        init_text=init_char,
        context_window=context_window,
        logging=logging.info,
        device=device,
        temperature=1)

if __name__ == "__main__":
    main()
