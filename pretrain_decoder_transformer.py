import os
import json
import math
import pathlib
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.DecoderTransformer import DecoderTransformer

from dataset_loader.character_dataset import CharacterEmbeddingDataset

from utils.classification_utils import compute_classification
from utils.model_utils import (
    save_model,
    load_model)

def main():
    project_name = "Decoder Transformer"

    parser = argparse.ArgumentParser(
        description="Pre-train Conditional Classifier model.")
    
    parser.add_argument(
        "--device",
        help="Which hardware device will model run on",
        choices=['cpu', 'cuda'],
        type=str,
        default="cpu")
    parser.add_argument(
        "--dataset-path",
        help="File path to json dataset file",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--load-optim",
        help="Load saved optim parameters with model",
        type=bool,
        default=False)
    parser.add_argument(
        "--tr-batch-size",
        help="Batch size for train dataset",
        type=int,
        default=64)
    parser.add_argument(
        "--tst-batch-size",
        help="Batch size for test dataset",
        type=int,
        default=128)
    parser.add_argument(
        "--checkpoint-steps",
        help="Step to checkpoint and test model",
        type=int,
        default=1_000)
    parser.add_argument(
        "--model-checkpoint",
        help="File path to model checkpoint being trained",
        required=False,
        default=None,
        type=pathlib.Path)
    parser.add_argument(
        "-c",
        "--config-path",
        help="File path to load json config file",
        required=True,
        type=pathlib.Path)
    parser.add_argument(
        "--out-dir",
        help="Folder path for output directory",
        required=True)

    args = vars(parser.parse_args())

    device = args["device"]  # Device to run model on.
    load_optim = args["load_optim"]  # Load saved optim parameters.
    dataset_path = args["dataset_path"]  # JSON file path (*.json).
    tr_batch_size = args["tr_batch_size"]  # Batch size for train dataset.
    tst_batch_size = args["tst_batch_size"]  # Batch size for test dataset.
    model_checkpoint = args["model_checkpoint"]  # Filepath to models saved.
    checkpoint_steps = args["checkpoint_steps"]  # Steps to checkpoint model.

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
    embedding_dim = config_dict["embedding_dim"]
    context_window = config_dict["context_window"]
    activation_type = config_dict["activation_type"]

    # Max global steps.
    max_global_steps = config_dict["max_global_steps"]

    # Load JSON dataset.
    with open(dataset_path, "r") as json_f:
        json_dataset = json.load(json_f)

    # Vocabulary size of NLP dataset.
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
        padding_idx=vocab_size,
        num_embeddings=vocab_size+1,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        out_classes=vocab_size,
        num_blocks=num_blocks,
        activation_type=activation_type)

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

        if load_optim:
            model_transformer_optim.load_state_dict(classifier_dict["optimizer"])

        global_steps = classifier_dict["global_steps"]
    else:
        model_transformer = model_transformer.to(device)

        model_transformer_optim = torch.optim.Adam(
            model_transformer.parameters(),
            lr=model_lr,
            betas=(0.5, 0.999))

        global_steps = 0

    # Model Params size.
    model_params_size = sum(param.numel() for param in model_transformer.parameters())

    # Cross Entropy Loss, ignores pad token.
    ce_loss = nn.CrossEntropyLoss(ignore_index=vocab_size)

    # Update learning rate in case of changes.
    for model_transformer_optim_ in model_transformer_optim.param_groups:
        model_transformer_optim_["lr"] = model_lr
    
    # Constructs a ``scaler`` once, at the beginning of the convergence run, using default arguments.
    # If your network fails to converge with default ``GradScaler`` arguments, please file an issue.
    # The same ``GradScaler`` instance should be used for the entire convergence run.
    # If you perform multiple convergence runs in the same script, each run should use
    # a dedicated fresh ``GradScaler`` instance. ``GradScaler`` instances are lightweight.
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

    # Training starts here.
    stop_training = False
    while not stop_training:
        total_loss = 0
        training_count = 0

        for index, (in_seq, target_seq) in enumerate(train_dataloader):
            # Checkpoint and test model.
            if global_steps % checkpoint_steps == 0:
                # Compute Classification Metric.
                accuracy = compute_classification(
                    model=model_transformer,
                    dataloader=test_dataloader,
                    device=device)
                message = "Test Accuracy: {:.5f}".format(accuracy)
                logging.info(message)

                # Save model that has achieved max TPR with the dataset.
                model_dict = {
                    "global_steps": global_steps,
                    "model": model_transformer.state_dict(),
                    "optimizer": model_transformer_optim.state_dict()}

                save_status = save_model(
                    model_dict=model_dict,
                    dest_path=out_dir,
                    file_name=f"{global_steps}_model.pt",
                    logging=logging.info)
                if save_status is True:
                    logging.info("Successfully saved model.")
                else:
                    logging.info("Error occured saving model.")
            
                # Generate a sample text to test generative capabilities.
                model_transformer.eval()

                # Pick a valid starting token.
                test_seq, _ = next(iter(test_dataloader))
                test_seq = test_seq.to(device)

                # Get first token from last chunk to start the prediction.
                generated_token = [test_seq[0, 0].item()]
                while len(generated_token) < context_window:
                    generated_token_ = generated_token[:]

                    generated_token_tensor = torch.tensor([generated_token_], device=device)

                    with torch.no_grad():
                        out_seq = model_transformer(generated_token_tensor)

                    out_seq = out_seq[0]
                    probs = F.softmax(out_seq / 1, dim=1)
                    next_token = torch.multinomial(probs, 1).squeeze(1)

                    generated_token.append(next_token[len(generated_token) - 1].item())

                # Remove invalid tokens if any like padding token, not in vocab list.
                cleaned_pred_tokens = [clean_token for clean_token in generated_token if clean_token < vocab_size]
                pred_token_list = [vocab[c] for c in cleaned_pred_tokens]
                pred_txt = "".join(pred_token_list)
                logging.info(f"Generated text: {repr(pred_txt)}")

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

            # Otherwise, optimizer.step() is skipped.
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

if __name__ == "__main__":
    main()

