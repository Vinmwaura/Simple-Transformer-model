import torch
import torch.nn.functional as F

def generate_text(
        model,
        vocab,
        start_token,
        context_window,
        logging,
        device,
        temperature=1):
    vocab_size = len(vocab)

    # Generate a sample text to test generative capabilities.
    model.eval()

    logging.info(f"Starting Character: \"{vocab[start_token]}\"")

    generated_token_list = [start_token]
    while len(generated_token_list) < context_window:
        all_generated_token = generated_token_list[:]

        generated_token_tensor = torch.tensor([all_generated_token], device=device)

        with torch.no_grad():
            out_seq = model(generated_token_tensor)  # (1, Seq)

        probs = F.softmax(out_seq[0] / temperature, dim=1)

        # Pick most likely token for next generation for each Token Sequence (Seq).
        next_token = torch.multinomial(probs, 1).squeeze(1)

        # Save last token for next prediction.
        generated_token_list.append(next_token[-1].item())

    # Remove invalid tokens if any like padding token, not in vocab list.
    cleaned_pred_tokens = [clean_token for clean_token in generated_token_list if clean_token < vocab_size]
    pred_token_list = [vocab[c] for c in cleaned_pred_tokens]
    pred_txt = "".join(pred_token_list)
    logging.info(f"Generated text: {repr(pred_txt)}")
