import time

import torch
import torch.nn.functional as F

# Prints a string one character at a time with a delay.
def print_one_char_at_a_time(text, logging, delay=0.1):
    try:
        for char in text:
            logging(char, end='', flush=True)
            time.sleep(delay)
    except KeyboardInterrupt:
        logging("\n\n:(\nStopped text generation.")

def generate_text(
        model,
        vocab,
        device,
        init_text,
        context_window,
        logging=print,
        print_slowly=False,
        temperature=1):
    vocab_size = len(vocab)

    if print_slowly and logging is not print:
        raise Exception("Can only print one character at a time, using print function.")

    model.eval()

    logging("#" * 100)
    logging(f"Initial Text:\n{repr(init_text)}\n")

    list_characters = list(init_text)
    token_list = [vocab.index(character) for character in list_characters]

    generated_token_list = token_list[:]
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

    if print_slowly:
        # Emulate effect of printing character by character.
        logging("Generated text:")
        print_one_char_at_a_time(pred_txt, logging, delay=0.1)
    else:
        logging(f"Generated text:\n{pred_txt}")
        logging("#" * 100)
