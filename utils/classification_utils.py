import torch

def compute_classification(
        model,
        dataloader,
        device="cpu"):
    total_correct = 0
    total_characters = 0

    model = model.to(device)

    model.eval()
    for index, (in_seq, target_seq) in enumerate(dataloader):
        print(f"{index + 1:,} / {len(dataloader):,}")

        in_seq = in_seq.to(device)  # (N, Seq)
        target_seq = target_seq.to(device)  # (N, Seq)

        with torch.no_grad():
            out_seq = model(in_seq)
        N, Seq, _ = out_seq.shape

        out_seq_flat = out_seq.view(N*Seq, -1)  # (N*Seq, Class)
        target_seq_flat = target_seq.flatten()  # (N*Seq)

        correct = torch.eq(
            torch.argmax(out_seq_flat, dim=1),
            target_seq_flat
        ).long()
        
        total_correct = total_correct + torch.sum(correct).item()
        total_characters = total_characters + (N * Seq)
    
    accuracy = total_correct / total_characters
    return accuracy
