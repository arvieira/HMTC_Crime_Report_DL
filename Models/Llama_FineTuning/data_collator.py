import torch

class CustomDataCollator:
    def __init__(self, tokenizer, label_pad_token_id=-100):
        self.tokenizer = tokenizer
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, features):
        # Pad only for inputs (input_ids, attention_mask, etc.)
        input_features = [
            {k: f[k] for k in f if k != "labels"} for f in features
        ]
        batch = self.tokenizer.pad(
            input_features,
            padding="longest",
            return_tensors="pt"
        )

        # Manual padding labels
        # First, getting labels
        label_seqs = [f["labels"] for f in features]
        max_label_len = max(len(l) for l in label_seqs)

        # Manual padding eith -100 (this symbol is ignored by loss)
        padded_labels = [
            l + [self.label_pad_token_id] * (max_label_len - len(l))
            for l in label_seqs
        ]

        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return batch