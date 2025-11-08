# file: train_text_lm.py
# Usage examples:
#  - Train LSTM: python train_text_lm.py --text_file data/corpus.txt --model_type lstm --epochs 10
#  - Fine-tune HF GPT2: python train_text_lm.py --text_file data/corpus.txt --model_type gpt2 --epochs 3
#
# Requirements: transformers, datasets, torch, tqdm
# pip install transformers datasets torch tqdm

import argparse
import os
import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from transformers import AutoTokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------
# Dataset utils
# -------------------------
class TokenizedTextDataset(Dataset):
    """
    Given a list/array of token ids, provide input-target pairs of length seq_len.
    Data is a single long concatenated token stream.
    """
    def __init__(self, token_ids: List[int], seq_len: int):
        self.seq_len = seq_len
        self.token_ids = token_ids
        self.n = len(token_ids) - seq_len

    def __len__(self):
        return max(0, self.n)

    def __getitem__(self, idx):
        x = torch.tensor(self.token_ids[idx: idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.token_ids[idx + 1: idx + 1 + self.seq_len], dtype=torch.long)
        # We predict next token for every position (teacher forcing)
        return x, y


def load_and_tokenize(text_path: str, tokenizer_name: str, seq_len: int, max_tokens: int = None):
    with open(text_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    # Ensure tokenizer has an EOS token (GPT2 uses no pad, uses eos as sep). Add pad token if missing.
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    # Tokenize into ids (fast)
    enc = tokenizer(text, return_attention_mask=False, add_special_tokens=False)
    input_ids = enc["input_ids"]

    if max_tokens is not None:
        input_ids = input_ids[:max_tokens]

    # For convenience return tokenizer too
    return input_ids, tokenizer


# -------------------------
# LSTM model (token-level)
# -------------------------
class LSTMLM(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: (batch, seq_len)
        emb = self.embedding(input_ids)                    # (batch, seq_len, emb_dim)
        out, _ = self.lstm(emb)                            # (batch, seq_len, hidden_dim)
        logits = self.fc(out)                              # (batch, seq_len, vocab)
        return logits


# -------------------------
# Training / eval helpers
# -------------------------
def train_epoch_supervised(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # labels will be full token ids (no -100 here)
    for xb, yb in tqdm(dataloader, desc="train", leave=False):
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)               # (batch, seq_len, vocab)
        # reshape for loss: (batch*seq_len, vocab)
        loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def eval_model_supervised(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in tqdm(dataloader, desc="eval", leave=False):
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits.view(-1, logits.size(-1)), yb.view(-1))
            total_loss += loss.item() * xb.size(0)
            # optional: gather for later
            preds.append(logits.argmax(-1).cpu())
            trues.append(yb.cpu())
    avg_loss = total_loss / len(dataloader.dataset)
    # compute perplexity
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return avg_loss, ppl, preds, trues


# -------------------------
# Generation helpers
# -------------------------
@torch.no_grad()
def generate_from_lstm(model: LSTMLM, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, device=DEVICE):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    generated = input_ids.tolist()[0]
    for _ in range(max_new_tokens):
        cur = torch.tensor([generated[-args.seq_len:]], device=device) if len(generated) >= args.seq_len else torch.tensor([generated], device=device)
        logits = model(cur)                       # (1, seq_len, vocab)
        last_logits = logits[0, -1, :] / (temperature if temperature > 0 else 1.0)
        probs = torch.softmax(last_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
    return tokenizer.decode(generated, skip_special_tokens=True)


@torch.no_grad()
def generate_from_hf_transformer(hf_model: GPT2LMHeadModel, tokenizer, prompt: str, max_new_tokens: int = 50, temperature: float = 1.0, device=DEVICE):
    hf_model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = hf_model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, do_sample=(temperature>0), temperature=temperature, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------
# Main routine
# -------------------------
def main(args):
    # 1) load and tokenize
    token_ids, tokenizer = load_and_tokenize(args.text_file, args.tokenizer, seq_len=args.seq_len, max_tokens=args.max_tokens)
    vocab_size = len(tokenizer)
    print(f"Loaded text, total tokens: {len(token_ids)}, vocab_size: {vocab_size}")

    # split tokens train/val
    split = int(len(token_ids) * args.train_frac)
    train_ids = token_ids[:split]
    val_ids = token_ids[split:]

    train_ds = TokenizedTextDataset(train_ids, seq_len=args.seq_len)
    val_ds = TokenizedTextDataset(val_ids, seq_len=args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    os.makedirs(args.out_dir, exist_ok=True)

    if args.model_type == "lstm":
        # instantiate LSTM model
        model = LSTMLM(vocab_size=vocab_size, emb_dim=args.emb_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers, dropout=args.dropout).to(DEVICE)
        optimizer = AdamW(model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)
        best_ppl = float("inf")
        for epoch in range(1, args.epochs + 1):
            print(f"Epoch {epoch}/{args.epochs} (LSTM)")
            tr_loss = train_epoch_supervised(model, train_loader, optimizer, scheduler, DEVICE)
            val_loss, val_ppl, _, _ = eval_model_supervised(model, val_loader, DEVICE)
            print(f" -> train_loss: {tr_loss:.4f} | val_loss: {val_loss:.4f} | val_ppl: {val_ppl:.4f}")
            # save best
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                torch.save(model.state_dict(), os.path.join(args.out_dir, "best_lstm.pt"))
        # save tokenizer
        tokenizer.save_pretrained(args.out_dir)
        print("Saved best LSTM and tokenizer to", args.out_dir)
        # sample generation
        print("\n=== Sample generation (LSTM) ===")
        print(generate_from_lstm(model, tokenizer, args.sample_prompt, max_new_tokens=args.sample_length, temperature=args.temperature))

    elif args.model_type == "gpt2":
        # HF GPT-2 fine-tune (LMHead)
        # Load pretrained gpt2 and its tokenizer (we already used tokenizer_name but ensure model and tokenizer match)
        hf_tokenizer = tokenizer  # our tokenizer returned by AutoTokenizer
        hf_model = GPT2LMHeadModel.from_pretrained(args.hf_model_name)
        # make sure model token embeddings match tokenizer
        hf_model.resize_token_embeddings(len(hf_tokenizer))
        hf_model = hf_model.to(DEVICE)

        optimizer = AdamW(hf_model.parameters(), lr=args.lr)
        total_steps = len(train_loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05*total_steps), num_training_steps=total_steps)

        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        best_ppl = float("inf")
        for epoch in range(1, args.epochs + 1):
            hf_model.train()
            tr_loss = 0.0
            for xb, yb in tqdm(train_loader, desc="train", leave=False):
                xb = xb.to(DEVICE)   # inputs
                yb = yb.to(DEVICE)   # labels (next tokens)
                outputs = hf_model(input_ids=xb, labels=yb)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(hf_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                tr_loss += loss.item() * xb.size(0)
            tr_loss = tr_loss / len(train_loader.dataset)
            # eval
            hf_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in tqdm(val_loader, desc="eval", leave=False):
                    xb = xb.to(DEVICE)
                    yb = yb.to(DEVICE)
                    outputs = hf_model(input_ids=xb, labels=yb)
                    val_loss += outputs.loss.item() * xb.size(0)
            val_loss = val_loss / len(val_loader.dataset)
            val_ppl = math.exp(val_loss) if val_loss < 100 else float("inf")
            print(f"Epoch {epoch}/{args.epochs} | train_loss {tr_loss:.4f} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.4f}")
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                hf_model.save_pretrained(args.out_dir)
                hf_tokenizer.save_pretrained(args.out_dir)
        print("Saved best HF model/tokenizer to", args.out_dir)
        # sample generation
        print("\n=== Sample generation (GPT2) ===")
        print(generate_from_hf_transformer(hf_model, hf_tokenizer, args.sample_prompt, max_new_tokens=args.sample_length, temperature=args.temperature))

    else:
        raise ValueError("Unknown model_type, choose 'lstm' or 'gpt2'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True, help="path to raw text file (utf-8)")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="HF tokenizer name (default gpt2)")
    parser.add_argument("--hf_model_name", type=str, default="gpt2", help="HF model for transformer path/name")
    parser.add_argument("--model_type", type=str, choices=["lstm", "gpt2"], default="lstm", help="train LSTM or fine-tune HF GPT2")
    parser.add_argument("--seq_len", type=int, default=128, help="context length for training")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--emb_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--train_frac", type=float, default=0.9)
    parser.add_argument("--out_dir", type=str, default="lm_models")
    parser.add_argument("--max_tokens", type=int, default=None, help="optionally limit number of tokens used from corpus (for fast tests)")
    parser.add_argument("--sample_prompt", type=str, default="Bonjour, voici une histoire:", help="prompt to sample after training")
    parser.add_argument("--sample_length", type=int, default=100, help="number of tokens to sample after prompt")
    parser.add_argument("--temperature", type=float, default=0.9)
    args = parser.parse_args()
    main(args)
