import argparse, os, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from .preprocess import preprocess_dataframe

LABELS = ["label_ad","label_irrel","label_rant"]

def make_dataset(df, tokenizer):
    texts = df["text"].tolist()
    enc = tokenizer(texts, padding=True, truncation=True, max_length=256)
    labels = df[LABELS].astype(float).values
    enc["labels"] = labels
    return Dataset.from_dict(enc)

def main(args):
    df = pd.read_csv(args.data)
    df = preprocess_dataframe(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        problem_type="multi_label_classification"
    )

    ds = make_dataset(df, tokenizer).train_test_split(test_size=0.25, seed=42)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        from sklearn.metrics import f1_score, precision_score, recall_score
        return {
            "macro_f1": f1_score(labels, preds, average="macro", zero_division=0),
            "macro_precision": precision_score(labels, preds, average="macro", zero_division=0),
            "macro_recall": recall_score(labels, preds, average="macro", zero_division=0),
        }

    args_out = os.path.join(args.out_dir)
    os.makedirs(args_out, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=args_out,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.save_model(args_out)
    tokenizer.save_pretrained(args_out)
    print(f"Saved transformer model to {args_out}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out_dir", default="models/transformer")
    p.add_argument("--model_name", default="distilbert-base-uncased")
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()
    main(args)
