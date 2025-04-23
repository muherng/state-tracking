import torch
import os
import json
import argparse
import numpy as np
import wandb
import random
import matplotlib.pyplot as plt

from transformers import (
    TrainingArguments, 
    Trainer, 
    EarlyStoppingCallback,
)

from utils.data_loaders import (
    ChunkedDataset,
)
from utils.data_collators import (
    DataCollatorForLanguageModelingWithNextTokenSupervision,
    DataCollatorForLanguageModelingWithDirectSupervision
)

from permutation_task import PermutationTask, compute_parity
from utils.model_utils import setup_tokenizer, setup_model


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", choices=[
        "gpt2", "gpt2-large", "gpt2-medium", "gpt2-xl", "distilgpt2",
        "EleutherAI/pythia-70M", "EleutherAI/pythia-160M", "EleutherAI/pythia-410M",
        "EleutherAI/pythia-1B", "EleutherAI/pythia-1.4B", "EleutherAI/pythia-2.8B",
        "EleutherAI/pythia-6.9B", "EleutherAI/pythia-12B","tree"
    ])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="saved_models")
    parser.add_argument("--num_items", type=int, default=3, choices=[3, 5], help="Number of items for permutation task")
    parser.add_argument("--supervision_type", type=str, default="direct_state", choices=["direct_state", "direct_topic", "next_token"])
    parser.add_argument("--layerwise_supervision_type", type=str, default=None, help="File containing layerwise supervision keys")
    parser.add_argument("--no_pretrain", action="store_true", default=False, help="If true, does not use pre-trained model")
    parser.add_argument("--from_checkpoint", type=str, default=None, help="If provided, loads model from checkpoint")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_bfloat16", action="store_true", default=False, help="If true, uses bfloat16")
    parser.add_argument("--save_all_checkpoints", type=int, default=None, help="If set, saves checkpoints and interval specified in argument")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_determinism", action="store_true", default=False)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--early_stopping", action="store_true", default=False)
    parser.add_argument("--is_parity_cur", action="store_true", default=False)
    parser.add_argument("--disable_wandb", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--full_tree", action="store_true", default=True)
    parser.add_argument("--chunk_size", type=int, default=64)
    parser.add_argument("--num_stories", type=int, default=10000)
    parser.add_argument("--generate_dataset", action="store_true", default=False, help="If true, generates dataset")
    parser.add_argument("--eval_lengths", type=str, default=None,
                        help="Skip training and evaluate a trained checkpoint on "
                        "each length given here, e.g. 8,16,24,32")
    return parser.parse_args()


def setup_data_collator(args, tokenizer, state_tokens, parity=None, layerwise_supervision_config=None):
    """
    Set up the data collator based on the supervision type.
    
    Args:
        args: Command-line arguments
        tokenizer: Tokenizer
        state_tokens: Dictionary mapping states to tokens
        parity: Dictionary mapping states to parity values
        layerwise_supervision_config: Configuration for layerwise supervision
        
    Returns:
        data_collator: Data collator for training
    """
    if args.supervision_type == "next_token":
        # For next token prediction, use standard language modeling collator
        data_collator = DataCollatorForLanguageModelingWithNextTokenSupervision(
            tokenizer=tokenizer,
            max_len=args.max_len,
        )
    elif args.supervision_type == "direct_state":
        # For direct state prediction, use sequence-to-sequence collator
        data_collator = DataCollatorForLanguageModelingWithDirectSupervision(
            tokenizer=tokenizer,
            max_len=args.max_len,
            STATE_TOKENS=state_tokens,
            label_key="state_seq",
            PARITY=parity if args.is_parity_cur else None,
            layerwise_intermediate_keys=layerwise_supervision_config
        )
    else:
        raise ValueError(f"Unknown supervision type: {args.supervision_type}")
        
    return data_collator


def prepare_dataset(args, tokenizer, state_tokens, data_collator, debug=False,train_test="train"):
    """Prepare the dataset for training."""
    print("Loading data")
    full_dataset = ChunkedDataset(args.data_dir, args.max_len, chunk_size=1, debug=debug,train_test=train_test)
    # Split into train/test
    print(f"Loaded full dataset with {len(full_dataset)} samples")
    total_size = len(full_dataset)
    train_size = int(0.95 * total_size)
    
    if args.full_determinism or args.data_determinism:
        # Use a fixed split by taking the first train_size examples for training
        indices = list(range(total_size))
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
        train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
        eval_dataset = torch.utils.data.Subset(full_dataset, eval_indices)
    else:
        torch.manual_seed(args.seed)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            full_dataset, 
            [train_size, total_size - train_size]
        )
    print("Train dataset size:", len(train_dataset))
    print("Eval dataset size:", len(eval_dataset))
    sample = train_dataset[0]
    print("Sample story:", sample['story'])
    print("Sample story:", sample['state_seq'])


    
    return train_dataset, eval_dataset


def setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator):
    """Set up the trainer with the appropriate arguments."""
    print("Setting up trainer")
    wandb.init(project="state-tracking", name=args.output_dir)
    training_args = TrainingArguments(
        eval_on_start=True,
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=500 if not args.save_all_checkpoints else args.save_all_checkpoints,
        save_strategy="steps",
        save_total_limit=None if args.save_all_checkpoints else 1,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="steps" if args.early_stopping else "epoch",
        eval_steps=2000 if args.early_stopping else None,
        batch_eval_metrics=True,
        remove_unused_columns=False,
        report_to="none" if args.disable_wandb else "wandb",
        dataloader_num_workers=1,
        bf16=args.use_bfloat16,
        max_grad_norm=1.0,
        seed=args.seed,
        data_seed=args.seed,
        full_determinism=args.full_determinism,
        metric_for_best_model="eval_loss" if args.early_stopping else None,
        greater_is_better=False if args.early_stopping else True,
        load_best_model_at_end=True if args.early_stopping else False,
    )
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    
    if args.early_stopping:
        trainer.add_callback(EarlyStoppingCallback(early_stopping_patience=3))
    
    return trainer

#Length Generalization command
#python train.py --model tree --chunk_size 1 --from_checkpoint 16 --eval_lengths 16,24,32,40

#Training command 

def main():
    """Main function."""
    args = parse_arguments()
    print('args.chunk_size:', args.chunk_size)
    print('args.max_len:', args.max_len)
    root_output = args.output_dir
    args.output_dir = args.output_dir + f"/{args.model}_{args.chunk_size}_{args.max_len}"
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up dataset directory based on max_len
    # (For instance, if training on max_len=4, then dataset_dir becomes "datasets/permutation_4")
    dataset_dir = os.path.join("datasets_new", f"permutation_{args.num_items}_{args.max_len}")
    
    # If the dataset does not already exist, generate it using the simulation function:
    if not os.path.exists(dataset_dir) or args.generate_dataset:
        print(f"Dataset for max_len={args.max_len} not found. Generating dataset...")
        # You might want to adjust these parameters as needed.
        task = PermutationTask(num_items=args.num_items)
        # Here we use story_length as the max_len (or map it as needed)
        num_steps = max(args.max_len,16)
        _stories, _states = task.simulate(
            steps=num_steps,  
            num_stories=args.num_stories,  # Adjust number of stories as needed
            story_so_far="",
            states_so_far=[task.init_state],
            write_dir=dataset_dir,
            train_ratio=0.99
        )
    else:
        print(f"Loading dataset from {dataset_dir}")
    
    # Now set args.data_dir to the dataset directory for prepare_dataset:
    args.data_dir = dataset_dir
    #Set args.from_checkpoint to the most recent checkpoint if available.
    checkpoint_root = root_output + f"/{args.model}_{args.chunk_size}_{args.from_checkpoint}"
    print('checkpoint_root:', checkpoint_root)
    if args.from_checkpoint is not None:
        if os.path.exists(checkpoint_root):
            ckpt_subdirs = [d for d in os.listdir(checkpoint_root) if d.startswith("checkpoint-") and d.split("-")[-1].isdigit()]
            print('ckpt_subdirs:', ckpt_subdirs)
            if ckpt_subdirs:
                latest_ckpt = max(ckpt_subdirs, key=lambda d: int(d.split("-")[-1]))
                args.from_checkpoint = os.path.join(checkpoint_root, latest_ckpt)
            else:
                print('set to None because no checkpoint found')
                args.from_checkpoint = None
        else:
            print('set to None because no checkpoint found 2')
            args.from_checkpoint = None
    print('args.from_checkpoint:', args.from_checkpoint)
    # Set up determinism
    if args.full_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up task
    task = PermutationTask(num_items=args.num_items)
    
    # Extract tokens and compute parity
    state_tokens = {state.permutation: state.to_string() for state in task.states}
    action_tokens = {action.permutation: action.to_string() for action in task.actions}
    parity = None
    
    # Update parity if needed
    if args.is_parity_cur:
        # Use the compute_parity function to calculate parity for all states
        parity = {state: compute_parity(state) for state in state_tokens}
    
    # Load layerwise supervision config if provided
    layerwise_supervision_config = None
    if args.layerwise_supervision_type is not None and os.path.exists(args.layerwise_supervision_type):
        with open(args.layerwise_supervision_type, "r") as f:
            layerwise_supervision_config = json.load(f)
    
    # Set up tokenizer
    tokenizer = setup_tokenizer(args.model, state_tokens, action_tokens)
    
    model = setup_model(
        tokenizer=tokenizer,
        model_name=args.model,
        checkpoint_path=args.from_checkpoint if hasattr(args, 'from_checkpoint') and args.from_checkpoint else None,
        use_bfloat16=args.use_bfloat16,
        no_pretrain=args.no_pretrain,
        output_dir=args.output_dir,
        use_custom_models=True,
        layerwise_supervision_config=layerwise_supervision_config,
        chunk_size = args.chunk_size if args.full_tree else args.max_len
    )
    
    # Set up data collator
    data_collator = setup_data_collator(
        args, tokenizer, state_tokens, parity, layerwise_supervision_config)

    # ---------- QUICK‑PATH: length‑generalisation ONLY ----------
    if args.eval_lengths:
        args.eval_lengths = str([int(8*i) for i in range(1,20)])[1:-1]
        if not args.from_checkpoint:
            raise ValueError("--eval_lengths needs a trained model supplied with --from_checkpoint")

        lengths = [int(x) for x in args.eval_lengths.split(",")]
        results = {}

        for L in lengths:
            print(f"\n### Evaluating sequence length {L} ###")
            args.max_len = L

            # Build / generate the appropriate dataset directory
            args.data_dir = os.path.join("datasets_new", f"permutation_{args.num_items}_{L}")
            if not os.path.exists(args.data_dir):
                print("  ↳ dataset missing – generating once")
                tmp_task = PermutationTask(num_items=args.num_items)
                tmp_task.simulate(steps=max(L,16),
                                  num_stories=args.num_stories,
                                  story_so_far="",
                                  states_so_far=[tmp_task.init_state],
                                  write_dir=args.data_dir,
                                  train_ratio=0.0)     # all examples go to eval

             # Prepare only the *eval* split; ignore training split
            _, eval_dataset = prepare_dataset(args, tokenizer, state_tokens, data_collator, debug=args.debug, train_test = "test")

            eval_trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=TrainingArguments(
                    output_dir=args.output_dir,
                    per_device_eval_batch_size=args.batch_size,
                    remove_unused_columns=False,
                    seed=args.seed,
                    report_to="none"),
                data_collator=data_collator,
                eval_dataset=eval_dataset,
            )

            metrics = eval_trainer.evaluate()
            print(metrics)
            results[L] = metrics["eval_loss"]
        
        # ---- PLOT: eval‑loss vs. sequence length ----
        xs = sorted(results.keys())
        ys = [results[x] for x in xs]

        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("Sequence length")
        plt.ylabel("Evaluation loss")
        plt.ylim(0, 20)
        plt.title("Length‑generalisation performance")
        plt.grid(True)

        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        plot_path = os.path.join(args.output_dir,
                                 f"length_generalisation_{timestamp}.png")
        plt.savefig(plot_path, bbox_inches="tight")
        print(f"\nPlot saved → {plot_path}")

        print("\n======  Summary  ======")
        for L, loss in results.items():
            print(f"L={L:>3}  |  eval_loss = {loss:.4f}")
        return

    # ---------- USUAL TRAINING PATH ----------
    train_dataset, eval_dataset = prepare_dataset(args, tokenizer, state_tokens, data_collator, debug=args.debug)
    trainer = setup_trainer(args, model, tokenizer, train_dataset, eval_dataset, data_collator)
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
