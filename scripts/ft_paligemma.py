"""
Tutorial: Fine-tuning a PaliGemma model
according to https://huggingface.co/blog/paligemma

# LORA fine-tuning and inference
https://huggingface.co/docs/peft/en/quicktour
"""

import argparse
import json
import os
import pickle as pkl
import sys
import tempfile
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
import wandb  # Import wandb
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)
from transformers.trainer_callback import TrainerCallback

################################################################################


def process_image(image, pillow_image=False):
    # assert the shape is 256x256
    try:
        assert image.shape == (256, 256, 3)
    except AssertionError:
        # resize the image
        image = cv2.resize(image, (256, 256))

    # make sure the dtype is unit 8
    try:
        assert image.dtype == np.uint8
    except AssertionError:
        # convert the image to uint8
        assert image.dtype == np.float32
        image = (image * 255).astype(np.uint8)

    assert image.shape == (256, 256, 3)
    assert image.dtype == np.uint8

    if pillow_image:
        image = Image.fromarray(image)
        image.convert("RGB")

    return image


def color_jitter(image):
    """Apply slight color jitter to the image."""
    jitter = transforms.ColorJitter(
        brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
    )
    return jitter(image)


def random_crop(image, size=(240, 240)):
    """Apply slight random cropping to the image."""
    crop = transforms.RandomCrop(size)
    return crop(image)


def augment_image(image):
    """Augment the image with color jittering and random cropping."""
    image = color_jitter(image)
    image = random_crop(image)
    return image


################################################################################


class CustomDataset(Dataset):
    def __init__(self, transform=None, augment=False):
        self.transform = transform
        self.augment = augment

        self.data = []

    def __len__(self):
        # Length of the dataset
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.augment:
            item["image"] = augment_image(item["image"])
        return item if not self.transform else self.transform(item)


################################################################################


class DrawerDataset(CustomDataset):
    question = "is the drawer open? answer yes or no"

    def __init__(
        self,
        positive_demo_paths: List[str],
        negative_demo_paths: List[str],
        transform=None,
        pillow_image=False,
        augment=False,
    ):
        super().__init__(transform, augment)

        pos_transitions = []
        neg_transitions = []

        for positive_demo_path in positive_demo_paths:
            with open(positive_demo_path, "rb") as f:
                pos_transitions += pkl.load(f)

        for negative_demo_path in negative_demo_paths:
            with open(negative_demo_path, "rb") as f:
                neg_transitions += pkl.load(f)

        np.random.shuffle(pos_transitions)
        np.random.shuffle(neg_transitions)

        # ensure that the dataset is balanced
        print(
            f"Positive examples: {len(pos_transitions)}, Negative examples: {len(neg_transitions)}"
        )
        # min_len = min(len(pos_transitions), len(neg_transitions))
        # print(f"Only using {min_len} examples")
        # pos_transitions = pos_transitions[:min_len]
        # neg_transitions = neg_transitions[:min_len]

        # Add positive examples
        for transition in pos_transitions:
            img = transition["observation"]["image_primary"]
            img = process_image(img, pillow_image)
            self.data.append({"image": img, "question": self.question, "answer": "yes"})

        # Add negative examples
        for transition in neg_transitions:
            img = transition["observation"]["image_primary"]
            img = process_image(img, pillow_image)
            self.data.append({"image": img, "question": self.question, "answer": "no"})

        np.random.shuffle(self.data)


################################################################################


class EggplantSinkDataset(CustomDataset):
    question = (
        "is the eggplant in the sink or in the basket? answer sink or basket or invalid"
    )

    def __init__(
        self,
        in_sink_demo_paths: List[str],
        in_basket_demo_paths: List[str],
        invalid_demo_paths: List[str],
        transform=None,
        pillow_image=False,
        augment=False,
    ):
        super().__init__(transform, augment)

        in_sink_transitions = []
        in_basket_transitions = []
        invalid_transitions = []

        for in_sink_demo_path in in_sink_demo_paths:
            with open(in_sink_demo_path, "rb") as f:
                in_sink_transitions += pkl.load(f)

        for in_basket_demo_path in in_basket_demo_paths:
            with open(in_basket_demo_path, "rb") as f:
                in_basket_transitions += pkl.load(f)

        for invalid_demo_path in invalid_demo_paths:
            with open(invalid_demo_path, "rb") as f:
                invalid_transitions += pkl.load(f)

        np.random.shuffle(in_sink_transitions)
        np.random.shuffle(in_basket_transitions)
        np.random.shuffle(invalid_transitions)
        # ensure that the dataset is balanced
        print(
            f"In sink examples: {len(in_sink_transitions)}, In basket examples: {len(in_basket_transitions)}, Invalid examples: {len(invalid_transitions)}"
        )
        # min_len = min(
        #     len(in_sink_transitions
        # ), len(in_basket_transitions
        # ), len(invalid_transitions
        # )
        # print(f"Only using {min_len} examples")
        # in_sink_transitions = in_sink_transitions[:min_len]
        # in_basket_transitions = in_basket_transitions[:min_len]
        # invalid_transitions = invalid_transitions[:min_len]

        # add answers to data cache
        for answer, transitions in zip(
            ["sink", "basket", "invalid"],
            [in_sink_transitions, in_basket_transitions, invalid_transitions],
        ):
            for transition in transitions:
                img = transition["observation"]["image_primary"]
                img = process_image(img, pillow_image)
                self.data.append(
                    {"image": img, "question": self.question, "answer": answer}
                )


################################################################################


class ClothDataset(DrawerDataset):
    question = "is the cloth folded or unfolded? answer yes or no"


################################################################################


class WandBTrainerCallback(TrainerCallback):
    def __init__(self):
        self.train_step = 0

    def on_init_end(self, args, state, control, **kwargs):
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        return control

    def on_train_end(self, args, state, control, **kwargs):
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        logs = {}
        if state.log_history:
            # Get the last logged values
            logs = state.log_history[-1].copy()
        wandb.log(logs, step=state.global_step)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            wandb.log(metrics, step=state.global_step)
        return control

    def on_save(self, args, state, control, **kwargs):
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            wandb.log(logs, step=state.global_step)
        return control


################################################################################


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processor_id", type=str, default="google/paligemma-3b-pt-224"
    )
    parser.add_argument("--model_id", type=str, default="google/paligemma-3b-pt-224")
    parser.add_argument("--n_train_epochs", type=int, default=80)
    parser.add_argument(
        "--batch_size", type=int, default=4
    )  # 2 for 12G, 4 for 24G, 24 for 80G

    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--eval_save_visualizations", action="store_true", default=False
    )

    parser.add_argument("--working_dir", type=str, default="~/datasets")
    parser.add_argument("--dataset_type", type=str, default="drawer")
    parser.add_argument("--quantize", action="store_true", default=True)
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project="auto_eval_paligemma_finetuning",
        entity="rail-iterated-offline-rl",
        config={
            "n_train_epochs": args.n_train_epochs,
            "batch_size": args.batch_size,
            "learning_rate": 2e-5,
            "weight_decay": 1e-6,
        },
        dir=tempfile.mkdtemp(),
        mode="disabled" if args.eval else "online",
    )

    # deterministic behavior
    np.random.seed(0)
    torch.manual_seed(0)

    # get paths with of pkl files and make dataset
    if args.dataset_type == "drawer":
        positive_demo_paths = Path(args.working_dir).glob("record-open-drawer*.pkl")
        negative_demo_paths = Path(args.working_dir).glob("record-close-drawer*.pkl")

        dataset = DrawerDataset(
            positive_demo_paths,
            negative_demo_paths,
            pillow_image=True,
            augment=True,
        )
    elif args.dataset_type == "sink":

        in_sink_demo_paths = Path(args.working_dir).glob("record-in-sink*.pkl")
        in_basket_demo_paths = Path(args.working_dir).glob("record-in-basket*.pkl")
        invalid_demo_paths = Path(args.working_dir).glob("record-invalid*.pkl")
        dataset = EggplantSinkDataset(
            in_sink_demo_paths=in_sink_demo_paths,
            in_basket_demo_paths=in_basket_demo_paths,
            invalid_demo_paths=invalid_demo_paths,
            pillow_image=True,
            augment=True,
        )
    elif args.dataset_type == "cloth":
        folded_demo_paths = Path(args.working_dir).glob("record-folded*.pkl")
        unfolded_demo_paths = Path(args.working_dir).glob("record-unfolded*.pkl")
        dataset = ClothDataset(
            positive_demo_paths=folded_demo_paths,
            negative_demo_paths=unfolded_demo_paths,
            pillow_image=True,
            augment=True,
        )
    else:
        raise RuntimeError(f"Invalid dataset type: {args.dataset_type}")

    ################################################################################
    # Split the dataset into training and validation sets
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    print(f"Training set size: {len(train_ds)}, Validation set size: {len(val_ds)}")

    # datasets.config.DOWNLOADED_DATASETS_PATH = Path("/hdd/huggingface")

    """
    PT checkpoints: Pretrained models that can be fine-tuned to downstream tasks.
    Mix checkpoints: PT models fine-tuned to a mixture of tasks.
    FT checkpoints: A set of fine-tuned models
    """
    device = "cuda"

    ################################################################################
    # eval mode

    if args.eval:
        print("Eval mode")
        from auto_eval.success_detector.paligemma import PaligemmaDetector

        detector = PaligemmaDetector(
            processor_id=args.processor_id,
            model_id=args.model_id,
            device=device,
            quantize=args.quantize,
        )

        # only load 100 examples
        total = 100
        class_balance = {}
        correct = 0

        progress_bar = tqdm(val_ds, total=total)
        for i, example in enumerate(progress_bar):
            if i == total:
                break

            decoded_output = detector(example["question"], example["image"])
            decoded_output = decoded_output.strip().lower()

            # visualize image along side the output
            img = example["image"]
            if args.eval_save_visualizations:
                overlay_text = (
                    f"VLM output: {decoded_output} | Label: {example['answer']}"
                )
                img_with_overlay = Image.new("RGB", img.size)
                img_with_overlay.paste(img)
                draw = ImageDraw.Draw(img_with_overlay)
                draw.text(
                    (10, 10), overlay_text, fill="white"
                )  # Adjust position and color as needed
                img_with_overlay.save(
                    f"{args.working_dir}/visualization/output_{i}.png"
                )  # Save the image with overlay

            # print(f"VLM output: {decoded_output} | Label: {example['answer']}")
            if example["answer"] in decoded_output:
                correct += 1

            # keep track of each class
            if example["answer"] in class_balance:
                class_balance[example["answer"]] += 1
            else:
                class_balance[example["answer"]] = 1

        print(f"Accuracy: {correct/min(total, i+1)}")
        print("Eval data class balance: ")
        for key, value in class_balance.items():
            print(f"Class: {key} | Count: {value}")
        exit()

    ################################################################################
    # Finetuning the model
    from peft import LoraConfig, get_peft_model
    from transformers import BitsAndBytesConfig, Trainer, TrainingArguments

    processor = PaliGemmaProcessor.from_pretrained(args.processor_id)

    def collate_fn(examples):
        texts = ["<image>" + example["question"] for example in examples]
        images = [example["image"] for example in examples]
        labels = [example["answer"] for example in examples]
        tokens = processor(
            text=texts,
            images=images,
            suffix=labels,
            return_tensors="pt",
            padding="longest",
        )
        tokens = tokens.to(torch.bfloat16).to(device)
        return tokens

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_type=torch.bfloat16,
    )

    lora_config = LoraConfig(
        r=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        args.model_id, quantization_config=bnb_config, device_map={"": 0}
    )

    # set the vision tower to be frozen
    # for param in model.vision_tower.parameters():
    #     param.requires_grad = False

    # # set the multi-modal projector to be trainable
    # for param in model.multi_modal_projector.parameters():
    #     param.requires_grad = True

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # trainable params: 11,298,816 || all params: 2,934,765,296 || trainable%: 0.3850

    # Save demo paths and metadata to config.json
    config_data = {
        "processor_id": args.processor_id,
        "model_id": args.model_id,
        "n_train_epochs": args.n_train_epochs,
        "batch_size": args.batch_size,
        "working_dir": args.working_dir,
        "command": sys.argv,
    }
    if args.dataset_type == "drawer":
        config_data["positive_demo_paths"] = (
            [str(path) for path in list(positive_demo_paths)],
        )
        config_data["negative_demo_paths"] = (
            [str(path) for path in list(negative_demo_paths)],
        )
    elif args.dataset_type == "sink":
        config_data["in_sink_demo_paths"] = (
            [str(path) for path in list(in_sink_demo_paths)],
        )
        config_data["in_basket_demo_paths"] = (
            [str(path) for path in list(in_basket_demo_paths)],
        )
        config_data["invalid_demo_paths"] = (
            [str(path) for path in list(invalid_demo_paths)],
        )
    config_path = os.path.join(args.working_dir, "checkpoints", "config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as config_file:
        json.dump(config_data, config_file, indent=4)

    args = TrainingArguments(
        output_dir=f"{args.working_dir}/checkpoints",
        num_train_epochs=args.n_train_epochs,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        learning_rate=2e-5,
        weight_decay=1e-6,
        adam_beta2=0.999,
        logging_steps=2,
        optim="paged_adamw_8bit",  # adamw_hf
        save_strategy="steps",
        save_steps=20,
        # push_to_hub=True,
        save_total_limit=50,
        bf16=True,
        report_to=["tensorboard"],
        dataloader_pin_memory=False,
        eval_strategy="steps",
        eval_steps=20,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        args=args,
        # Add logging to wandb
        callbacks=[WandBTrainerCallback()],
    )
    trainer.train()
    wandb.finish()  # Finish the wandb run
