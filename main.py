# ============================================================
# Reasoning Segmentation with:
# - Qwen2.5-VL (pretrained VLM)
# - SAM / SAM2 encoder-decoder
# - LoRA fine-tuning
# - Multi-reasoning tokens
# - Cross-attention fusion
# ============================================================
import os
import cv2
import torch
import numpy as np
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration
)

from peft import (
    LoraConfig,
    get_peft_model
)

# ============================================================
# SPECIAL TOKENS
# ============================================================

SPECIAL_TOKENS = [

    # Atomic reasoning
    "[SEG_BUILDING]",
    "[SEG_ROAD]",
    "[SEG_FLOOD]",

    # Composite reasoning
    "[SEG_BUILDING_FLOODED]",
    "[SEG_BUILDING_NON_FLOODED]",

    "[SEG_ROAD_FLOODED]",
    "[SEG_ROAD_NON_FLOODED]",

    # Scene semantics
    "[SEG_WATER]",
    "[SEG_TREE]",
    "[SEG_VEHICLE]",
    "[SEG_POOL]",
    "[SEG_GRASS]"
]

# ============================================================
# CROSS ATTENTION FUSION
# ============================================================

class CrossAttentionFusion(nn.Module):

    def __init__(self, dim=256, heads=8):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, visual_tokens, reasoning_tokens):
        dtype = visual_tokens.dtype

        reasoning_tokens = reasoning_tokens.to(dtype)

        attended, _ = self.cross_attn(
            query=visual_tokens,
            key=reasoning_tokens,
            value=reasoning_tokens
        )

        x = self.norm1(visual_tokens + attended)

        ffn_out = self.ffn(x)

        out = self.norm2(x + ffn_out)

        return out


# ============================================================
# MULTI MASK DECODER
# ============================================================

class MultiMaskDecoder(nn.Module):

    def __init__(self, dim=256):
        super().__init__()

        self.mask_heads = nn.ModuleDict({

            "building_flooded":
                nn.Conv2d(dim, 1, 1),

            "building_non_flooded":
                nn.Conv2d(dim, 1, 1),

            "road_flooded":
                nn.Conv2d(dim, 1, 1),

            "road_non_flooded":
                nn.Conv2d(dim, 1, 1),

            "water":
                nn.Conv2d(dim, 1, 1),

            "tree":
                nn.Conv2d(dim, 1, 1),

            "vehicle":
                nn.Conv2d(dim, 1, 1),

            "pool":
                nn.Conv2d(dim, 1, 1),

            "grass":
                nn.Conv2d(dim, 1, 1)
        })

    def forward(self, x):

        outputs = {}

        for key, head in self.mask_heads.items():

            outputs[key] = head(x)

        return outputs

# ============================================================
# FIX 2:
# CLASS-SPECIFIC DECODER
# ============================================================

class MaskDecoder(nn.Module):

    def __init__(self, dim=256):
        super().__init__()

        self.decoder = nn.Sequential(

            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),

            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),

            nn.Conv2d(dim, 1, 1)
        )

    def forward(self, x):

        return self.decoder(x)


# ============================================================
# FIX 3:
# FULL IMPROVED MODEL
# ============================================================

class ReasoningSegmentationModel(nn.Module):

    def __init__(
        self,
        sam_encoder,
        sam_dim=256,
        vlm_name="Qwen/Qwen2.5-VL-3B-Instruct"
    ):

        super().__init__()

        # ----------------------------------------------------
        # FROZEN SAM ENCODER
        # ----------------------------------------------------

        self.sam_encoder = sam_encoder

        for p in self.sam_encoder.parameters():
            p.requires_grad = False

        # ----------------------------------------------------
        # LOAD QWEN2.5-VL
        # ----------------------------------------------------

        self.processor = AutoProcessor.from_pretrained(
            vlm_name
        )

        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            vlm_name,
            # torch_dtype=torch.bfloat16,
            device_map=None
        )
        self.vlm = self.vlm.half()
        self.vlm = self.vlm.to(device)

        # ----------------------------------------------------
        # ADD TOKENS
        # ----------------------------------------------------

        tokenizer = self.processor.tokenizer

        tokenizer.add_tokens(SPECIAL_TOKENS)

        self.vlm.resize_token_embeddings(
            len(tokenizer)
        )

        self.tokenizer = tokenizer

        # ----------------------------------------------------
        # LORA
        # ----------------------------------------------------

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj"
            ]
        )

        self.vlm = get_peft_model(
            self.vlm,
            lora_config
        )

        # ----------------------------------------------------
        # DIMENSIONS
        # ----------------------------------------------------

        hidden_dim = self.vlm.config.text_config.hidden_size

        # ----------------------------------------------------
        # PROJECTOR
        # ----------------------------------------------------

        # self.projector = nn.Linear(
        #     hidden_dim,
        #     sam_dim
        # ).to(torch.bfloat16)

        self.projector = nn.Linear(
            hidden_dim,
            sam_dim
        )
        self.projector = self.projector.half()

        # ----------------------------------------------------
        # CLASS TOKENS
        # ----------------------------------------------------

        self.token_mapping = {

            "building_flooded":
                "[SEG_BUILDING_FLOODED]",

            "building_non_flooded":
                "[SEG_BUILDING_NON_FLOODED]",

            "road_flooded":
                "[SEG_ROAD_FLOODED]",

            "road_non_flooded":
                "[SEG_ROAD_NON_FLOODED]",

            "water":
                "[SEG_WATER]",

            "tree":
                "[SEG_TREE]",

            "vehicle":
                "[SEG_VEHICLE]",

            "pool":
                "[SEG_POOL]",

            "grass":
                "[SEG_GRASS]"
        }

        # ----------------------------------------------------
        # FUSION BLOCKS
        # ----------------------------------------------------

        self.fusion_blocks = nn.ModuleDict()

        self.mask_decoders = nn.ModuleDict()

        for class_name in self.token_mapping:

            self.fusion_blocks[class_name] = (
                CrossAttentionFusion(sam_dim)
            )

            self.mask_decoders[class_name] = (
                MaskDecoder(sam_dim)
            )

    # ========================================================
    # SAFE TOKEN EXTRACTION
    # ========================================================

    def extract_token_embedding(
        self,
        hidden_states,
        input_ids,
        token
    ):

        token_id = self.tokenizer.convert_tokens_to_ids(
            token
        )

        B = input_ids.shape[0]

        embeddings = []

        for b in range(B):

            positions = (
                input_ids[b] == token_id
            ).nonzero(as_tuple=True)[0]

            if len(positions) == 0:
                pos = -1
            else:
                pos = positions[0]

            emb = hidden_states[b, pos]

            embeddings.append(emb)

        embeddings = torch.stack(embeddings)

        return embeddings

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, image, prompt):
        image = image.float()
        dtype = next(self.parameters()).dtype
        # ----------------------------------------------------
        # SAM FEATURES
        # ----------------------------------------------------

        with torch.no_grad():
            image_embeddings = self.sam_encoder(image.float())
        B, C, H, W = image_embeddings.shape

        # ----------------------------------------------------
        # VISUAL TOKENS
        # ----------------------------------------------------

        visual_tokens = image_embeddings.flatten(2)
        visual_tokens = visual_tokens.transpose(1, 2)
        visual_tokens = visual_tokens.to(dtype)

        # ----------------------------------------------------
        # FULL PROMPT
        # ----------------------------------------------------

        full_prompt = (
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                + prompt
                + " "
                + " ".join(SPECIAL_TOKENS)
        )

        # ----------------------------------------------------
        # VLM INPUT
        # ----------------------------------------------------

        images = [
            img.permute(1, 2, 0).cpu().numpy()
            for img in image
        ]

        inputs = self.processor(
            text=[full_prompt] * B,
            images=images,
            return_tensors="pt",
            padding=True
        )

        inputs = {
            k: v.to(image.device)
            for k, v in inputs.items()
        }

        # ----------------------------------------------------
        # QWEN FORWARD
        # ----------------------------------------------------

        outputs = self.vlm(
            **inputs,
            output_hidden_states=True
        )

        hidden_states = outputs.hidden_states[-1]
        hidden_states = hidden_states.half()
        
        # ----------------------------------------------------
        # CLASS-SPECIFIC MASKS
        # ----------------------------------------------------

        masks = {}

        for class_name, token in self.token_mapping.items():

            # --------------------------------------------
            # TOKEN EMBEDDING
            # --------------------------------------------

            token_embedding = self.extract_token_embedding(
                hidden_states,
                inputs["input_ids"],
                token
            )

            token_embedding = self.projector(
                token_embedding
            )

            token_embedding = token_embedding.unsqueeze(1)

            # --------------------------------------------
            # CROSS ATTENTION
            # --------------------------------------------

            fused = self.fusion_blocks[class_name](
                visual_tokens,
                token_embedding
            )

            # --------------------------------------------
            # RESTORE SPATIAL
            # --------------------------------------------

            fused = fused.transpose(1, 2)

            fused = fused.reshape(
                B,
                C,
                H,
                W
            )

            # --------------------------------------------
            # CLASS DECODER
            # --------------------------------------------

            masks[class_name] = (
                self.mask_decoders[class_name](fused)
            )

        return masks

# ============================================================
# LOSSES
# ============================================================

def dice_loss(
    pred,
    target,
    smooth=1e-6
):

    pred = torch.sigmoid(pred)

    intersection = (
        pred * target
    ).sum()

    union = pred.sum() + target.sum()

    dice = (
        2 * intersection + smooth
    ) / (
        union + smooth
    )

    return 1 - dice


def segmentation_loss(
    predictions,
    targets
):

    total_loss = 0

    for key in predictions.keys():

        pred = predictions[key]

        target = targets[key]

        bce = F.binary_cross_entropy_with_logits(
            pred,
            target
        )

        dice = dice_loss(
            pred,
            target
        )

        total_loss += bce + dice

    return total_loss



def dice_loss(pred, target, smooth=1e-6):

    pred = torch.sigmoid(pred)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()

    return 1 - (2 * intersection + smooth) / (union + smooth)


def total_loss(preds, masks):

    loss = 0

    for key in preds:

        pred = preds[key]
        target = masks[key]

        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = dice_loss(pred, target)

        loss += bce + dice

    return loss



# ============================================================
# CLASS MAP
# ============================================================

CLASS_MAP = {
    0: "background",
    1: "building_flooded",
    2: "building_non_flooded",
    3: "road_flooded",
    4: "road_non_flooded",
    5: "water",
    6: "tree",
    7: "vehicle",
    8: "pool",
    9: "grass"
}

def class_to_id(name):
    mapping = {
        "background": 0,
        "building_flooded": 1,
        "building_non_flooded": 2,
        "road_flooded": 3,
        "road_non_flooded": 4,
        "water": 5,
        "tree": 6,
        "vehicle": 7,
        "pool": 8,
        "grass": 9
    }
    return mapping[name]
# ============================================================
# FIX 1:
# RGB MASK ? CLASS ID CONVERSION
# ============================================================

COLOR_MAP = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (180, 120, 120): 2,
    (160, 150, 20): 3,
    (140, 140, 140): 4,
    (61, 230, 250): 5,
    (0, 82, 255): 6,
    (255, 0, 245): 7,
    (255, 235, 0): 8,
    (4, 250, 7): 9
}

def rgb_to_mask(mask_rgb):

    H, W, _ = mask_rgb.shape

    mask = np.zeros((H, W), dtype=np.uint8)

    for rgb, class_id in COLOR_MAP.items():

        matches = np.all(mask_rgb == rgb, axis=-1)

        mask[matches] = class_id

    return mask


# ============================================================
# FIXED DATASET
# ============================================================

class FloodNetDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(
            self.image_dir,
            self.images[idx]
        )

        mask_path = os.path.join(
            self.mask_dir,
            self.masks[idx]
        )

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image = cv2.resize(image, (1024, 1024))

        image = (
            torch.tensor(image)
            .permute(2, 0, 1)
            .float() / 255.0
        )

        # ----------------------------------------------------
        # RGB MASK
        # ----------------------------------------------------

        # mask_rgb = cv2.imread(mask_path)
        #
        # mask_rgb = cv2.cvtColor(
        #     mask_rgb,
        #     cv2.COLOR_BGR2RGB
        # )
        #
        # mask_rgb = cv2.resize(
        #     mask_rgb,
        #     (1024, 1024),
        #     interpolation=cv2.INTER_NEAREST
        # )
        #
        # mask = rgb_to_mask(mask_rgb)
        #
        # mask = torch.tensor(mask).long()

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(  mask,
                            (1024, 1024),
                            interpolation=cv2.INTER_NEAREST
                            )
        mask = torch.tensor(mask).long()

        return {
            "image": image,
            "mask": mask
        }




def train_one_epoch(model, loader, optimizer, device):

    model.train()

    total_loss = 0

    for batch in tqdm(loader):

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        prompt = (
            "Segment flooded buildings, roads, water, and vegetation. "
        )

        preds = model(image, prompt)

        loss = 0

        for key in preds:

            pred = preds[key]

            target = (mask == class_to_id(key)).float().unsqueeze(1)

            loss += (
                F.binary_cross_entropy_with_logits(pred, target)
                + dice_loss(pred, target)
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader, device):

    model.eval()

    total_loss = 0

    with torch.no_grad():

        for batch in loader:

            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            prompt = "Segment flooded buildings and roads."

            preds = model(image, prompt)

            loss = 0

            for key in preds:

                pred = preds[key]
                target = (mask == class_to_id(key)).float().unsqueeze(1)

                loss += (
                    F.binary_cross_entropy_with_logits(pred, target)
                    + dice_loss(pred, target)
                )

            total_loss += loss.item()

    return total_loss / len(loader)



def test(model, loader, device, save_dir):

    model.eval()

    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():

        for i, batch in enumerate(loader):

            image = batch["image"].to(device)

            prompt = (
                "Segment flooded buildings, roads, water, and vegetation."
            )

            preds = model(image, prompt)

            fused = None

            for key in preds:

                pred = torch.sigmoid(preds[key])[0, 0].cpu().numpy()

                if fused is None:
                    fused = pred
                else:
                    fused += pred

            fused = fused / len(preds)

            fused = (fused > 0.5).astype(np.uint8) * 255

            cv2.imwrite(
                os.path.join(save_dir, f"pred_{i}.png"),
                fused
            )

def save_checkpoint(model, optimizer, epoch, val_loss, best_loss, save_dir):

    os.makedirs(save_dir, exist_ok=True)

    if val_loss < best_loss:

        best_loss = val_loss

        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            },
            os.path.join(save_dir, "best_model.pt")
        )

        print("? Saved best model")

    return best_loss


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    epochs=50,
    save_dir="checkpoints"
):

    best_loss = float("inf")

    model.to(device)

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device
        )

        val_loss = evaluate(
            model,
            val_loader,
            device
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")

        best_loss = save_checkpoint(
            model,
            optimizer,
            epoch,
            val_loss,
            best_loss,
            save_dir
        )






# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

# ============================================================
# DATASET PATHS
# ============================================================
main_directory = "/home/exouser/Downloads/FloodNet/FloodNet-Supervised_v1.0"
train_image_dir = os.path.join(main_directory, "train/train-org-img")
train_mask_dir = os.path.join(main_directory, "train/train-label-img")

val_image_dir = os.path.join(main_directory, "val/val-org-img")
val_mask_dir = os.path.join(main_directory, "val/val-label-img")

test_image_dir = os.path.join(main_directory, "test/test-org-img")
test_mask_dir = os.path.join(main_directory, "test/test-label-img")

# ============================================================
# DATASETS
# ============================================================

train_dataset = FloodNetDataset(
    image_dir=train_image_dir,
    mask_dir=train_mask_dir
)


val_dataset = FloodNetDataset(
    image_dir=val_image_dir,
    mask_dir=val_mask_dir
)

test_dataset = FloodNetDataset(
    image_dir=test_image_dir,
    mask_dir=test_mask_dir
)

# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# ============================================================
# LOAD SAM ENCODER
# ============================================================

from segment_anything import sam_model_registry

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

sam = sam_model_registry[model_type](
    checkpoint=sam_checkpoint
)

sam_encoder = sam.image_encoder

# ============================================================
# BUILD MODEL
# ============================================================

model = ReasoningSegmentationModel(
    sam_encoder=sam_encoder,
    sam_dim=256,
    vlm_name="Qwen/Qwen2.5-VL-3B-Instruct"
)

model = model.to(device)

print("Model initialized successfully")

# ============================================================
# OPTIMIZER
# ============================================================

optimizer = optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=1e-4,
    weight_decay=1e-4
)

# ============================================================
# TRAIN
# ============================================================

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    device=device,
    epochs=50,
    save_dir="checkpoints"
)

# ============================================================
# LOAD BEST MODEL
# ============================================================

checkpoint = torch.load(
    "checkpoints/best_model.pt",
    map_location=device
)

model.load_state_dict(
    checkpoint["model"]
)

print("Best model loaded")

# ============================================================
# TEST
# ============================================================

test(
    model=model,
    loader=test_loader,
    device=device,
    save_dir="test_predictions"
)

print("Testing complete")
