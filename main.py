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
            batch_first=True,
            dropout=0.0
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        self.vis_norm = nn.LayerNorm(dim)
        self.txt_norm = nn.LayerNorm(dim)

    def forward(self, visual_tokens, reasoning_tokens):
        dtype = visual_tokens.dtype

        reasoning_tokens = reasoning_tokens.to(dtype)

        visual_tokens = self.vis_norm(visual_tokens)
        reasoning_tokens = self.txt_norm(reasoning_tokens)

        # print("visual af:", visual_tokens.mean().item(), visual_tokens.std().item())
        # print("reason af:", reasoning_tokens.mean().item(), reasoning_tokens.std().item())

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

            nn.ConvTranspose2d(dim, dim // 2, 2, stride=2),
            nn.GELU(),

            nn.ConvTranspose2d(dim // 2, dim // 4, 2, stride=2),
            nn.GELU(),

            nn.ConvTranspose2d(dim // 4, dim // 8, 2, stride=2),
            nn.GELU(),

            nn.ConvTranspose2d(dim // 8, dim // 16, 2, stride=2),
            nn.GELU(),

            nn.Conv2d(dim // 16, 1, 1)
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
        self.vlm = self.vlm
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
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
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
        # self.projector = self.projector

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

        self.mask_decoder = MaskDecoder(sam_dim)

        # ----------------------------------------------------
        # decoder BLOCKS
        # ----------------------------------------------------
        self.class_embed = nn.Embedding(len(self.token_mapping), sam_dim)

        self.vlm = self.vlm.float()
        self.projector = self.projector.float()
        self.class_embed = self.class_embed.float()

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
                raise ValueError(f"Token {token} not found")
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

        images = []

        for img in image:
            img_qwen = F.interpolate(
                img.unsqueeze(0),
                size=(400, 400),
                mode="bilinear",
                align_corners=False
            )[0]
            images.append(
                img_qwen.permute(1, 2, 0).cpu().numpy()
            )

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
        if torch.isnan(hidden_states).any():
            print("NaN detected in VLM hidden states")
            exit()
        # hidden_states = hidden_states.half()

        # ----------------------------------------------------
        # CLASS-SPECIFIC MASKS
        # ----------------------------------------------------

        masks = {}

        for class_name, token in self.token_mapping.items():
            token_embedding = self.extract_token_embedding(
                hidden_states,
                inputs["input_ids"],
                token
            )

            token_embedding = self.projector(token_embedding)
            token_embedding = token_embedding.unsqueeze(1)

            # print("visual df:", visual_tokens.mean().item(), visual_tokens.std().item())
            # print("reason df:", token_embedding.mean().item(), token_embedding.std().item())

            fused = self.fusion_blocks[class_name](
                visual_tokens,
                token_embedding
            )

            fused = fused.transpose(1, 2)
            fused = fused.reshape(B, C, H, W)

            # ----------------------------
            # ADD CLASS CONDITIONING
            # ----------------------------

            class_idx = list(self.token_mapping.keys()).index(class_name)
            class_idx = torch.tensor(class_idx, device=fused.device)

            class_vec = self.class_embed(class_idx)
            class_vec = class_vec.view(1, C, 1, 1)

            fused = fused + class_vec

            # ----------------------------
            # SHARED DECODER
            # ----------------------------

            masks[class_name] = self.mask_decoder(fused)

        return masks

# ============================================================
# LOSSES
# ============================================================

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
    pred = torch.sigmoid(torch.clamp(pred, -10, 10))

    pred = pred.flatten(1)
    target = target.flatten(1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2 * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()


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




def train_one_epoch(model, loader, optimizer, device, save_vis_dir=None):

    model.train()

    total_loss = 0
    total_dice = 0

    for i, batch in enumerate(tqdm(loader)):

        image = batch["image"].to(device)
        mask = batch["mask"].to(device)

        prompt = "Segment flooded buildings, roads, water, and vegetation."

        preds = model(image, prompt)

        loss = 0

        # combine all classes for visualization
        pred_class_masks = []

        for key in preds:

            pred = preds[key]
            target = (mask == class_to_id(key)).float().unsqueeze(1)

            loss += (
                F.binary_cross_entropy_with_logits(pred, target)
                + dice_loss(pred, target)
            )

            pred_class_masks.append(torch.sigmoid(pred))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # ---- metrics (example: aggregate first class or mean) ----
        pred_mask = torch.argmax(torch.cat(pred_class_masks, dim=1), dim=1)
        gt_mask = mask

        dice = dice_score(pred_mask.float(), gt_mask.float())
        total_dice += dice.item()

        # ---- save sample visualization every N steps ----
        if save_vis_dir and i % 50 == 0:
            os.makedirs(save_vis_dir, exist_ok=True)

            save_visualization(
                image[0],
                gt_mask[0],
                pred_mask[0],
                os.path.join(save_vis_dir, f"train_{i}.png")
            )

        if i > 0 and i % 10 == 0:
            break

    print(f"Train Loss: {total_loss/len(loader):.4f} | Dice: {total_dice/len(loader):.4f}")

    return total_loss / len(loader)

def evaluate(model, loader, device, save_vis_dir=None):

    model.eval()

    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():

        for i, batch in enumerate(loader):

            image = batch["image"].to(device)
            mask = batch["mask"].to(device)

            prompt = "Segment flooded buildings, roads, water, and vegetation."

            preds = model(image, prompt)

            loss = 0
            pred_class_masks = []

            for key in preds:

                pred = preds[key]
                target = (mask == class_to_id(key)).float().unsqueeze(1)

                loss += (
                    F.binary_cross_entropy_with_logits(pred, target)
                    + dice_loss(pred, target)
                )

                pred_class_masks.append(torch.sigmoid(pred))

            total_loss += loss.item()

            pred_mask = torch.argmax(torch.cat(pred_class_masks, dim=1), dim=1)

            dice = dice_score(pred_mask.float(), mask.float())
            iou = iou_score(pred_mask.float(), mask.float())

            total_dice += dice.item()
            total_iou += iou.item()

            if save_vis_dir and i % 20 == 0:
                os.makedirs(save_vis_dir, exist_ok=True)

                save_visualization(
                    image[0],
                    mask[0],
                    pred_mask[0],
                    os.path.join(save_vis_dir, f"eval_{i}.png")
                )

    print(
        f"Eval Loss: {total_loss/len(loader):.4f} | "
        f"Dice: {total_dice/len(loader):.4f} | "
        f"IoU: {total_iou/len(loader):.4f}"
    )

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

            pred_stack = []

            for key in preds:
                pred = torch.sigmoid(
                    preds[key]
                )

                pred_stack.append(pred)

            pred_stack = torch.cat(
                pred_stack,
                dim=1
            )

            final_mask = torch.argmax(
                pred_stack,
                dim=1
            )

            fused = final_mask[0].cpu().numpy().astype(np.uint8)

            cv2.imwrite(
                os.path.join(save_dir, f"pred_{i}.png"),
                fused
            )


import matplotlib.pyplot as plt

def save_visualization(image, gt_mask, pred_mask, save_path):
    """
    image: (3, H, W) tensor
    gt_mask: (H, W)
    pred_mask: (H, W)
    """

    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)

    gt_mask = gt_mask.cpu().numpy()
    pred_mask = pred_mask.cpu().numpy()

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].imshow(image)
    axs[0].set_title("Input")
    axs[0].axis("off")

    axs[1].imshow(gt_mask, cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")

    axs[2].imshow(pred_mask, cmap="gray")
    axs[2].set_title("Prediction")
    axs[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def dice_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    return (2. * inter + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred, target, eps=1e-6):
    pred = (pred > 0.5).float()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + eps) / (union + eps)


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
    scheduler,
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

        scheduler.step()

        # best_loss = save_checkpoint(
        #     model,
        #     optimizer,
        #     epoch,
        #     val_loss,
        #     best_loss,
        #     save_dir
        # )






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
    lr=1e-5,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=50
)
# ============================================================
# TRAIN
# ============================================================

train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
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
