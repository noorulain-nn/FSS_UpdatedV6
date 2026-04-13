"""
APM.py  —  Memory Module updated to work with FPN decoder output
================================================================
ONLY CHANGE from previous APM.py:
  Before: memory slots had shape [num_slots, 2048]
          because backbone output had 2048 channels

  Now:    memory slots have shape [num_slots, 256]
          because the FPN decoder output has 256 channels

  The cosine similarity logic, the adaptive EMA update rule,
  the novel prototype building — ALL unchanged.
  Just the feature_dim changes from 2048 to decoder.out_channels.

Also: the spatial resolution the memory module works at is now
  56×56 (from decoder) instead of 7×7 (from raw backbone).
  This means 3136 spatial comparisons per image instead of 49.
  The memory module code doesn't change — it handles any (h, w).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Decoder import FPNDecoder


class MemoryModule(nn.Module):
    """
    Unchanged from before — stores class prototypes and computes
    spatial cosine similarity. The feature_dim is now 256 (decoder
    output channels) instead of 2048 (raw backbone channels).
    """

    def __init__(self, num_base_classes, feature_dim):
        super().__init__()
        self.num_base_classes = num_base_classes
        self.feature_dim      = feature_dim
        self.num_base_slots   = num_base_classes + 1   # +1 for background

        self.memory = nn.Parameter(
            torch.randn(self.num_base_slots, feature_dim),
            requires_grad=False
        )
        nn.init.normal_(self.memory, mean=0.0, std=0.01)

        self.slot_ready       = [False] * self.num_base_slots
        self.novel_prototypes = {}

        print(f"[APM] Memory: {self.num_base_slots} slots × {feature_dim} dims "
              f"(decoder channels)")

    def forward(self, feature_map, novel_cls_id=None):
        """
        feature_map : [B, 256, 56, 56]  ← decoder output (was [B,2048,7,7])
        Returns logits [B, num_slots, 56, 56]
        Everything else identical to before.
        """
        B, D, h, w = feature_map.shape
        feat_norm  = F.normalize(feature_map, p=2, dim=1)

        if novel_cls_id is None:
            mem = F.normalize(self.memory, p=2, dim=1)
        else:
            bg_proto    = F.normalize(self.memory[0], p=2, dim=0)
            novel_proto = self.novel_prototypes[novel_cls_id]
            mem = torch.stack([bg_proto, novel_proto], dim=0)

        S         = mem.shape[0]
        feat_flat = feat_norm.view(B, D, h * w)
        mem_T     = mem.t()
        sim       = torch.bmm(
            feat_flat.permute(0, 2, 1),
            mem_T.unsqueeze(0).expand(B, -1, -1)
        )
        logits = sim.permute(0, 2, 1).view(B, S, h, w)
        return logits

    def update_from_batch(self, feature_map, binary_masks, class_labels):
        """Identical to before — updates base class prototypes via EMA."""
        B = feature_map.shape[0]
        for i in range(B):
            feat_i  = feature_map[i].unsqueeze(0)
            mask_i  = binary_masks[i].unsqueeze(0)
            cls     = class_labels[i]
            fg_slot = cls + 1
            self._update_slot(feat_i, (mask_i == 1).long(), fg_slot)
            self._update_slot(feat_i, (mask_i == 0).long(), 0)

    def _update_slot(self, feature_map, mask, slot_idx):
        D, h, w   = feature_map.shape[1:]
        mask_down = F.interpolate(
            mask.float().unsqueeze(1), size=(h, w), mode="nearest"
        )
        valid     = (mask_down != 255).float()
        mask_down = mask_down * valid
        denom     = mask_down.sum(dim=[0,2,3]).clamp(min=1e-6)
        proto_new = F.normalize(
            (feature_map * mask_down).sum(dim=[0,2,3]) / denom, p=2, dim=0
        )
        if not self.slot_ready[slot_idx]:
            self.memory.data[slot_idx] = proto_new
            self.slot_ready[slot_idx]  = True
        else:
            proto_old = F.normalize(self.memory.data[slot_idx], p=2, dim=0)
            sim       = F.cosine_similarity(
                proto_new.unsqueeze(0), proto_old.unsqueeze(0)
            ).item()
            alpha     = max(0.0, min(1.0 - sim, 1.0))
            self.memory.data[slot_idx] = (
                (1 - alpha) * self.memory.data[slot_idx] + alpha * proto_new
            )

    @torch.no_grad()
    def build_novel_prototype(self, support_features, support_masks, novel_cls_id):
        """Identical to before — builds prototype from K support images."""
        accumulated, count = None, 0
        for feat_i, mask_i in zip(support_features, support_masks):
            D, h, w   = feat_i.shape[1:]
            mask_down = F.interpolate(
                mask_i.float().unsqueeze(1), size=(h, w), mode="nearest"
            )
            valid     = (mask_down != 255).float()
            mask_down = mask_down * valid
            denom     = mask_down.sum(dim=[0,2,3]).clamp(min=1e-6)
            proto     = (feat_i * mask_down).sum(dim=[0,2,3]) / denom
            accumulated = proto if accumulated is None else accumulated + proto
            count += 1
        self.novel_prototypes[novel_cls_id] = F.normalize(
            accumulated / count, p=2, dim=0
        )
        print(f"[APM] Novel prototype built (class {novel_cls_id}) "
              f"from {count} support image(s). Dim={accumulated.shape[0]}")


# ─────────────────────────────────────────────────────────────────
# Full model: backbone + decoder + memory
# ─────────────────────────────────────────────────────────────────
class SegAPM(nn.Module):
    """
    Complete model with FPN decoder inserted between backbone and memory.

    Pipeline:
      Input [B, 3, 224, 224]
        ↓ backbone (layer4 fine-tunes)
      feat2 [B, 512,  28, 28]
      feat3 [B, 1024, 14, 14]
      feat4 [B, 2048,  7,  7]
        ↓ FPN decoder (all layers fine-tune)
      fused [B, 256, 56, 56]   ← richer, higher-res features
        ↓ memory module (cosine similarity)
      logits [B, num_slots, 56, 56]
        ↓ (in main.py) bilinear upsample
      logits [B, num_slots, 224, 224]
        ↓ CrossEntropyLoss
      segmentation loss

    What is trained during Phase 1:
      • backbone.layer4    — same as before
      • decoder (all layers) — NEW, trained from scratch
      • memory             — EMA updates, no gradients

    What is frozen during Phase 2 & 3:
      • everything
    """

    def __init__(self, backbone, num_base_classes, decoder_out_channels=256):
        super().__init__()
        self.backbone      = backbone
        self.decoder       = FPNDecoder(out_channels=decoder_out_channels)
        self.memory_module = MemoryModule(num_base_classes, decoder_out_channels)

        dec_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"[SegAPM] Decoder params: {dec_params:,}")
        print(f"[SegAPM] Memory feature dim: {decoder_out_channels} "
              f"(decoder output channels)")

    def forward(self, x, novel_cls_id=None):
        """
        Returns
        -------
        logits      : [B, num_slots, 56, 56]   ← higher res than before (was 7×7)
        fused_feats : [B, 256, 56, 56]          — for memory update
        """
        # Step 1: backbone → three feature maps
        feat2, feat3, feat4 = self.backbone(x)

        # Step 2: decoder fuses them → richer 56×56 map
        fused = self.decoder(feat2, feat3, feat4)

        # Step 3: memory module compares each of 56×56 = 3136 locations
        #         against stored prototypes
        logits = self.memory_module(fused, novel_cls_id)

        return logits, fused

    def freeze_everything(self):
        for param in self.parameters():
            param.requires_grad = False
        print("[SegAPM] All weights frozen for Phase 2 & 3.")