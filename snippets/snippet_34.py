"""
Stage 2 of Lift3D-VLA: lightweight fine-tuning of the 3D vision encoder
and an auxiliary dual-branch decoder, while the LLM (and its attention
LoRAs from Stage 1) stay frozen. Only ~10% of the total Stage-1 step
budget is used here. At inference the dual decoder is dropped.

Article-ready, self-contained PyTorch example.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Model components
# ---------------------------------------------------------------------------

class FrozenLLMBackbone(nn.Module):
    """A stand-in for a frozen 7B-class LLM with cross-attention to 3D tokens.

    Everything is trainable=False. The `attention_lora` parameters are also
    frozen (they were trained in Stage 1, kept fixed here).
    """

    def __init__(self, d: int = 512) -> None:
        super().__init__()
        self.d = d
        # Symbolic "attention LoRA" parameters (frozen in Stage 2).
        self.attention_lora = nn.ParameterDict({
            "q_down": nn.Parameter(torch.randn(d, 8) * 0.02, requires_grad=False),
            "q_up":   nn.Parameter(torch.randn(8, d) * 0.02, requires_grad=False),
        })
        # Frozen transformer body.
        body = nn.TransformerEncoderLayer(d_model=d, nhead=8, dim_feedforward=4 * d, batch_first=True)
        self.body = body.eval()
        for p in self.body.parameters():
            p.requires_grad = False

    def forward(self, text_emb: torch.Tensor, vision_emb: torch.Tensor) -> torch.Tensor:
        # Concatenate text + vision tokens, run through frozen body.
        tokens = torch.cat([text_emb, vision_emb], dim=1)
        return self.body(tokens)


class Vision3DEncoder(nn.Module):
    """Trainable in Stage 2: lifts point clouds to token embeddings.

    Geometry/dynamics-aware: a small PointNet-like stem + a temporal mixer.
    """

    def __init__(self, in_dim: int = 6, d: int = 512) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(),
            nn.Linear(128, d),
        )
        self.temporal = nn.GRU(input_size=d, hidden_size=d, num_layers=1, batch_first=True)

    def forward(self, pc_seq: torch.Tensor) -> torch.Tensor:
        # pc_seq: (B, T, N, in_dim)  -> (B, T*N, d)
        B, T, N, _ = pc_seq.shape
        x = self.proj(pc_seq.reshape(B, T * N, -1))
        x = x.view(B, T, N, -1).mean(dim=2)  # pool over points
        x, _ = self.temporal(x)              # (B, T, d)
        return x.reshape(B, T, -1)


class DualBranchDecoder(nn.Module):
    """Two heads sharing a trunk.

    * ``main``  — produces the action / text continuation fed to the LLM
                  (the branch that *survives* inference).
    * ``aux``   — geometry/dynamics auxiliary head used **only** during
                  Stage 2 to inject a richer learning signal into the
                  3D encoder. It is **discarded at inference**.
    """

    def __init__(self, d: int = 512, action_dim: int = 7) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(d, d), nn.GELU(),
            nn.Linear(d, d),
        )
        self.main = nn.Linear(d, d)         # kept at inference
        self.aux  = nn.Linear(d, action_dim)  # dropped at inference

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(z)
        return self.main(h), self.aux(h)


class Lift3DVLA(nn.Module):
    def __init__(self, d: int = 512, action_dim: int = 7) -> None:
        super().__init__()
        self.llm = FrozenLLMBackbone(d=d)
        self.vision = Vision3DEncoder(d=d)
        self.decoder = DualBranchDecoder(d=d, action_dim=action_dim)

    # Inference-only view: drop the auxiliary head.
    def as_inference_model(self) -> "InferenceVLA":
        return InferenceVLA(self.llm, self.vision, self.decoder.trunk, self.decoder.main)


class InferenceVLA(nn.Module):
    """Deployment module: no aux head, no decoder trunk fork."""

    def __init__(self, llm: FrozenLLMBackbone, vision: Vision3DEncoder,
                 trunk: nn.Module, main_head: nn.Module) -> None:
        super().__init__()
        self.llm, self.vision = llm, vision
        self.trunk, self.main_head = trunk, main_head

    def forward(self, text_emb: torch.Tensor, pc_seq: torch.Tensor) -> torch.Tensor:
        z = self.vision(pc_seq)
        z = self.trunk(z)
        z = self.main_head(z)
        return self.llm(text_emb, z)


# ---------------------------------------------------------------------------
# 2. Stage-2 trainer (10% of Stage-1 step budget)
# ---------------------------------------------------------------------------

@dataclass
class Stage2Config:
    total_stage1_steps: int = 20_000
    fraction: float = 0.10
    lr_vision: float = 1e-4
    lr_decoder: float = 3e-4
    aux_loss_weight: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0
    # Steps actually scheduled for Stage 2 (resolved at construction time).
    steps: int = field(init=False)

    def __post_init__(self) -> None:
        self.steps = max(1, int(math.ceil(self.total_stage1_steps * self.fraction)))


def set_requires_grad(module: nn.Module, flag: bool, *, also: Iterable[str] = ()) -> None:
    """Freeze / unfreeze ``module`` and (optionally) named submodules by prefix."""
    for p in module.parameters():
        p.requires_grad = flag
    for name, sub in module.named_modules():
        if any(name.startswith(prefix) for prefix in also):
            for p in sub.parameters():
                p.requires_grad = flag


class Stage2Trainer:
    """Surgeon-style Stage 2: narrow scope, short horizon, discarded head."""

    def __init__(self, model: Lift3DVLA, cfg: Stage2Config) -> None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        self.cfg = cfg
        self.model = model.to(cfg.device)

        # ---- Freeze everything by default. ----
        set_requires_grad(self.model, False)
        # Belt-and-braces: keep the LLM body and its LoRAs frozen explicitly.
        set_requires_grad(self.model.llm, False, also=("attention_lora",))
        # LLM body must stay in eval mode (dropout, etc.).
        self.model.llm.body.eval()

        # ---- Unfreeze the Stage-2 trainable parts only. ----
        for p in self.model.vision.parameters():
            p.requires_grad = True
        for p in self.model.decoder.parameters():
            p.requires_grad = True

        # ---- Two parameter groups (different LRs are a common trick). ----
        self.opt = torch.optim.AdamW(
            [
                {"params": self.model.vision.parameters(), "lr": cfg.lr_vision},
                {"params": self.model.decoder.parameters(), "lr": cfg.lr_decoder},
            ],
            weight_decay=0.01,
        )

    # ------------------------------------------------------------------ #
    def _step(self, batch: dict) -> dict:
        pc_seq   = batch["pc_seq"].to(self.cfg.device)      # (B, T, N, 6)
        text_emb = batch["text_emb"].to(self.cfg.device)    # (B, L, d)
        action   = batch["action"].to(self.cfg.device)      # (B, T, action_dim)

        # Forward through the (still-detached) LLM is fine because
        # requires_grad is False, so no graph is built for it.
        with torch.no_grad():
            _ = self.model.llm(text_emb, self.model.vision(pc_seq) * 0.0)  # warm-up sanity

        # Real Stage-2 forward: vision -> dual decoder.
        z = self.model.vision(pc_seq)
        main_logits, aux_pred = self.model.decoder(z)         # both heads used here
        llm_out = self.model.llm(text_emb, main_logits)        # frozen LLM consumes `main`

        # Losses.
        lm_loss     = llm_out.mean() * 0.0                     # placeholder
        aux_loss    = F.mse_loss(aux_pred, action)
        # In a real impl you'd also have a next-token loss on `lm_loss`.
        loss = lm_loss + self.cfg.aux_loss_weight * aux_loss

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        return {"loss": float(loss.detach()), "trainable_params": n_trainable}

    # ------------------------------------------------------------------ #
    def fit(self, dataloader) -> None:
        print(f"[Stage 2] scheduled steps: {self.cfg.steps} "
              f"(10% of {self.cfg.total_stage1_steps})")
        step = 0
        while step < self.cfg.steps:
            for batch in dataloader:
                if step >= self.cfg.steps:
                    break
                stats = self._step(batch)
                if step % max(1, self.cfg.steps // 10) == 0:
                    print(f"  step {step:>5d}/{self.cfg.steps}  "
                          f"loss={stats['loss']:.4f}  "
                          f"trainable={stats['trainable_params']/1e6:.2f}M")
                step += 1

    # ------------------------------------------------------------------ #
    def export_for_inference(self) -> InferenceVLA:
        """Strip the auxiliary decoder branch before shipping the model."""
        self.model.eval()
        return self.model.as_inference_model()


# ---------------------------------------------------------------------------
# 3. Toy data + demo run
# ---------------------------------------------------------------------------

def toy_dataloader(num_batches: int = 4, batch_size: int = 2, d: int = 512):
    """A deterministic iterator that yields enough batches for the demo."""
    for _ in range(num_batches):
        yield {
            "pc_seq":   torch.randn(batch_size, 4, 256, 6),          # 4 frames
            "text_emb": torch.randn(batch_size, 16, d),
            "action":   torch.randn(batch_size, 4, 7),
        }


def main() -> None:
    cfg = Stage2Config(total_stage1_steps=20_000, fraction=0.10)
    print(f"Stage 2 will run for {cfg.steps} steps "
          f"({cfg.fraction*100:.0f}% of Stage 1).")

    model = Lift3DVLA()
    trainer = Stage2Trainer(model, cfg)

    # Sanity check: LLM/LoRA frozen, vision + decoder trainable.
    for name, p in model.named_parameters():
        tag = "TRAIN" if p.requires_grad else "frozen"
        if "vision" in name or "decoder" in name or "llm" in name:
            print(f"  [{tag}] {name}: {tuple(p.shape)}")

    trainer.fit(toy_dataloader(num_batches=cfg.steps))

    inf = trainer.export_for_inference()
    print("\n[Inference] exported module:", type(inf).__name__)
    with torch.no_grad():
        out = inf(torch.randn(1, 16, 512), torch.randn(1, 4, 256, 6))
    print("  output shape:", tuple(out.shape))


if __name__ == "__main__":
    main()