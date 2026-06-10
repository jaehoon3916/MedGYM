"""SFT trainer: LoRA-fine-tune the policy to imitate the reward-oracle's golden actions.

Loss = NLL of the golden action given the policy prompt = -policy.action_logprob(prompt_ids, action_ids).
Reuses the trainable LocalQwenPolicy (LoRA + action_logprob + save).
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any


class SFTTrainer:
    def __init__(self, policy, config: dict[str, Any]):
        import torch

        self.policy = policy
        self.cfg = config
        s = config.get("sft", {})
        self.epochs = int(s.get("epochs", 3))
        self.bs = int(s.get("batch_size", 4))
        self.lr = float(s.get("learning_rate", 1e-4))
        self.save_every = int(s.get("save_every", 1))
        self.save_dir = Path(config.get("experiment", {}).get("output_dir", "outputs")) / "sft"
        self.tok = policy._tokenizer
        params = [p for p in policy._model.parameters() if p.requires_grad]
        self.optim = torch.optim.AdamW(params, lr=self.lr)
        self.torch = torch

    def _encode(self, messages, action):
        prompt_ids = self.tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, enable_thinking=False,
        )
        action_ids = self.tok.encode(action, add_special_tokens=False)
        if self.tok.eos_token_id is not None:
            action_ids = action_ids + [self.tok.eos_token_id]
        return prompt_ids, action_ids

    def train(self, data: list[dict]) -> None:
        for ep in range(self.epochs):
            random.shuffle(data)
            tot, nb = 0.0, 0
            for i in range(0, len(data), self.bs):
                batch = data[i:i + self.bs]
                self.optim.zero_grad()
                loss = None
                n = 0
                for ex in batch:
                    pid, aid = self._encode(ex["messages"], ex["action"])
                    if not aid:
                        continue
                    nll = -self.policy.action_logprob(pid, aid)
                    loss = nll if loss is None else loss + nll
                    n += 1
                if loss is None:
                    continue
                (loss / n).backward()
                self.optim.step()
                tot += float(loss) / n
                nb += 1
            print(f"[sft epoch {ep}] mean_nll={tot / max(1, nb):.4f}")
            if (ep + 1) % self.save_every == 0:
                self.policy.save(str(self.save_dir / f"epoch_{ep + 1}"))

        self.policy.save(str(self.save_dir / "final"))
        print(f"SFT done. Final adapter → {self.save_dir / 'final'}")
