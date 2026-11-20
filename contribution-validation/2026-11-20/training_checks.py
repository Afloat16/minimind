"""Regression checks against the actual trainer functions, without downloading data.

Usage: python training_checks.py CASE REPOSITORY [--baseline BASELINE_REPOSITORY]
Checks live outside the proposed upstream patch because upstream removed tests.
"""
import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import weakref
from contextlib import nullcontext
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

torch.set_num_threads(1)


def functions(repo, filename, namespace=None):
    path = Path(repo) / filename
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
    scope = dict(torch=torch, nn=nn, F=F, time=time, math=math, os=os,
                 DistributedDataParallel=DDP, nullcontext=nullcontext)
    if namespace:
        scope.update(namespace)
    exec(compile(tree, str(path), "exec"), scope)
    return scope


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.arange(16, dtype=torch.float32).reshape(4, 4) / 50)
        self.aux = nn.Parameter(torch.tensor(0.3))
        self.dtypes = []
        self.grad_modes = []
        self.refs = []
        self.before_forward = None

    def forward(self, input_ids, labels=None):
        if self.before_forward is not None:
            self.before_forward()
        logits = F.one_hot(input_ids.long() % 4, 4).float() @ self.weight + self.aux * 0
        self.dtypes.append(logits.dtype)
        self.grad_modes.append(torch.is_grad_enabled())
        self.refs.append(weakref.ref(logits))
        loss = None if labels is None else F.cross_entropy(logits[:, :-1].reshape(-1, 4), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss, aux_loss=self.aux.square())


def setup(repo, trainer, model=None, accumulation=1, save_interval=1000, main=False):
    model = model or TinyLM()
    scope = functions(repo, "trainer/" + trainer + ".py")
    args = SimpleNamespace(device="cpu", epochs=1, learning_rate=0.05, accumulation_steps=accumulation,
                           grad_clip=100, log_interval=1000, save_interval=save_interval,
                           save_dir="unused-fixture", save_weight="test", lora_name="test")
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    scope.update(args=args, model=model, optimizer=optimizer, scaler=torch.amp.GradScaler("cpu", enabled=False),
                 autocast_ctx=nullcontext(), Logger=lambda *a: None, get_lr=lambda *a: args.learning_rate,
                 is_main_process=lambda: main, lm_config=SimpleNamespace(use_moe=False, hidden_size=4),
                 lm_checkpoint=lambda *a, **k: None, save_lora=lambda *a, **k: None)
    return scope


def batches(count=1, offset=0):
    return [(torch.tensor([[i + offset, i + offset + 1, i + offset + 2]]) % 4,
             torch.tensor([[i + offset + 1, i + offset + 2, i + offset + 3]]) % 4)
            for i in range(count)]


def dpo_batches(count=1, offset=0):
    result = []
    for x, y in batches(count, offset):
        result.append(dict(x_chosen=x, y_chosen=y, x_rejected=x.flip(1), y_rejected=y.flip(1),
                           mask_chosen=torch.ones_like(x), mask_rejected=torch.ones_like(x)))
    return result


def run(scope, trainer, loader, total=None, start=0, teacher=None, alpha=0.5, use_moe=False):
    total = total or len(loader)
    if trainer == "train_distillation":
        scope["train_epoch"](0, loader, total, teacher or TinyLM(), SimpleNamespace(use_moe=use_moe, hidden_size=4), start, alpha=alpha)
    elif trainer == "train_dpo":
        teacher = teacher or TinyLM()
        teacher.eval().requires_grad_(False)
        scope["train_epoch"](0, loader, total, teacher, SimpleNamespace(use_moe=False, hidden_size=4), start)
    elif trainer == "train_lora":
        scope["train_epoch"](0, loader, total, list(scope["model"].parameters()), start)
    else:
        scope["train_epoch"](0, loader, total, start)


def router_loss(repo):
    observed = {}
    for alpha in [0.0, 0.5, 1.0]:
        scope = setup(repo, "train_distillation")
        before = scope["model"].aux.detach().clone()
        run(scope, "train_distillation", batches(), alpha=alpha, use_moe=True)
        delta = (before - scope["model"].aux.detach()).item()
        observed[str(alpha)] = delta
        assert abs(delta - 0.03) < 1e-6, f"alpha={alpha}: router update {delta}, expected 0.03"
    print(json.dumps({"aux_parameter_updates": observed}))


def teacher_autocast(repo):
    scope = setup(repo, "train_distillation")
    scope["autocast_ctx"] = torch.autocast("cpu", dtype=torch.bfloat16)
    teacher = TinyLM()
    run(scope, "train_distillation", batches(2), teacher=teacher)
    assert teacher.dtypes == [torch.bfloat16] * 2, str(teacher.dtypes)
    assert teacher.grad_modes == [False] * 2
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.isfinite(p).all() for p in scope["model"].parameters())
    student_logits = torch.tensor([[40000.0, -40000.0]], dtype=torch.float16, requires_grad=True)
    teacher_logits = torch.tensor([[-40000.0, 40000.0]], dtype=torch.float16)
    loss = scope["distillation_loss"](student_logits, teacher_logits, temperature=0.5)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(student_logits.grad).all()
    print(json.dumps({"teacher_output_dtype": "bfloat16", "teacher_grad_disabled": True, "steps": 2,
                      "extreme_fp16_kl": loss.item(), "finite_loss_and_gradient": True}))


def reference_lifetime(repo, baseline_repo):
    scope = setup(repo, "train_dpo")
    teacher = TinyLM()
    observations = []
    def observe():
        observations.append(teacher.refs[-1]() is None)
    scope["model"].before_forward = observe
    run(scope, "train_dpo", dpo_batches(2), teacher=teacher)
    assert observations == [True, True], f"reference logits still live at policy forward: {observations}"
    assert all(torch.isfinite(p).all() for p in scope["model"].parameters())
    original = setup(baseline_repo, "train_dpo")
    run(original, "train_dpo", dpo_batches(2))
    for name, value in scope["model"].state_dict().items():
        torch.testing.assert_close(original["model"].state_dict()[name], value, atol=0, rtol=0)
    print(json.dumps({"reference_logits_freed_before_policy": observations, "steps": 2, "weights_identical": True}))


def checkpoint_boundaries(repo):
    trainers = ["train_pretrain", "train_full_sft", "train_lora", "train_dpo", "train_distillation"]
    observed = {}
    for trainer in trainers:
        loader = dpo_batches(13) if trainer == "train_dpo" else batches(13)
        scope = setup(repo, trainer, accumulation=4, save_interval=5, main=True)
        saved = []
        def checkpoint(*a, **k):
            pending = any(p.grad is not None for p in scope["model"].parameters())
            saved.append((k["step"], copy.deepcopy(scope["model"].state_dict()), copy.deepcopy(scope["optimizer"].state_dict()), pending))
        scope["lm_checkpoint"] = checkpoint
        with mock.patch.object(torch, "save"):
            run(scope, trainer, loader)
        observed[trainer] = [x[0] for x in saved]
        assert [x[0] for x in saved] == [8, 12, 13], observed[trainer]
        assert not any(x[3] for x in saved), "saved during a pending accumulation window"
        uninterrupted = copy.deepcopy(scope["model"].state_dict())
        resumed = setup(repo, trainer, accumulation=4, save_interval=5)
        resumed["model"].load_state_dict(saved[0][1])
        resumed["optimizer"].load_state_dict(saved[0][2])
        run(resumed, trainer, loader[8:], total=13, start=8)
        for key, value in uninterrupted.items():
            torch.testing.assert_close(resumed["model"].state_dict()[key], value, atol=0, rtol=0)
    # Exercise the actual new save predicate for both RL loops as well.
    for trainer in ["train_grpo", "train_agent"]:
        path = Path(repo) / "trainer" / (trainer + ".py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assigns = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "should_save" for t in n.targets)]
        assert len(assigns) == 1, trainer + " must defer periodic saves"
        expr = compile(ast.Expression(assigns[0].value), str(path), "eval")
        steps = [step for step in range(1, 14) if eval(expr, {"step": step, "iters": 13, "args": SimpleNamespace(accumulation_steps=4, save_interval=5)})]
        assert steps == [8, 12, 13], steps
        observed[trainer] = steps
    print(json.dumps({"checkpoint_steps": observed, "resume_matches_uninterrupted": True}))


def cpu_checkpoint(repo):
    calls = []
    tensor_refs = []
    actual_load = torch.load
    class LoadModel(TinyLM):
        def __init__(self, config):
            super().__init__()
        def to(self, device):
            assert all(ref() is None for ref in tensor_refs), "checkpoint tensors retained when moving model"
            assert device == "cuda:0"
            return self
    def load_on_cpu(path, **kwargs):
        calls.append(kwargs.get("map_location"))
        assert kwargs.get("map_location") == "cpu", f"checkpoint staged on {kwargs.get('map_location')}"
        result = actual_load(path, **kwargs)
        tensor_refs.extend(weakref.ref(tensor) for tensor in result.values())
        return result
    scope = functions(repo, "trainer/trainer_utils.py", {"Sampler": torch.utils.data.Sampler,
                      "np": None, "random": None, "dist": dist,
                      "AutoTokenizer": SimpleNamespace(from_pretrained=lambda *a, **k: object()),
                      "MiniMindForCausalLM": LoadModel})
    scope["get_model_params"] = lambda *a: None
    scope["Logger"] = lambda *a: None
    with tempfile.TemporaryDirectory() as tmp:
        original = TinyLM()
        torch.save(original.state_dict(), Path(tmp) / "full_sft_4.pth")
        with mock.patch.object(torch, "load", side_effect=load_on_cpu):
            loaded, _ = scope["init_model"](SimpleNamespace(use_moe=False, hidden_size=4), "full_sft", save_dir=tmp, device="cuda:0")
        for key, value in original.state_dict().items():
            torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
    paths = ["trainer/trainer_utils.py", "trainer/train_ppo.py", "eval_llm.py", "scripts/eval_toolcall.py",
             "scripts/serve_openai_api.py", "model/model_lora.py"]
    load_sites = 0
    for filename in paths:
        tree = ast.parse((Path(repo) / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch" and node.func.attr == "load":
                mapping = next(k.value for k in node.keywords if k.arg == "map_location")
                assert isinstance(mapping, ast.Constant) and mapping.value == "cpu", filename
                load_sites += 1
    assert load_sites == 7
    print(json.dumps({"stage_device": calls, "checkpoint_tensors_released_before_model_transfer": True,
                      "exact_state_roundtrip": True, "cpu_load_sites": load_sites,
                      "gpu_memory_benchmark": "not run"}))








if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("case", choices=["router", "teacher", "reference", "checkpoint", "cpu"])
    parser.add_argument("repo")
    parser.add_argument("--baseline", help="Unmodified upstream checkout used by the reference-logit equivalence check")
    args = parser.parse_args()
    repo = str(Path(args.repo).resolve())
    if args.case == "reference":
        if not args.baseline:
            parser.error("the reference case requires --baseline BASELINE_REPOSITORY")
        reference_lifetime(repo, str(Path(args.baseline).resolve()))
    else:
        {"router": router_loss, "teacher": teacher_autocast,
         "checkpoint": checkpoint_boundaries, "cpu": cpu_checkpoint}[args.case](repo)
