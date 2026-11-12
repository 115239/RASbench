"""Standalone single-GPU GR00T training entry matched to this delivery format."""

import argparse, json, os, sys, hashlib
from pathlib import Path

P = Path(__file__).resolve().parents[2] / "restored/VLA/data/grasp_755"
sys.path.insert(0, str(P / "vendor/Isaac-GR00T"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--backbone", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--profile",
        choices=["rebuttal_train"],
        default="rebuttal_train",
    )
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--mix", type=float, nargs=3, default=[1, 1, 1])
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--save-total-limit", type=int, default=2)
    ap.add_argument("--resume", type=Path)
    ap.add_argument("--restart-schedule", action="store_true", help="Keep optimizer moments, start a new learning-rate schedule in a new output directory")
    ap.add_argument("--stop-after", type=int, help="Stop after this global step without changing the learning-rate schedule")
    ap.add_argument("--eval-adapter", type=Path)
    ap.add_argument("--eval-split", default="validation")
    ap.add_argument("--eval-limit", type=int, default=0)
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path)
    known, _ = pre.parse_known_args()
    if known.config:
        recipe = json.loads(known.config.read_text())
        mapping = {
            "learning_rate": "lr",
            "batch_size": "batch",
            "phase_group_weights": "mix",
        }
        ap.set_defaults(
            **{
                mapping.get(k, k): v
                for k, v in recipe.items()
                if k
                in [
                    "profile",
                    "steps",
                    "learning_rate",
                    "batch_size",
                    "seed",
                    "phase_group_weights",
                    "save_total_limit",
                ]
            }
        )
    args = ap.parse_args()
    assert not args.restart_schedule or args.resume is not None
    if args.stop_after is not None:
        assert 0 < args.stop_after <= args.steps
    assert (
        args.steps > 0
        and args.batch > 0
        and args.save_steps > 0
        and args.save_total_limit >= 1
        and args.lr > 0
        and all(x > 0 for x in args.mix)
    )
    assert args.checkpoint.is_dir() and args.backbone.is_dir(), (
        "checkpoint and backbone must be supplied by the contestant"
    )
    status = json.loads((P / "PACKAGE_STATUS.json").read_text())
    assert status["status"] == "READY_FOR_TRAINING", (
        "run verify_package.py before training"
    )
    import torch

    assert torch.cuda.device_count() == 1, (
        "select one free GPU via CUDA_VISIBLE_DEVICES"
    )
    torch.cuda.set_per_process_memory_fraction(0.20)
    from market_dataset import register_modality, make_dataset

    register_modality()
    from gr00t.configs.base_config import get_default_config
    from gr00t.data.dataset.factory import DatasetFactory
    from gr00t.experiment import experiment as engine
    from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
    from transformers import AutoModel, TrainerCallback

    config = get_default_config().load_dict(
        {
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": [str(P)],
                        "mix_ratio": 1.0,
                        "embodiment_tag": "new_embodiment",
                    }
                ],
            }
        }
    )
    config.load_config_path = None
    config.model.model_name = str(args.backbone.resolve())
    config.model.tune_llm = config.model.tune_visual = config.model.tune_vlln = False
    config.model.tune_projector = config.model.tune_diffusion_model = True
    weights = [1.0] * 9 + [3.0]
    config.model.action_channel_weights = weights
    config.model.action_horizon = 40
    config.model.apply_sincos_state_encoding = False
    config.model.use_percentiles = config.model.use_relative_action = (
        config.model.use_mean_std
    ) = False
    config.model.image_crop_size = config.model.image_target_size = None
    config.model.shortest_image_edge = 256
    config.model.crop_fraction = 0.95
    t = config.training
    t.start_from_checkpoint = str((args.resume or args.checkpoint).resolve())
    t.resume_from_checkpoint = bool(args.resume)
    t.output_dir = str(args.output.resolve())
    t.experiment_name = None
    t.max_steps = args.steps
    t.warmup_steps = min(100, args.steps)
    t.learning_rate = args.lr
    t.weight_decay = 0.0
    t.global_batch_size = 1
    t.gradient_accumulation_steps = args.batch
    t.bf16 = True
    t.fp16 = False
    t.lr_scheduler_type = "cosine"
    t.dataloader_num_workers = 0
    t.num_gpus = 1
    t.use_ddp = t.use_wandb = False
    t.optim = "adamw_torch"
    t.save_only_model = False
    t.save_steps = args.save_steps
    t.save_total_limit = args.save_total_limit
    t.logging_steps = 10
    t.transformers_local_files_only = True
    config.data.seed = args.seed
    config.data.override_pretraining_statistics = True
    config.data.ds_weights_alpha = None
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {
        "profile": args.profile,
        "index_sha256": hashlib.sha256(
            (P / "meta" / f"{args.profile}.jsonl").read_bytes()
        ).hexdigest(),
        "statistics_sha256": hashlib.sha256(
            (P / "meta" / f"{args.profile}_statistics.json").read_bytes()
        ).hexdigest(),
        "batch": args.batch,
        "seed": args.seed,
        "mix": args.mix,
        "lr": args.lr,
        "schedule_steps": args.steps,
    }
    if args.resume:
        from resume_contract import validate_resume
        old = json.loads((args.resume / "dataset_binding.json").read_text())
        old_step = json.loads((args.resume / "trainer_state.json").read_text())["global_step"]
        binding = validate_resume(old, binding, old_step, args.restart_schedule)
        if args.stop_after is not None:
            assert args.stop_after > old_step
        if args.restart_schedule:
            assert args.output.resolve() != args.resume.resolve().parent
            assert not any(args.output.glob("checkpoint-*")), "schedule restart requires a new run"
        (args.output / "RESUME_ORIGIN.json").write_text(json.dumps({
            "checkpoint": str(args.resume.resolve()), "step": old_step,
            "optimizer_moments_preserved": True,
            "schedule_restarted": args.restart_schedule,
            "scheduler_origin": binding.get("scheduler_origin", 0),
            "target_step": args.steps,
        }, indent=2))
        assert all(
            (args.resume / n).is_file()
            for n in [
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "trainer_state.json",
            ]
        )
    elif any(args.output.glob("checkpoint-*")):
        raise ValueError("use a new output directory or --resume")
    refs = {}

    def create(pipeline):
        model, info = AutoModel.from_pretrained(
            t.start_from_checkpoint,
            model_name=config.model.model_name,
            tune_llm=False,
            tune_visual=False,
            tune_vlln=False,
            tune_projector=True,
            tune_diffusion_model=True,
            state_dropout_prob=config.model.state_dropout_prob,
            backbone_trainable_params_fp32=config.model.backbone_trainable_params_fp32,
            load_bf16=config.model.load_bf16,
            transformers_loading_kwargs=pipeline.transformers_loading_kwargs,
            output_loading_info=True,
            **pipeline.transformers_loading_kwargs,
        )
        assert not any(
            info.get(k) for k in ["missing_keys", "unexpected_keys", "mismatched_keys"]
        ), info
        from groot_lora import install
        adapter_info = install(model)
        (args.output / "adapter_config.json").write_text(json.dumps(adapter_info, indent=2))
        model.config.action_channel_weights = weights
        model.action_head.config.action_channel_weights = weights
        trained = [n for n, p in model.named_parameters() if p.requires_grad]
        assert trained and not any(
            "backbone" in n or "vlln" in n or "vl_self_attention" in n for n in trained
        )
        (args.output / "freeze_check.json").write_text(
            json.dumps({"trainable_names": trained, "weights": weights}, indent=2)
        )
        # Recompute action-transformer activations to fit shared GPU memory.
        import re
        from torch.utils.checkpoint import checkpoint
        wrapped_blocks = []
        for name, module in model.named_modules():
            if name.startswith("action_head") and re.search(r"transformer_blocks\.\d+$", name):
                original_forward = module.forward
                def recompute(*inputs, _original=original_forward, **kwargs):
                    if torch.is_grad_enabled():
                        return checkpoint(_original, *inputs, use_reentrant=False, **kwargs)
                    return _original(*inputs, **kwargs)
                module.forward = recompute
                wrapped_blocks.append(name)
        assert wrapped_blocks, "no activation checkpoint blocks found"
        (args.output / "activation_checkpointing.json").write_text(json.dumps(wrapped_blocks, indent=2))
        return model

    def build(factory, processor):
        ds = make_dataset(P, processor, args.profile, args.mix, args.seed)
        refs["dataset"] = ds
        return ds, None

    class Callback(TrainerCallback):
        def on_step_end(self, hf_args, state, control, **kwargs):
            import time
            progress = {
                "global_step": state.global_step,
                "target_steps": args.steps,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "physical_acceptance": False,
            }
            temporary = args.output / "PROGRESS.json.tmp"
            temporary.write_text(json.dumps(progress, indent=2))
            temporary.replace(args.output / "PROGRESS.json")
            if state.global_step >= args.steps:
                control.should_save = True
            if args.stop_after is not None and state.global_step >= args.stop_after:
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_log(self, hf_args, state, control, logs=None, **kwargs):
            import math
            for key in ("loss", "grad_norm"):
                if key in (logs or {}) and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"nonfinite training {key}")
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps({"step": state.global_step, **(logs or {})}) + "\n")

        def on_save(self, hf_args, state, control, **kwargs):
            ck = args.output / f"checkpoint-{state.global_step}"
            (ck / "dataset_binding.json").write_text(json.dumps(binding, indent=2))
            (ck / "market_statistics.json").write_text(
                json.dumps(refs["dataset"].stats, indent=2)
            )

    class Trainer(engine.Gr00tTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.add_callback(Callback())

        def create_scheduler(self, num_training_steps, optimizer=None):
            if "scheduler_origin" not in binding:
                return super().create_scheduler(num_training_steps, optimizer)
            from transformers import get_scheduler
            if self.lr_scheduler is None:
                remaining = args.steps - binding["scheduler_origin"]
                self.lr_scheduler = get_scheduler(
                    self.args.lr_scheduler_type, optimizer or self.optimizer,
                    num_warmup_steps=min(50, remaining), num_training_steps=remaining,
                )
                self._created_lr_scheduler = True
            return self.lr_scheduler

        def _load_optimizer_and_scheduler(self, checkpoint):
            super()._load_optimizer_and_scheduler(checkpoint)
            if args.restart_schedule:
                for group in self.optimizer.param_groups:
                    group["lr"] = group["initial_lr"] = args.lr
                self.lr_scheduler = None
                self.create_scheduler(args.steps, self.optimizer)

        def get_train_dataloader(self):
            self.args.ignore_data_skip = True
            self.train_dataset.start_sample = int(self.state.global_step) * args.batch
            return torch.utils.data.DataLoader(
                self.train_dataset,
                batch_size=1,
                collate_fn=self._get_collator_with_removed_columns(
                    self.data_collator, description="training"
                ),
                num_workers=0,
                pin_memory=True,
                generator=torch.Generator().manual_seed(args.seed),
            )

        def train(self, resume_from_checkpoint=None, **kw):
            return super().train(
                resume_from_checkpoint=str(args.resume) if args.resume else None, **kw
            )

        def save_model(self, output_dir=None, _internal_call=False):
            if output_dir is None:
                ck = args.output / f"checkpoint-{self.state.global_step}"
                if ck.exists():
                    (args.output / "FINAL_CHECKPOINT.json").write_text(
                        json.dumps({"checkpoint": ck.name})
                    )
                    return
            from safetensors.torch import save_file
            target = Path(output_dir or args.output)
            target.mkdir(parents=True, exist_ok=True)
            save_file({n: p.detach().cpu().contiguous() for n, p in self.model.named_parameters() if p.requires_grad}, str(target / "adapter.safetensors"))
            (target / "ADAPTER_CHECKPOINT.json").write_text(json.dumps({"base": str(args.checkpoint), "adapter_config": str(args.output / "adapter_config.json"), "step": self.state.global_step, "requires_custom_adapter_loader": True}, indent=2))

    DatasetFactory.build = build
    Gr00tN1d7Pipeline._create_model = create
    engine.Gr00tTrainer = Trainer
    (args.output / "dataset_binding.json").write_text(json.dumps(binding, indent=2))
    if args.eval_adapter:
        import numpy as np
        from safetensors.torch import load_file
        from gr00t.data.types import VLAStepData
        from gr00t.data.embodiment_tags import EmbodimentTag
        from market_dataset import STATE_GROUPS, ACTION_GROUPS
        from offline_eval import evaluate
        save_cfg = args.output / "experiment_cfg"
        save_cfg.mkdir(parents=True, exist_ok=True)
        config.validate()
        pipeline = Gr00tN1d7Pipeline(config, save_cfg)
        pipeline.setup()
        model = pipeline.return_model()
        values = load_file(str(args.eval_adapter))
        params = dict(model.named_parameters())
        assert set(values) == {n for n,p in params.items() if p.requires_grad}
        with torch.no_grad():
            for n,v in values.items(): params[n].copy_(v)
        model.eval().cuda()
        processor = pipeline.return_processor(); processor.eval()
        collate = pipeline.return_collator()
        def predict(r, i):
            torch.manual_seed(20261111+i)
            states = {k:r["state"][None,lo:hi] for k,(lo,hi) in STATE_GROUPS.items()}
            data = VLAStepData(images={k:[v] for k,v in r["images"].items()},states=states,actions={},text=r["prompt"],embodiment=EmbodimentTag.NEW_EMBODIMENT)
            batch=collate([processor([{"type":"episode_step","content":data}])])
            def move(x):
                if isinstance(x,dict):return {k:move(v) for k,v in x.items()}
                if isinstance(x,list):return [move(v) for v in x]
                if torch.is_tensor(x):return x.to(device="cuda",dtype=torch.bfloat16 if x.is_floating_point() else x.dtype)
                return x
            batch=move(batch)
            with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16): pred=model.get_action(**batch)["action_pred"].float().cpu().numpy()
            decoded=processor.decode_action(pred,EmbodimentTag.NEW_EMBODIMENT,{k:v[None] for k,v in states.items()})
            return np.concatenate([decoded[k][0] for k in ACTION_GROUPS],axis=-1)
        evaluate(predict,args.eval_split,args.output/"evaluation",args.eval_limit)
    else:
        engine.run(config)


if __name__ == "__main__":
    main()
