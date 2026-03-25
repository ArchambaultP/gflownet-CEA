"""
Logger subclass that uploads checkpoints as W&B Artifacts.

Drop-in replacement for gflownet.utils.logger.Logger. Overrides
save_checkpoint to also log the saved .ckpt file as a versioned
wandb artifact, linked to the current run.

Usage in config:
  logger:
    _target_: gflownet.utils.logger_artifacts.ArtifactLogger
    # ... same parameters as Logger ...
    artifact_ckpts: true          # enable/disable artifact uploads
    artifact_type: model          # wandb artifact type
"""

import torch

from gflownet.utils.logger import Logger


class ArtifactLogger(Logger):
    """
    Extends Logger to upload checkpoints as W&B Artifacts.

    Each checkpoint is logged as a versioned artifact named
    ``ckpt-<wandb_run_id>``, with metadata capturing the training step,
    and config hyperparameters (seed, lr, step_fraction, beta) for easy
    filtering in the W&B UI.

    Parameters
    ----------
    artifact_ckpts : bool
        Whether to upload checkpoints as artifacts. Default: True.
    artifact_type : str
        The wandb artifact type string. Default: "model".
    **kwargs
        All other arguments are forwarded to Logger.__init__.
    """

    def __init__(
        self,
        config: dict,
        do: dict,
        project_name: str,
        logdir: dict,
        lightweight: bool,
        debug: bool,
        run_name=None,
        run_name_date: bool = True,
        run_name_job: bool = True,
        run_id: str = None,
        tags: list = None,
        context: str = "0",
        notes: str = None,
        entity: str = None,
        progressbar: dict = {"skip": False, "n_iters_mean": 100},
        is_resumed: bool = False,
        artifact_ckpts: bool = True,
        artifact_type: str = "model",
    ):
        super().__init__(
            config=config,
            do=do,
            project_name=project_name,
            logdir=logdir,
            lightweight=lightweight,
            debug=debug,
            run_name=run_name,
            run_name_date=run_name_date,
            run_name_job=run_name_job,
            run_id=run_id,
            tags=tags,
            context=context,
            notes=notes,
            entity=entity,
            progressbar=progressbar,
            is_resumed=is_resumed,
        )
        self.artifact_ckpts = artifact_ckpts
        self.artifact_type = artifact_type

    def save_checkpoint(
        self,
        forward_policy,
        backward_policy,
        state_flow,
        logZ,
        optimizer,
        buffer,
        step: int,
        final: bool = False,
    ):
        # Save checkpoint to disk (original behavior)
        super().save_checkpoint(
            forward_policy,
            backward_policy,
            state_flow,
            logZ,
            optimizer,
            buffer,
            step,
            final,
        )

        # Upload as wandb artifact
        if not self.artifact_ckpts or not self.do.online:
            return

        ckpt_id = "final" if final else "iter_{:06d}".format(step)
        ckpt_path = self.ckpts_dir / (ckpt_id + ".ckpt")

        if not ckpt_path.exists():
            return

        # Build artifact metadata from config
        cfg = self.config
        metadata = {"step": step, "final": final}
        try:
            metadata["seed"] = cfg.gflownet.seed
        except Exception:
            pass
        try:
            metadata["lr"] = cfg.gflownet.optimizer.lr
        except Exception:
            pass
        try:
            metadata["lr_z_mult"] = cfg.gflownet.optimizer.lr_z_mult
        except Exception:
            pass
        try:
            metadata["step_fraction"] = cfg.env.step_fraction
        except Exception:
            pass
        try:
            metadata["beta"] = cfg.proxy.beta
        except Exception:
            pass
        try:
            metadata["random_action_prob"] = cfg.gflownet.random_action_prob
        except Exception:
            pass

        # Name artifact after the run so all versions are grouped together
        artifact_name = f"ckpt-{self.run.id}"
        artifact = self.wandb.Artifact(
            name=artifact_name,
            type=self.artifact_type,
            metadata=metadata,
        )
        artifact.add_file(str(ckpt_path), name=ckpt_id + ".ckpt")

        # Add aliases: always "latest", plus "final" for the last checkpoint
        aliases = ["latest"]
        if final:
            aliases.append("final")

        try:
            logged = self.run.log_artifact(artifact, aliases=aliases)
            # Wait for the upload to complete before wandb.finish() can close
            logged.wait()
            print(f"  [wandb] Uploaded artifact {artifact_name} "
                  f"({ckpt_path.stat().st_size} bytes, step={step})")
        except Exception as e:
            print(f"  [wandb] ERROR uploading artifact: {e}")