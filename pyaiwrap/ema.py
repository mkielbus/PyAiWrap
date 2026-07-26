from typing import Any, Dict, Iterator, Set, Tuple

import torch
import torch.nn as nn


class WeightEma:
    """Exponential moving average of a model's weights, maintained alongside training.

    The training model is never perturbed by the averaging. ``update`` folds the live
    weights into a shadow copy; ``applyTo``/``restore`` temporarily swap the shadow into
    the model (for validation, visualisation and checkpointing) and then put the raw
    training weights back so the next optimisation step continues from them.

    Floating-point parameters and buffers (e.g. BatchNorm ``running_mean``/``running_var``)
    are averaged; integer buffers (``num_batches_tracked``) and frozen parameters are copied
    verbatim so the shadow stays a complete, loadable ``state_dict``.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_updates: int = 0) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        if warmup_updates < 0:
            raise ValueError(f"EMA warmup_updates must be >= 0, got {warmup_updates}")

        self._decay: float = decay
        self._warmup_updates: int = warmup_updates
        self._num_updates: int = 0
        self._frozen_names: Set[str] = {
            name for name, param in model.named_parameters() if not param.requires_grad
        }
        self._shadow: Dict[str, torch.Tensor] = {
            name: tensor.detach().clone() for name, tensor in self._trackedTensors(model)
        }
        self._backup: Dict[str, torch.Tensor] = {}

    @staticmethod
    def _trackedTensors(model: nn.Module) -> Iterator[Tuple[str, torch.Tensor]]:
        """Yield every persistent tensor of the model (parameters first, then buffers)."""
        yield from model.named_parameters()
        yield from model.named_buffers()

    def _currentDecay(self) -> float:
        """Ramp the decay from 0 to its target over the first ``warmup_updates`` steps.

        A cold shadow starts equal to the randomly initialised weights; ramping the decay
        lets those be forgotten quickly instead of lingering in the average for many epochs.
        """
        if self._warmup_updates == 0 or self._num_updates >= self._warmup_updates:
            return self._decay
        return self._decay * (self._num_updates / self._warmup_updates)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """Fold the model's current weights into the shadow average (call once per step)."""
        self._num_updates += 1
        decay = self._currentDecay()
        for name, tensor in self._trackedTensors(model):
            shadow = self._shadow[name]
            if tensor.is_floating_point() and name not in self._frozen_names:
                shadow.mul_(decay).add_(tensor.detach(), alpha=1.0 - decay)
            else:
                shadow.copy_(tensor)

    @torch.no_grad()
    def applyTo(self, model: nn.Module) -> None:
        """Swap the averaged weights into the model, stashing the raw weights for ``restore``."""
        self._backup = {
            name: tensor.detach().clone() for name, tensor in self._trackedTensors(model)
        }
        for name, tensor in self._trackedTensors(model):
            tensor.copy_(self._shadow[name])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        """Undo ``applyTo``, putting the raw training weights back into the model."""
        if not self._backup:
            return
        for name, tensor in self._trackedTensors(model):
            tensor.copy_(self._backup[name])
        self._backup = {}

    @torch.no_grad()
    def reset(self, model: nn.Module) -> None:
        """Re-seed the shadow from the model's current weights.

        Used when resuming a run whose checkpoint predates EMA, so the average starts from
        the restored training weights rather than a stale random initialisation.
        """
        self._num_updates = 0
        self._shadow = {
            name: tensor.detach().clone() for name, tensor in self._trackedTensors(model)
        }
        self._backup = {}

    def emaStateDict(self) -> Dict[str, torch.Tensor]:
        """Return the averaged weights as a ``state_dict`` loadable via ``load_state_dict``."""
        return {name: tensor.detach().clone() for name, tensor in self._shadow.items()}

    def stateDict(self) -> Dict[str, Any]:
        """Serialise the full EMA state for resumable checkpoints."""
        return {
            "decay": self._decay,
            "warmup_updates": self._warmup_updates,
            "num_updates": self._num_updates,
            "shadow": self._shadow
        }

    def loadStateDict(self, state: Dict[str, Any]) -> None:
        """Restore EMA state produced by ``stateDict`` (weights and update counter)."""
        self._decay = state["decay"]
        self._warmup_updates = state.get("warmup_updates", self._warmup_updates)
        self._num_updates = state["num_updates"]
        self._shadow = state["shadow"]
        self._backup = {}
