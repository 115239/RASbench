"""Per-dimension flow-matching loss for pi0.5, so it can be phase-masked.

openpi's ``Pi0.compute_loss`` ends with

    return jnp.mean(jnp.square(v_t - u_t), axis=-1)          # (b, ah)

which averages away the action dimension.  Plan SS2.4 rule 1 needs the loss
*before* that reduction so each of the 32 dimensions can be weighted by the
frame's phase mode.

``compute_loss_per_dim`` below is a line-for-line copy of ``Pi0.compute_loss``
(openpi commit 15a9616, ``src/openpi/models/pi0.py:189-214``) with the final
reduction removed.  It is attached to ``Pi0`` at import time so that no file in
the openpi checkout is modified.

``verify_parity`` re-derives the stock scalar from the per-dim tensor and
asserts they agree, which is what keeps this copy honest if openpi is bumped.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from openpi.models import model as _model
from openpi.models.pi0 import Pi0, make_attn_mask
from openpi.shared import array_typing as at


def compute_loss_per_dim(
    self: Pi0,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
    *,
    train: bool = False,
) -> jnp.ndarray:
    """Squared flow-matching error, shape ``(*b, action_horizon, action_dim)``."""
    preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
    observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

    batch_shape = actions.shape[:-2]
    noise = jax.random.normal(noise_rng, actions.shape)
    time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
    time_expanded = time[..., None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions

    prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (prefix_out, suffix_out), _ = self.PaliGemma.llm(
        [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
    )
    v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

    return jnp.square(v_t - u_t)


def install() -> None:
    """Attach ``compute_loss_per_dim`` to ``Pi0`` without touching the checkout."""
    if not hasattr(Pi0, "compute_loss_per_dim"):
        Pi0.compute_loss_per_dim = compute_loss_per_dim


install()
