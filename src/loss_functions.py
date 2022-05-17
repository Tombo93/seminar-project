import torch
from torch.optim import Optimizer
from src.agent import Agent


def ppo_policy_loss(
    actor_critic: Agent,
    clip_ratio: float,
    obs: torch.Tensor,
    act: torch.Tensor,
    adv: torch.Tensor,
    logp_old: torch.Tensor,
) -> torch.Tensor:
    _, logp = actor_critic.policy(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_clip = -(torch.min(ratio * adv, clip_adv)).mean()
    return loss_clip


def ppo_value_loss(
    actor_critic: Agent, obs: torch.Tensor, ret: torch.Tensor
) -> torch.Tensor:
    return ((actor_critic.value_func(obs) - ret) ** 2).mean()


def update_models(
    agent: Agent,
    data: dict,
    pi_optim: Optimizer,
    val_optim: Optimizer,
    update_cycles: int,
    logger=None,
) -> None:
    for _ in range(update_cycles):
        pi_optim.zero_grad()
        pi_loss = ppo_policy_loss(
            agent,
            0.2,
            data["obs"],
            data["act"],
            data["adv"],
            data["logp"],
        )
        pi_loss.backward()
        pi_optim.step()

        val_optim.zero_grad()
        v_loss = ppo_value_loss(agent, data["obs"], data["ret"])
        v_loss.backward()
        val_optim.step()
