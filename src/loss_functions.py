import torch
from torch.optim import Optimizer
from src.agent import Agent


def policy_loss(
    actor_critic,
    clip_ratio: float,
    obs: torch.Tensor,
    act: torch.Tensor,
    adv: torch.Tensor,
    logp_old: torch.Tensor,
) -> torch.Tensor:
    _, logp = actor_critic.policy(obs, act)
    ratio = torch.exp(logp - logp_old)
    clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv
    loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()
    return loss_pi


def value_loss(actor_critic, obs: torch.Tensor, ret: torch.Tensor) -> torch.Tensor:
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
        pi_loss = policy_loss(
            actor_critic=agent,
            clip_ratio=0.2,
            obs=data["obs"],
            act=data["act"],
            adv=data["adv"],
            logp_old=data["logp"],
        )
        pi_loss.backward()
        pi_optim.step()

        val_optim.zero_grad()
        v_loss = value_loss(agent=agent, obs=data["obs"], ret=data["ret"])
        v_loss.backward()
        val_optim.step()
