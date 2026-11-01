import torch
import torch.distributions as distributions

from simulusion.models.dreamer import Dreamer, DreamerAgent
from simulusion.utils.dreamer_utils import symlog as _symlog


class DreamerV2(Dreamer):
    def _get_head_config(self) -> tuple[str, dict]:
        return "symlog_twohot", {
            "num_buckets": getattr(self.args, "num_buckets", 255),
            "symlog_range": getattr(self.args, "symlog_range", 10.0),
        }

    def _compute_kl_loss(
        self,
        prior: dict,
        posterior: dict,
        post_dist: distributions.Distribution,
        prior_dist: distributions.Distribution,
    ) -> torch.Tensor:
        post_no_grad = self.rssm.detach_state(posterior)
        prior_no_grad = self.rssm.detach_state(prior)
        post_mean_no_grad, post_std_no_grad = (
            post_no_grad["mean"],
            post_no_grad["std"],
        )
        prior_mean_no_grad, prior_std_no_grad = (
            prior_no_grad["mean"],
            prior_no_grad["std"],
        )

        kl_loss = self.args.kl_alpha * torch.mean(
            distributions.kl.kl_divergence(
                self.rssm.get_dist(post_mean_no_grad, post_std_no_grad),
                prior_dist,
            )
        )
        kl_loss += (1 - self.args.kl_alpha) * torch.mean(
            distributions.kl.kl_divergence(
                post_dist,
                self.rssm.get_dist(prior_mean_no_grad, prior_std_no_grad),
            )
        )
        return kl_loss

    def _get_reward_target(self, rews: torch.Tensor) -> torch.Tensor:
        return _symlog(rews[:-1])

    def _compute_actor_loss(
        self, returns: torch.Tensor, discounts: torch.Tensor
    ) -> torch.Tensor:
        # Dynamics backpropagation (DreamerV2 Sec. 2.4, rho = 0 for continuous
        # actions): the actor maximises the lambda-returns themselves, so they
        # must keep their graph. The two-hot heads already decode rewards and
        # values back to the real scale. Squashing the returns through symlog
        # here (not in the paper) made the actor saturate: on DMC cartpole
        # swing-up it pinned every action at -1 and its gradient decayed to
        # ~1e-8, stalling the return at ~75 while DreamerV1 reached ~500.
        weight = discounts.detach()
        return -torch.mean(weight * returns)

    def _compute_value_loss(
        self,
        value_feat: torch.Tensor,
        value_targ: torch.Tensor,
        discounts: torch.Tensor,
    ) -> torch.Tensor:
        value_dist = self.value_model(value_feat)
        target = _symlog(value_targ)
        log_prob = value_dist.log_prob(target)
        return -torch.mean(discounts * log_prob)


class DreamerV2Agent(DreamerAgent):
    """High-level agent that trains :class:`DreamerV2` (``create_model("dreamer-v2")``)."""

    core_cls = DreamerV2
    algo_name = "Dreamerv2"
