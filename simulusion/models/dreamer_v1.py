from simulusion.models.dreamer import Dreamer, DreamerAgent


class DreamerV1(Dreamer):
    pass


class DreamerV1Agent(DreamerAgent):
    """High-level agent that trains :class:`DreamerV1` (``create_model("dreamer-v1")``)."""

    core_cls = DreamerV1
    algo_name = "Dreamerv1"
