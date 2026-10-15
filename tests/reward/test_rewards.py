import torch
from simulusion.reward.dreamer_v1_reward import RewardModel
from simulusion.reward.dreamer_v1_value import ValueModel


class TestRewardModel:
    def test_init(self):
        model = RewardModel(
            belief_size=200, state_size=30, hidden_size=100, activation_function="relu"
        )

        assert model.fc1.in_features == 230
        assert model.fc3.out_features == 1

    def test_forward(self):
        model = RewardModel(belief_size=200, state_size=30, hidden_size=100)

        belief = torch.randn(4, 200)
        state = torch.randn(4, 30)

        reward = model(belief, state)

        assert reward.shape == (4,)

    def test_forward_single_sample(self):
        model = RewardModel(belief_size=200, state_size=30, hidden_size=100)

        belief = torch.randn(1, 200)
        state = torch.randn(1, 30)

        reward = model(belief, state)

        assert reward.shape == (1,)

    def test_different_activation(self):
        model = RewardModel(
            belief_size=200, state_size=30, hidden_size=100, activation_function="tanh"
        )

        belief = torch.randn(4, 200)
        state = torch.randn(4, 30)

        reward = model(belief, state)

        assert reward.shape == (4,)


class TestValueModel:
    def test_init(self):
        model = ValueModel(
            belief_size=200, state_size=30, hidden_size=100, activation_function="relu"
        )

        assert model.fc1.in_features == 230
        assert model.fc4.out_features == 1

    def test_forward(self):
        model = ValueModel(belief_size=200, state_size=30, hidden_size=100)

        belief = torch.randn(4, 200)
        state = torch.randn(4, 30)

        value = model(belief, state)

        assert value.shape == (4,)

    def test_forward_single_sample(self):
        model = ValueModel(belief_size=200, state_size=30, hidden_size=100)

        belief = torch.randn(1, 200)
        state = torch.randn(1, 30)

        value = model(belief, state)

        assert value.shape == (1,)

    def test_different_activation(self):
        model = ValueModel(
            belief_size=200, state_size=30, hidden_size=100, activation_function="tanh"
        )

        belief = torch.randn(4, 200)
        state = torch.randn(4, 30)

        value = model(belief, state)

        assert value.shape == (4,)
