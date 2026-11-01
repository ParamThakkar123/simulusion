import torch
from simulusion.utils.dreamer_utils import (
    get_parameters,
    FreezeParameters,
)


class TestDreamerUtils:
    def test_get_parameters(self):
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(5, 3)

        params = get_parameters([model1, model2])

        assert len(params) == 4

    def test_get_parameters_empty(self):
        params = get_parameters([])

        assert len(params) == 0


class TestFreezeParameters:
    def test_freeze_parameters_context(self):
        model = torch.nn.Linear(10, 5)

        assert model.weight.requires_grad

        with FreezeParameters([model]):
            assert not model.weight.requires_grad

        assert model.weight.requires_grad

    def test_freeze_parameters_nested(self):
        model = torch.nn.Linear(10, 5)

        with FreezeParameters([model]):
            assert not model.weight.requires_grad

            with FreezeParameters([model]):
                assert not model.weight.requires_grad

            assert not model.weight.requires_grad

        assert model.weight.requires_grad

    def test_freeze_multiple_modules(self):
        model1 = torch.nn.Linear(10, 5)
        model2 = torch.nn.Linear(5, 3)

        with FreezeParameters([model1, model2]):
            assert not model1.weight.requires_grad
            assert not model2.weight.requires_grad

        assert model1.weight.requires_grad
        assert model2.weight.requires_grad
