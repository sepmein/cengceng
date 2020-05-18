import torch
from cengceng import Model, Sir


def test_model_construction():
    model = Model()
    assert isinstance(model, Model)


def test_sir_model_is_the_instance_of_model_class():
    model = Sir()
    assert isinstance(model, Model)
    assert isinstance(model, Sir)


def test_sir_model():
    model = Sir()
    assert isinstance(model.beta, torch.Tensor)


if __name__ == "__main__":
    test_model_construction()
    print("Everything passed")
