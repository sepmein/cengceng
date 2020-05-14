from cengceng import Model


def test_model_construction():
    model = Model()
    assert type(model, Model)


if __name__ == "__main__":
    test_model_construction()
    print("Everything passed")
