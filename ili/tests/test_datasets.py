from datasets import tinyimagenet

def test_tinyimagenet():
    (x_train, y_train), (x_test, y_test) = tinyimagenet.load_data()
    assert x_train.shape == (100000, 64, 64, 3)
    assert y_train.shape == (100000,)
    assert x_test.shape == (10000, 64, 64, 3)
    assert y_test.shape == (10000,)
