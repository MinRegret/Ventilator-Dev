from lung.controllers.core import Controller


class DummyController(Controller):
    def compute_action(self, state, t):
        return 0, 1


def test_creation():
    dummy = DummyController()

    assert dummy.name = dummy.__class__.__name__
