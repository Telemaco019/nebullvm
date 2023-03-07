from surfer.runner import util


def test_get_requirements():
    requirements = util.get_requirements()
    assert len(requirements) > 0
