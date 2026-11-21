import math

import pytest

import camera

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return camera.Camera(60.0, 0.1, 100.0)


def _check_center():
    p = _c().project((0.0, 0.0, -10.0))
    return p is not None and abs(p[0]) < 1e-9


def _check_near_clip():
    return _c().project((0.0, 0.0, 0.0)) is None


def _check_far_clip():
    return _c().project((0.0, 0.0, -200.0)) is None


def _check_behind():
    return _c().project((0.0, 0.0, 10.0)) is None


def _check_cull():
    pts = [(0.0, 0.0, -10.0), (0.0, 0.0, 10.0)]
    return _c().cull(pts) == [0]


def _check_cull_empty():
    return _c().cull([]) == []


def _check_bad_fov():
    return _raises(lambda: camera.Camera(0.0, 0.1, 100.0))


def _check_bad_range():
    return _raises(lambda: camera.Camera(60.0, 10.0, 1.0))


def _check_type():
    return isinstance(_c().project((0.0, 0.0, -10.0)), tuple)


def _check_det():
    return _c().project((1.0, 1.0, -5.0)) == _c().project((1.0, 1.0, -5.0))


def _check_ndc_range():
    p = _c().project((0.0, 0.0, -1.0))
    return -1.0 <= p[0] <= 1.0 and -1.0 <= p[1] <= 1.0


def _check_bad_point():
    return _raises(lambda: _c().project((0.0, 0.0)))


def _check_two():
    return len(_c().cull([(0.0, 0.0, -5.0), (0.0, 0.0, -6.0)])) == 2


def _check_scale():
    a = _c().project((1.0, 0.0, -10.0))[0]
    b = _c().project((1.0, 0.0, -20.0))[0]
    return b < a


def _check_no_state():
    c = _c()
    c.project((0.0, 0.0, -5.0))
    return c.project((0.0, 0.0, -5.0)) == c.project((0.0, 0.0, -5.0))

CASES = [("center", "_check_center"), ("near_clip", "_check_near_clip"), ("far_clip", "_check_far_clip"), ("behind", "_check_behind"), ("cull", "_check_cull"), ("cull_empty", "_check_cull_empty"), ("bad_fov", "_check_bad_fov"), ("bad_range", "_check_bad_range"), ("type", "_check_type"), ("det", "_check_det"), ("ndc_range", "_check_ndc_range"), ("bad_point", "_check_bad_point"), ("two", "_check_two"), ("scale", "_check_scale"), ("no_state", "_check_no_state")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    assert globals()[check]() is True
