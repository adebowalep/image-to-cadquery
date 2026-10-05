"""Tests for the provided evaluation metrics (metrics/valid_syntax_rate.py, metrics/best_iou.py)."""

from metrics.best_iou import get_iou_best
from metrics.valid_syntax_rate import evaluate_syntax_rate_simple

VALID_BOX = 'result = cq.Workplane("XY").box(10, 10, 10)'
VALID_BOX_TALL = 'result = cq.Workplane("XY").box(10, 10, 40)'
SYNTAX_ERROR = 'result = cq.Workplane("XY").box(10, 10, 10'
NO_CQ_OBJECT = "x = 1 + 1"


def test_valid_syntax_rate_all_valid():
    codes = {"a": VALID_BOX, "b": VALID_BOX_TALL}
    assert evaluate_syntax_rate_simple(codes) == 1.0


def test_valid_syntax_rate_mixed():
    codes = {"a": VALID_BOX, "b": SYNTAX_ERROR, "c": NO_CQ_OBJECT}
    assert evaluate_syntax_rate_simple(codes) == 1 / 3


def test_valid_syntax_rate_empty():
    assert evaluate_syntax_rate_simple({}) == 0.0


def test_best_iou_identical_shapes_is_near_one():
    iou = get_iou_best(VALID_BOX, VALID_BOX)
    assert iou > 0.95


def test_best_iou_different_shapes_is_lower_than_identical():
    same = get_iou_best(VALID_BOX, VALID_BOX)
    different = get_iou_best(VALID_BOX, VALID_BOX_TALL)
    assert different < same
