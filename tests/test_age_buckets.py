"""Tests for the exact age-bucket boundary definitions in scripts/models.py:

    0-17  -> "0-18"
    18-60 -> "18-60"
    61+   -> "60+"
    None  -> "unknown"
"""

from models import (
    AGE_GROUP_0_18,
    AGE_GROUP_18_60,
    AGE_GROUP_60_PLUS,
    AGE_GROUP_UNKNOWN,
    age_to_group,
)


def test_age_group_lower_bound():
    assert age_to_group(0) == AGE_GROUP_0_18


def test_age_group_0_18_upper_boundary():
    # Bucket definitions are exact integer-year boundaries (0-17 -> "0-18");
    # age 17 is the last age in this bucket, 18 already belongs to the next.
    assert age_to_group(17) == AGE_GROUP_0_18
    assert age_to_group(17.9) == AGE_GROUP_18_60


def test_age_group_18_60_lower_boundary():
    assert age_to_group(18) == AGE_GROUP_18_60


def test_age_group_18_60_upper_boundary():
    assert age_to_group(60) == AGE_GROUP_18_60
    assert age_to_group(60.9) == AGE_GROUP_60_PLUS


def test_age_group_60_plus_lower_boundary():
    assert age_to_group(61) == AGE_GROUP_60_PLUS


def test_age_group_60_plus_high_value():
    assert age_to_group(95) == AGE_GROUP_60_PLUS


def test_age_group_none_is_unknown():
    assert age_to_group(None) == AGE_GROUP_UNKNOWN


def test_age_group_negative_is_unknown():
    assert age_to_group(-5) == AGE_GROUP_UNKNOWN
