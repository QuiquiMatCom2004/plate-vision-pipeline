import pytest
from pydantic import ValidationError

from cv_agent_pipeline.schema import Diet, FoodItem, PlateAnalysis


def test_plate_analysis_valid(plate_kwargs):
    plate = PlateAnalysis(**plate_kwargs)
    assert plate.meal_type_guess == "lunch"
    assert plate.dietary_flags == [Diet.HIGH_PROTEIN]


def test_total_calories_computed(plate_kwargs, valid_foot_item):
    plate_kwargs["food_items"] = [valid_foot_item, valid_foot_item]
    plate = PlateAnalysis(**plate_kwargs)
    assert plate.total_calories == valid_foot_item.calories_estimate * 2


def test_negative_calories_rejected(valid_macros):
    with pytest.raises(ValidationError):
        FoodItem(
            name="Hawaiian pizza",
            coco_class="pizza",
            portion_estimate_g=500,
            calories_estimate=-100,
            macros=valid_macros,
        )


def test_balanced_score_out_of_range(plate_kwargs):
    plate_kwargs["balanced_score"] = 1.5
    with pytest.raises(ValidationError):
        PlateAnalysis(**plate_kwargs)


def test_diet_flags_serialize_as_string(plate_kwargs):
    plate = PlateAnalysis(**plate_kwargs)
    dumped = plate.model_dump(mode="json")
    assert dumped["dietary_flags"] == ["high protein"]


def test_invalid_meal_type_rejected(plate_kwargs):
    plate_kwargs["meal_type_guess"] = "brunch"
    with pytest.raises(ValidationError):
        PlateAnalysis(**plate_kwargs)
