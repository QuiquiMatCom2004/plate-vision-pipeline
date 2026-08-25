import pytest

from plate_vision_pipeline.schema import Diet, FoodItem, Macros

@pytest.fixture
def valid_macros():
    return Macros(protein_g=10.0, carbs_g=30.3, fat_g=7.0)

@pytest.fixture
def valid_foot_item(valid_macros):
    return FoodItem(
        name="Hawaiian pizza",
        coco_class="pizza",
        portion_estimate_g=500,
        macros=valid_macros,
        calories_estimate= 1200
    )

@pytest.fixture
def plate_kwargs(valid_foot_item, valid_macros):
    """Campos mínimos válidos de PlateAnalysis. Los tests de rechazo parten
    de este dict y mutan un solo campo para aislar la causa del fallo."""
    return dict(
        food_items=[valid_foot_item],
        meal_type_guess="lunch",
        dietary_flags=[Diet.HIGH_PROTEIN],
        macros_total=valid_macros,
        balanced_score=0.8,
    )
