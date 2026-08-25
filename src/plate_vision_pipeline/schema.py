from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

class Diet(str, Enum):
    '''Dietary Flags'''
    HIGH_PROTEIN = "high protein"
    LOW_PROTEIN = "low protein"
    HIGH_CALORIE ="high calorie"
    LOW_CALORIE = "low calorie"
    HIGH_SODIUM = "high sodium"
    LOW_SODIUM = "low sodium"
    HIGH_SUGAR = "high sugar"
    LOW_SUGAR = "low sugar"
    HIGH_FAT = "high fat"
    LOW_FAT = "low fat"
    HIGH_FIBER = "high fiber"

class Macros(BaseModel):
    protein_g: float = Field(..., ge=0,description="Estimated grams of protein in this portion")
    carbs_g:float = Field(..., ge=0,description="Estimated grams of carbohydrates in this portion")
    fat_g: float = Field(... , ge=0, description="Estimated grams of fat in this portion")


class FoodItem(BaseModel):
    name : str = Field(..., description="Common name of the food Item")
    coco_class: str = Field(..., description="COCO dataset class as detected by YOLO")
    portion_estimate_g: int = Field(..., ge=10, le=1000, description="Estimated weight of this portion in grams")
    calories_estimate: int = Field(..., ge=0, le=3000, description="Estimated calories in this portion")
    macros: Macros
    preparation_note: str | None = None

class PlateAnalysis(BaseModel):
    food_items : list[FoodItem]
    meal_type_guess: Literal["breakfast" , "lunch", "dinner", "snack"]
    dietary_flags : list[Diet]
    macros_total: Macros
    balanced_score : float = Field(..., ge=0.0, le=1.0, description="Nutritional balance score from 0.0 (unbalanced) to 1.0 (perfectly balanced)")
    notes: str | None = None

    @computed_field
    @property
    def total_calories(self) -> int:
        return sum(item.calories_estimate for item in self.food_items)
