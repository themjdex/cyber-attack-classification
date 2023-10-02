from dataclasses import dataclass, field
from typing import List


@dataclass()
class FeatureParams:
    features: List[str]
    cat_features: List[str]
    target_col: str = field(default='Label')

