from dataclasses import dataclass, field


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.25)
    random_state: int = field(default=42)
    shuffle: bool = field(default=True)
