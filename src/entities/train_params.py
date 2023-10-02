from dataclasses import dataclass, field


@dataclass()
class TrainingParams:
    model_type: str = field(default="CatBoostClassifier")
    loss_function: str = field(default="MultiClass")
    random_state: int = field(default=42)
    n_estimators: int = field(default=100)
    iterations: int = field(default=100)
    learning_rate: float = field(default=0.1)
    depth: int = field(default=5)
    bagging_temperature: float = field(default=0.2)
    thread_count: int = field(default=-1)
