from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from typing import Dict, Any

def get_random_forest_model(
    n_estimators: int = 100,
    max_depth: Any = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: str = 'sqrt',
    class_weight: Any = None,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Constructs Random Forest model corresponding to Orange Data Mining / Scikit-Learn defaults.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )

def get_decision_tree_model(
    max_depth: Any = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Any = None,
    random_state: int = 42
) -> DecisionTreeClassifier:
    """
    Constructs Decision Tree comparison model.
    """
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state
    )

def get_constant_model() -> DummyClassifier:
    """
    Constructs Constant baseline model that predicts the majority class for all instances.
    """
    return DummyClassifier(strategy='most_frequent')
