"""Decision-tree interpretability for SWE-ReBench agent trajectories."""

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from causal_engine import CausalInferenceEngine, TrajectoryFeatures
from models import EvaluationRun


@dataclass
class FeatureImportance:
    feature: str
    importance: float


@dataclass
class StrategyValidationReport:
    train_accuracy: float
    test_accuracy: float
    num_train: int
    num_test: int
    top_features: list[FeatureImportance]


@dataclass
class FidelityReport:
    agreement_rate: float
    num_samples: int
    num_disagreements: int


class AgentInterpretabilityModule:
    """Fit and inspect a decision-tree surrogate for agent outcomes."""

    DEFAULT_FEATURES: tuple[str, ...] = (
        "num_steps",
        "bash_count",
        "edit_count",
        "search_count",
        "open_count",
        "create_count",
        "submit_count",
        "error_count",
        "error_rate",
        "edit_ratio",
        "search_ratio",
        "bash_ratio",
        "time_spent_seconds",
        "has_generated_patch",
        "used_search_before_edit",
        "reproduced_error_before_fix",
        "high_edit_ratio",
        "high_command_count",
    )

    def __init__(
        self,
        max_depth: int = 4,
        min_samples_leaf: int = 1,
        random_state: int = 0,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.feature_extractor = CausalInferenceEngine()
        self.model: Optional[DecisionTreeClassifier] = None
        self.feature_names: list[str] = list(self.DEFAULT_FEATURES)

    def extract_trajectory_features(
        self,
        runs: Iterable[EvaluationRun],
        feature_names: Optional[list[str]] = None,
    ) -> tuple[list[TrajectoryFeatures], list[list[float]], list[int], list[str]]:
        """Extract features, matrix, labels, and column names from runs."""

        features = self.feature_extractor.extract_features(runs)
        names = feature_names or list(self.DEFAULT_FEATURES)
        matrix, names = self.feature_extractor.feature_matrix(features, names)
        labels = [1 if feature.resolved else 0 for feature in features]
        return features, matrix, labels, names

    def fit_decision_tree(
        self,
        runs: Iterable[EvaluationRun],
        feature_names: Optional[list[str]] = None,
    ) -> DecisionTreeClassifier:
        """Fit a decision-tree classifier from trajectory features to resolution."""

        _, matrix, labels, names = self.extract_trajectory_features(runs, feature_names)
        if not matrix:
            raise ValueError("At least one evaluation run is required")

        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        model.fit(matrix, labels)

        self.model = model
        self.feature_names = names
        return model

    def feature_importance_ranking(
        self,
        model: Optional[DecisionTreeClassifier] = None,
        feature_names: Optional[list[str]] = None,
    ) -> list[FeatureImportance]:
        """Return feature importances sorted from most to least important."""

        fitted_model = model or self._require_model()
        names = feature_names or self.feature_names
        importances = [
            FeatureImportance(feature=name, importance=float(importance))
            for name, importance in zip(names, fitted_model.feature_importances_)
        ]
        return sorted(importances, key=lambda item: item.importance, reverse=True)

    def decision_rules(
        self,
        model: Optional[DecisionTreeClassifier] = None,
        feature_names: Optional[list[str]] = None,
    ) -> str:
        """Render the fitted tree as readable if/else rules."""

        fitted_model = model or self._require_model()
        names = feature_names or self.feature_names
        return export_text(fitted_model, feature_names=names)

    def validate_strategies(
        self,
        runs: Iterable[EvaluationRun],
        test_size: float = 0.3,
        feature_names: Optional[list[str]] = None,
    ) -> StrategyValidationReport:
        """Validate tree-discovered strategies on a held-out split."""

        _, matrix, labels, names = self.extract_trajectory_features(runs, feature_names)
        if len(matrix) < 2:
            raise ValueError("At least two evaluation runs are required for validation")

        stratify = labels if self._can_stratify(labels, test_size) else None
        split = train_test_split(
            matrix,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify,
        )
        x_train, x_test, y_train, y_test = split

        model = DecisionTreeClassifier(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        model.fit(x_train, y_train)

        self.model = model
        self.feature_names = names

        train_predictions = model.predict(x_train)
        test_predictions = model.predict(x_test)

        return StrategyValidationReport(
            train_accuracy=float(accuracy_score(y_train, train_predictions)),
            test_accuracy=float(accuracy_score(y_test, test_predictions)),
            num_train=len(x_train),
            num_test=len(x_test),
            top_features=self.feature_importance_ranking(model, names),
        )

    def fidelity_report(
        self,
        runs: Iterable[EvaluationRun],
        black_box_predictions: Optional[Callable[[EvaluationRun], bool]] = None,
        model: Optional[DecisionTreeClassifier] = None,
        feature_names: Optional[list[str]] = None,
    ) -> FidelityReport:
        """Measure agreement between the tree surrogate and black-box outcomes."""

        run_list = list(runs)
        _, matrix, labels, names = self.extract_trajectory_features(run_list, feature_names)
        if not matrix:
            return FidelityReport(agreement_rate=0.0, num_samples=0, num_disagreements=0)

        fitted_model = model or self.model or self.fit_decision_tree(run_list, names)
        tree_predictions = [bool(value) for value in fitted_model.predict(matrix)]
        black_box = (
            [bool(black_box_predictions(run)) for run in run_list]
            if black_box_predictions
            else [bool(label) for label in labels]
        )

        agreements = sum(
            1 for tree_prediction, black_box_prediction in zip(tree_predictions, black_box)
            if tree_prediction == black_box_prediction
        )
        num_samples = len(black_box)
        num_disagreements = num_samples - agreements

        return FidelityReport(
            agreement_rate=agreements / num_samples if num_samples else 0.0,
            num_samples=num_samples,
            num_disagreements=num_disagreements,
        )

    def _require_model(self) -> DecisionTreeClassifier:
        if self.model is None:
            raise ValueError("No decision tree has been fitted")
        return self.model

    def _can_stratify(self, labels: list[int], test_size: float) -> bool:
        class_counts = {label: labels.count(label) for label in set(labels)}
        if len(class_counts) < 2 or min(class_counts.values()) < 2:
            return False

        num_test = max(1, round(len(labels) * test_size))
        num_train = len(labels) - num_test
        return num_test >= len(class_counts) and num_train >= len(class_counts)
