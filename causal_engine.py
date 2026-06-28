"""Causal analysis utilities for SWE-ReBench evaluation trajectories."""

from dataclasses import asdict, dataclass
from math import exp, sqrt
from statistics import mean
from typing import Iterable, Optional

from models import EvaluationRun


@dataclass
class TrajectoryFeatures:
    """Numeric and binary features extracted from one evaluation run."""

    run_id: str
    task_id: str
    model_id: str
    resolved: bool
    num_steps: int
    bash_count: int
    edit_count: int
    search_count: int
    open_count: int
    create_count: int
    submit_count: int
    error_count: int
    error_rate: float
    edit_ratio: float
    search_ratio: float
    bash_ratio: float
    time_spent_seconds: float
    has_generated_patch: bool
    used_search_before_edit: bool
    reproduced_error_before_fix: bool
    high_edit_ratio: bool
    high_command_count: bool

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class MatchedPair:
    treated_run_id: str
    control_run_id: str
    propensity_distance: float


@dataclass
class TreatmentEffect:
    treatment: str
    description: str
    treated_count: int
    control_count: int
    ate: float
    att: float
    matched_att: float
    matched_pairs: list[MatchedPair]


@dataclass
class RootCauseReport:
    treatment_effects: list[TreatmentEffect]
    top_positive: list[TreatmentEffect]
    top_negative: list[TreatmentEffect]


class CausalInferenceEngine:
    """Estimate causal effects from agent trajectory features.

    This is intentionally lightweight: it avoids a hard dependency on
    scikit-learn while still providing the core pieces needed for first-pass
    causal analysis: feature extraction, propensity scoring, nearest-neighbor
    matching, and treatment effect summaries.
    """

    DEFAULT_TREATMENTS: dict[str, str] = {
        "used_search_before_edit": "Used search before the first edit",
        "reproduced_error_before_fix": "Observed an error before making an edit",
        "high_edit_ratio": "Spent an above-median share of actions editing",
        "high_command_count": "Used an above-median number of actions",
        "has_generated_patch": "Generated a non-empty patch",
    }

    DEFAULT_COVARIATES: tuple[str, ...] = (
        "num_steps",
        "bash_count",
        "edit_count",
        "search_count",
        "open_count",
        "create_count",
        "error_rate",
        "time_spent_seconds",
    )

    def extract_features(self, runs: Iterable[EvaluationRun]) -> list[TrajectoryFeatures]:
        """Extract behavioral trajectory features from evaluation runs."""

        raw_features = [self._extract_single_run(run) for run in runs]
        if not raw_features:
            return []

        median_steps = self._median([feature.num_steps for feature in raw_features])
        median_edit_ratio = self._median([feature.edit_ratio for feature in raw_features])

        features = []
        for feature in raw_features:
            feature.high_command_count = feature.num_steps > median_steps
            feature.high_edit_ratio = feature.edit_ratio > median_edit_ratio
            features.append(feature)
        return features

    def feature_matrix(
        self,
        features: list[TrajectoryFeatures],
        feature_names: Optional[list[str]] = None,
    ) -> tuple[list[list[float]], list[str]]:
        """Return a numeric feature matrix and the column names used."""

        names = feature_names or list(self.DEFAULT_COVARIATES)
        matrix = [
            [self._numeric_value(feature, name) for name in names]
            for feature in features
        ]
        return matrix, names

    def estimate_propensity_scores(
        self,
        features: list[TrajectoryFeatures],
        treatment: str,
        covariates: Optional[list[str]] = None,
    ) -> dict[str, float]:
        """Estimate P(treatment | covariates) with logistic regression."""

        if not features:
            return {}

        covariate_names = covariates or [
            name for name in self.DEFAULT_COVARIATES if name != treatment
        ]
        matrix, _ = self.feature_matrix(features, covariate_names)
        labels = [self._binary_value(feature, treatment) for feature in features]

        if len(set(labels)) == 1:
            constant = 0.99 if labels[0] else 0.01
            return {feature.run_id: constant for feature in features}

        scaled = self._standardize(matrix)
        weights = self._fit_logistic_regression(scaled, labels)

        scores = {}
        for feature, row in zip(features, scaled):
            score = self._sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], row)))
            scores[feature.run_id] = min(max(score, 0.01), 0.99)
        return scores

    def match_by_propensity(
        self,
        features: list[TrajectoryFeatures],
        treatment: str,
        scores: Optional[dict[str, float]] = None,
        caliper: Optional[float] = None,
    ) -> list[MatchedPair]:
        """Match each treated run with the nearest untreated run."""

        if not features:
            return []

        propensity_scores = scores or self.estimate_propensity_scores(features, treatment)
        missing_ids = {feature.run_id for feature in features} - set(propensity_scores)
        if missing_ids:
            raise ValueError(f"Missing propensity scores for run ids: {sorted(missing_ids)}")
        treated = [feature for feature in features if self._binary_value(feature, treatment)]
        controls = [feature for feature in features if not self._binary_value(feature, treatment)]

        pairs = []
        used_controls: set[str] = set()
        for treated_feature in treated:
            candidates = [
                control for control in controls if control.run_id not in used_controls
            ]
            if not candidates:
                break

            best_control = min(
                candidates,
                key=lambda control: abs(
                    propensity_scores[treated_feature.run_id] - propensity_scores[control.run_id]
                ),
            )
            distance = abs(
                propensity_scores[treated_feature.run_id] - propensity_scores[best_control.run_id]
            )
            if caliper is not None and distance > caliper:
                continue

            used_controls.add(best_control.run_id)
            pairs.append(
                MatchedPair(
                    treated_run_id=treated_feature.run_id,
                    control_run_id=best_control.run_id,
                    propensity_distance=distance,
                )
            )
        return pairs

    def estimate_treatment_effect(
        self,
        features: list[TrajectoryFeatures],
        treatment: str,
        description: Optional[str] = None,
        covariates: Optional[list[str]] = None,
    ) -> TreatmentEffect:
        """Estimate ATE, ATT, and matched ATT for a binary treatment."""

        treated = [feature for feature in features if self._binary_value(feature, treatment)]
        controls = [feature for feature in features if not self._binary_value(feature, treatment)]
        treated_outcomes = [float(feature.resolved) for feature in treated]
        control_outcomes = [float(feature.resolved) for feature in controls]

        treated_mean = mean(treated_outcomes) if treated_outcomes else 0.0
        control_mean = mean(control_outcomes) if control_outcomes else 0.0
        ate = treated_mean - control_mean if treated and controls else 0.0
        att = ate

        scores = self.estimate_propensity_scores(features, treatment, covariates)
        pairs = self.match_by_propensity(features, treatment, scores)
        outcome_by_run_id = {feature.run_id: float(feature.resolved) for feature in features}
        if pairs:
            matched_att = mean(
                outcome_by_run_id[pair.treated_run_id] - outcome_by_run_id[pair.control_run_id]
                for pair in pairs
            )
        else:
            matched_att = 0.0

        return TreatmentEffect(
            treatment=treatment,
            description=description or self.DEFAULT_TREATMENTS.get(treatment, treatment),
            treated_count=len(treated),
            control_count=len(controls),
            ate=ate,
            att=att,
            matched_att=matched_att,
            matched_pairs=pairs,
        )

    def root_cause_report(
        self,
        runs: Iterable[EvaluationRun],
        treatments: Optional[list[str]] = None,
    ) -> RootCauseReport:
        """Rank treatments by their matched effect on resolution."""

        features = self.extract_features(runs)
        treatment_names = treatments or list(self.DEFAULT_TREATMENTS)
        effects = [
            self.estimate_treatment_effect(
                features,
                treatment,
                self.DEFAULT_TREATMENTS.get(treatment, treatment),
            )
            for treatment in treatment_names
        ]
        effects.sort(key=lambda effect: abs(effect.matched_att), reverse=True)

        positive = sorted(
            [effect for effect in effects if effect.matched_att > 0],
            key=lambda effect: effect.matched_att,
            reverse=True,
        )
        negative = sorted(
            [effect for effect in effects if effect.matched_att < 0],
            key=lambda effect: effect.matched_att,
        )

        return RootCauseReport(
            treatment_effects=effects,
            top_positive=positive,
            top_negative=negative,
        )

    def _extract_single_run(self, run: EvaluationRun) -> TrajectoryFeatures:
        action_types = [action.action_type.lower() for action in run.actions]
        contents = [action.content.lower() for action in run.actions]
        observations = [(action.observation or "").lower() for action in run.actions]
        num_steps = len(run.actions)

        edit_count = self._count_matching(action_types, contents, ("edit", "replace"))
        search_count = self._count_matching(
            action_types,
            contents,
            ("search", "search_file", "grep", "rg", "find"),
        )
        open_count = self._count_matching(action_types, contents, ("open", "cat", "sed"))
        create_count = self._count_matching(action_types, contents, ("create", "touch"))
        submit_count = self._count_matching(action_types, contents, ("submit",))
        bash_count = action_types.count("bash")
        error_count = sum(
            1 for action, observation in zip(run.actions, observations)
            if not action.success or "error" in observation or "traceback" in observation
        )

        first_edit_index = self._first_action_index(action_types, contents, ("edit", "replace"))
        first_search_index = self._first_action_index(
            action_types,
            contents,
            ("search", "search_file", "grep", "rg", "find"),
        )
        first_error_index = self._first_error_index(run.actions, observations)

        return TrajectoryFeatures(
            run_id=run.run_id,
            task_id=run.task_id,
            model_id=run.model_id,
            resolved=run.resolved,
            num_steps=num_steps,
            bash_count=bash_count,
            edit_count=edit_count,
            search_count=search_count,
            open_count=open_count,
            create_count=create_count,
            submit_count=submit_count,
            error_count=error_count,
            error_rate=error_count / num_steps if num_steps else 0.0,
            edit_ratio=edit_count / num_steps if num_steps else 0.0,
            search_ratio=search_count / num_steps if num_steps else 0.0,
            bash_ratio=bash_count / num_steps if num_steps else 0.0,
            time_spent_seconds=float(run.execution_time_seconds or 0.0),
            has_generated_patch=bool(run.generated_patch and run.generated_patch.strip()),
            used_search_before_edit=(
                first_search_index is not None
                and first_edit_index is not None
                and first_search_index < first_edit_index
            ),
            reproduced_error_before_fix=(
                first_error_index is not None
                and first_edit_index is not None
                and first_error_index < first_edit_index
            ),
            high_edit_ratio=False,
            high_command_count=False,
        )

    def _count_matching(
        self,
        action_types: list[str],
        contents: list[str],
        needles: tuple[str, ...],
    ) -> int:
        return sum(
            1 for action_type, content in zip(action_types, contents)
            if action_type in needles or content.startswith(needles)
        )

    def _first_action_index(
        self,
        action_types: list[str],
        contents: list[str],
        needles: tuple[str, ...],
    ) -> Optional[int]:
        for index, (action_type, content) in enumerate(zip(action_types, contents)):
            if action_type in needles or content.startswith(needles):
                return index
        return None

    def _first_error_index(self, actions, observations: list[str]) -> Optional[int]:
        for index, (action, observation) in enumerate(zip(actions, observations)):
            if not action.success or "error" in observation or "traceback" in observation:
                return index
        return None

    def _numeric_value(self, feature: TrajectoryFeatures, name: str) -> float:
        value = getattr(feature, name)
        if isinstance(value, bool):
            return 1.0 if value else 0.0
        return float(value)

    def _binary_value(self, feature: TrajectoryFeatures, name: str) -> bool:
        value = getattr(feature, name)
        if isinstance(value, bool):
            return value
        return float(value) > 0.0

    def _standardize(self, matrix: list[list[float]]) -> list[list[float]]:
        if not matrix:
            return []

        columns = list(zip(*matrix))
        means = [mean(column) for column in columns]
        stds = [
            sqrt(sum((value - column_mean) ** 2 for value in column) / len(column)) or 1.0
            for column, column_mean in zip(columns, means)
        ]

        return [
            [
                (value - column_mean) / column_std
                for value, column_mean, column_std in zip(row, means, stds)
            ]
            for row in matrix
        ]

    def _fit_logistic_regression(
        self,
        matrix: list[list[float]],
        labels: list[bool],
        learning_rate: float = 0.1,
        iterations: int = 400,
        l2: float = 0.01,
    ) -> list[float]:
        if not matrix:
            return [0.0]

        weights = [0.0] * (len(matrix[0]) + 1)
        y = [1.0 if label else 0.0 for label in labels]
        n = len(matrix)

        for _ in range(iterations):
            gradients = [0.0] * len(weights)
            for row, target in zip(matrix, y):
                prediction = self._sigmoid(weights[0] + sum(w * x for w, x in zip(weights[1:], row)))
                error = prediction - target
                gradients[0] += error
                for index, value in enumerate(row, start=1):
                    gradients[index] += error * value

            weights[0] -= learning_rate * gradients[0] / n
            for index in range(1, len(weights)):
                penalty = l2 * weights[index]
                weights[index] -= learning_rate * ((gradients[index] / n) + penalty)

        return weights

    def _sigmoid(self, value: float) -> float:
        if value >= 0:
            z = exp(-value)
            return 1.0 / (1.0 + z)
        z = exp(value)
        return z / (1.0 + z)

    def _median(self, values: list[float]) -> float:
        if not values:
            return 0.0

        sorted_values = sorted(values)
        midpoint = len(sorted_values) // 2
        if len(sorted_values) % 2:
            return float(sorted_values[midpoint])
        return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2)
