"""
    Set Covering Machine -- The Set Covering Machine in Python
"""
from __future__ import print_function, division, absolute_import, unicode_literals

from six import iteritems

import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    check_is_fitted,
    check_random_state,
)
from warnings import warn

from io import StringIO


def _class_to_string(instance):
    """
    Returns a string representation of the public attributes of a class.

    Parameters:
    -----------
    instance: object
        An instance of any class.

    Returns:
    --------
    string_rep: string
        A string representation of the class and its public attributes.

    Notes:
    -----
    Private attributes must be marked with a leading underscore.
    """
    return (
        instance.__class__.__name__
        + "("
        + ",".join(
            [
                str(k) + "=" + str(v)
                for k, v in iteritems(instance.__dict__)
                if str(k[0]) != "_"
            ]
        )
        + ")"
    )


class BaseModel(object):
    def __init__(self):
        self.rules = []
        super(BaseModel, self).__init__()

    def add(self, rule):
        self.rules.append(rule)

    def predict(self, X):
        raise NotImplementedError()

    def predict_proba(self, X):
        raise NotImplementedError()

    def remove(self, index):
        del self.rules[index]

    @property
    def example_dependencies(self):
        return [d for ba in self.rules for d in ba.example_dependencies]

    @property
    def type(self):
        raise NotImplementedError()

    def _to_string(self, separator=" "):
        return separator.join([str(a) for a in self.rules])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __iter__(self):
        for ba in self.rules:
            yield ba

    def __len__(self):
        return len(self.rules)

    def __str__(self):
        return self._to_string()


class ConjunctionModel(BaseModel):
    def predict(self, X):
        predictions = np.ones(X.shape[0], bool)
        for a in self.rules:
            predictions = np.logical_and(predictions, a.classify(X))
        return predictions.astype(np.uint8)

    @property
    def type(self):
        return "conjunction"

    def __str__(self):
        return self._to_string(separator=" and ")


class BaseRule(object):
    """
    A rule mixin class

    """

    def __init__(self):
        super(BaseRule, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the rule.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        raise NotImplementedError()

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        raise NotImplementedError()

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return _class_to_string(self)


class DecisionStump(BaseRule):
    """
    A decision stump is a rule that applies a threshold to the value of some feature

    Parameters:
    -----------
    feature_idx: uint
        The index of the feature
    threshold: float
        The threshold at which the outcome of the rule changes
    kind: str, default="greater"
        The case in which the rule returns 1, either "greater" or "less_equal".

    """

    def __init__(self, feature_idx, threshold, kind="greater"):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.kind = kind
        super(DecisionStump, self).__init__()

    def classify(self, X):
        """
        Classifies a set of examples using the decision stump.

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features), dtype=np.float
            The feature vectors of examples to classify.

        Returns:
        --------
        classifications: array-like, shape=(n_examples,), dtype=bool
            The outcome of the rule (True or False) for each example.

        """
        if self.kind == "greater":
            c = X[:, self.feature_idx] > self.threshold
        else:
            c = X[:, self.feature_idx] <= self.threshold
        return c

    def inverse(self):
        """
        Creates a rule that is the opposite of the current rule (self).

        Returns:
        --------
        inverse: BaseRule
            A rule that is the inverse of self.

        """
        return DecisionStump(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            kind="greater" if self.kind == "less_equal" else "less_equal",
        )

    def __str__(self):
        return "X[{0:d}] {1!s} {2:.3f}".format(
            self.feature_idx, ">" if self.kind == "greater" else "<=", self.threshold
        )

class SetCoveringMachine(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        p=1.0,
        model_type="conjunction",
        max_rules=10,
        stopping_method=None,
        stopping_method_threshold=None,
        random_state=None,
    ):
        self.p = p
        self.model_type = model_type
        self.max_rules = max_rules
        self.stopping_method = stopping_method
        self.stopping_method_threshold = stopping_method_threshold
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "p": self.p,
            "model_type": self.model_type,
            "max_rules": self.max_rules,
            "stopping_method": self.stopping_method,
            "stopping_method_threshold": self.stopping_method_threshold,
            "random_state": self.random_state,
        }

    def set_params(self, **parameters):
        for parameter, value in iteritems(parameters):
            setattr(self, parameter, value)
        return self

    def fit(self, X, y, tiebreaker=None, iteration_callback=None, **fit_params):
        """
        Fit a SCM model.

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The features of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        self: object
            Returns self.

        """
        self.stream = StringIO()

        # temporary : 
        assert self.model_type == 'conjunction'

        if self.model_type == "conjunction":
            self._add_attribute_to_model = self._append_conjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_conjunction
        elif self.model_type == "disjunction":
            self._add_attribute_to_model = self._append_disjunction_model
            self._get_example_idx_by_class = self._get_example_idx_by_class_disjunction
        else:
            raise ValueError("Unsupported model type.")
        

        # Initialize callbacks
        if iteration_callback is None:
            iteration_callback = lambda x: None

        # Parse additional fit parameters
        self.stream.write("\nParsing additional fit parameters")
        utility_function_additional_args = {}
        if fit_params is not None:
            for key, value in iteritems(fit_params):
                if key[:9] == "utility__":
                    utility_function_additional_args[key[9:]] = value

        # Validate the input data
        self.stream.write("\nValidating the input data")
        self.classes_, y, total_n_ex_by_class = np.unique(
            y, return_inverse=True, return_counts=True
        )
        if len(self.classes_) != 2:
            raise ValueError("y must contain two unique classes.")
        self.stream.write(
            "\nThe data contains {0:d} examples. Negative class is {1!s} (n: {2:d}) and positive class is {3!s} (n: {4:d}).".format(
                len(y),
                self.classes_[0],
                total_n_ex_by_class[0],
                self.classes_[1],
                total_n_ex_by_class[1],
            )
        )
        print(f'the data contains {len(y)} examples. Negative class is {self.classes_[0]} (n: {total_n_ex_by_class[0]}) and positive class is {self.classes_[1]} (n: {total_n_ex_by_class[1]})')

        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        elif isinstance(X, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for X:", type(X))

        if isinstance(y, list):
            y = np.array(y)
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise ValueError("unexpected type for y:", type(X))

        # Create an empty model
        self.stream.write("\nInitializing empty model")
        self.model_ = ConjunctionModel()
        self.stream.write("\nTraining start")

        remaining_y = y
        n_negatives_at_start = len(remaining_y) - sum(remaining_y)
        print('remaining_y len, 1, and 0s', len(remaining_y), sum(remaining_y), len(remaining_y) - sum(remaining_y))
        
        all_possible_rules = []
        for feat_id in range(X.shape[1]):
            unique_values = list(set(X[:, feat_id]))
            unique_values.sort()
            for threshold in unique_values:
                for kind in ["greater", "less_equal"]:
                    all_possible_rules.append((feat_id, threshold, kind))
        idx_of_rules_to_consider = list(range(len(all_possible_rules)))
        
        print('all_possible_rules:', len(all_possible_rules))
        print('X.shape:', X.shape)

        big_pred_matrix = np.zeros((X.shape[0], len(all_possible_rules)), dtype=int)

        n_operations_to_do = X.shape[0]*len(all_possible_rules)
        n_operations_done = 0
        print(f'n_operations_to_do = {n_operations_to_do}')

        for sample_id in range(X.shape[0]):
            x = X[sample_id]
            for rule_id in range(len(all_possible_rules)):
                rule_feat_id, rule_threshold, rule_kind = all_possible_rules[rule_id]
                sample_feat_value = x[rule_feat_id]
                if rule_kind == "greater":
                    pred = 1 if (sample_feat_value > rule_threshold) else 0
                elif rule_kind == "less_equal":
                    pred = 1 if (sample_feat_value <= rule_threshold) else 0
                else:
                    raise ValueError("unexpected rule kind:", rule_kind)
                big_pred_matrix[sample_id, rule_id] = pred
                n_operations_done += 1
                if n_operations_done % 10000000 == 0:
                    print(f'n rule prediction operations done = {n_operations_done} / {n_operations_to_do} ({100*n_operations_done/n_operations_to_do:.2f}%)')

        # Calculate residuals
        residuals = (big_pred_matrix != remaining_y[:, None]).astype(int)

        X_remaining_samples = X.copy()

        continue_criterion = True
        while continue_criterion:
            print('one iteration of the while loop')
            scores_of_rules = []
            assert residuals.shape[1] == len(idx_of_rules_to_consider)
            self.stream.write(f'\nlen(all_possible_rules) = {len(all_possible_rules)}')
            n_residuals_computed = 0
            for i in range(residuals.shape[1]):
                res = residuals[:, i]  # erreurs de la regle
                utility_true_negatives = np.logical_not(
                    np.logical_or(res, remaining_y)
                ).astype(int)
                utility_false_negatives = np.logical_and(res, remaining_y).astype(int)
                utility = sum(utility_true_negatives) - self.p * sum(
                    utility_false_negatives
                )
                rule_id_in_all_possible_rules = idx_of_rules_to_consider[i]
                rule_feat_id, rule_threshold, rule_kind = all_possible_rules[rule_id_in_all_possible_rules]
                score_of_rule = utility
                scores_of_rules.append(score_of_rule)
                #print(f'rule {i} : utility score = {score_of_rule}')
                n_residuals_computed += 1
                if n_residuals_computed % 10000 == 0:
                    print(f'    n residuals computed = {n_residuals_computed} / {residuals.shape[1]}')
            maximal_score = max(scores_of_rules)
            best_rules_idx_in_big_pred_matrix = [k for k, v in enumerate(scores_of_rules) if v == maximal_score]
            best_rule_id_in_big_pred_matrix = best_rules_idx_in_big_pred_matrix[0]
            best_rules_idx_in_all_possible_rules = [idx_of_rules_to_consider[k] for k in best_rules_idx_in_big_pred_matrix]
            best_rule_id_in_all_possible_rules = best_rules_idx_in_all_possible_rules[0]
            if maximal_score > 0:
                for i in best_rules_idx_in_big_pred_matrix:
                    rule_id_in_all_possible_rules = idx_of_rules_to_consider[i]
                    feat_id, threshold, kind = all_possible_rules[rule_id_in_all_possible_rules]
                    print(f'rule {i} (feat {feat_id} {kind} {threshold}) has score {scores_of_rules[i]}')
            # number of ules wih bast score value
            n_rules_with_equal_best_score = len(best_rules_idx_in_all_possible_rules)
            print('n_rules_with_equal_best_score', n_rules_with_equal_best_score)
            print('best score = ', maximal_score)
            best_rule_score = maximal_score
            #print('best_rule_score', best_rule_score)
            best_rule_feat_id, best_rule_threshold, best_rule_kind = all_possible_rules[best_rule_id_in_all_possible_rules]
            self.stream.write('\nselected best rule : "feature {} {:2} {}"'.format(best_rule_feat_id, '>' if best_rule_kind == 'greater' else '<=', best_rule_threshold))
            print('selected best rule : "feature {} {:2} {}"'.format(best_rule_feat_id, '>' if best_rule_kind == 'greater' else '<=', best_rule_threshold))
            mask = np.zeros(big_pred_matrix.shape, dtype=bool)
            predictions_of_selected_rule = big_pred_matrix[:, best_rule_id_in_big_pred_matrix]
            classified_neg_examples = [(r == p == 0) for r, p in zip(residuals[:, best_rule_id_in_big_pred_matrix], predictions_of_selected_rule)]
            if sum(classified_neg_examples) == 0:
                print('the last added rule do not classify any negative example: breaking the while loop')
                break
            samples_to_keep = np.array(predictions_of_selected_rule).astype(bool)
            # computing the columns to keep
            X_remaining_samples = X_remaining_samples[samples_to_keep]
            assert big_pred_matrix.shape[1] == len(idx_of_rules_to_consider)
            updated_idx_of_rules_to_consider = []
            for i in idx_of_rules_to_consider:
                rule = all_possible_rules[i]
                feat_id, threshold, _ = rule
                if threshold in X_remaining_samples[:, feat_id]:
                    updated_idx_of_rules_to_consider.append(i)
            columns_to_keep = [rule_id in updated_idx_of_rules_to_consider for rule_id in idx_of_rules_to_consider]
            for i in range(big_pred_matrix.shape[0]):
                if samples_to_keep[i]:
                    mask[i] = columns_to_keep
            new_dimensions = (sum(samples_to_keep), sum(columns_to_keep))
            self.stream.write('\nnew_dimensions = {}'.format(new_dimensions))
            updated_big_pred_matrix = big_pred_matrix[mask].reshape(new_dimensions)
            updated_residuals = residuals[mask].reshape(new_dimensions)
            #
            assert updated_big_pred_matrix.shape == new_dimensions
            assert updated_big_pred_matrix.shape[1] == len(updated_idx_of_rules_to_consider)
            #
            remaining_y = remaining_y[samples_to_keep]
            big_pred_matrix = updated_big_pred_matrix
            residuals = updated_residuals
            print('remaining_y len, 1, and 0s', len(remaining_y), sum(remaining_y), len(remaining_y) - sum(remaining_y))
            idx_of_rules_to_consider = updated_idx_of_rules_to_consider

            stump = DecisionStump(
                feature_idx=best_rule_feat_id,
                threshold=best_rule_threshold,
                kind=best_rule_kind,
            )

            self.stream.write("\nThe best rule has score {}".format(best_rule_score))
            self._add_attribute_to_model(stump)

            there_are_still_space_to_add_rules_in_model = len(self.model_) < self.max_rules
            there_are_still_y_to_cover = len(remaining_y) > 0
            there_are_still_rules_to_consider = len(idx_of_rules_to_consider) > 0
            last_rule_covered_at_least_one_negative = sum(classified_neg_examples) > 0
            print(f'there are still space to add rules in model = {there_are_still_space_to_add_rules_in_model} | len(self.model_) = {len(self.model_)} (max = {self.max_rules})')
            print(f'there are still y to cover =                  {there_are_still_y_to_cover} | len(remaining_y) = {len(remaining_y)}')
            print(f'there are still rules to consider =           {there_are_still_rules_to_consider} | len(idx_of_rules_to_consider) = {len(idx_of_rules_to_consider)}')
            print(f'last_rule_covered_at_least_one_negative =     {last_rule_covered_at_least_one_negative} | n neg examples classified by last rule = {sum(classified_neg_examples)}')
            if self.stopping_method is None:    
                continue_criterion = True
            elif self.stopping_method == "percentagenegativescovered":
                percentage_threshold = self.stopping_method_threshold
                self.stream.write(f"\n self.stopping_method == percentagenegativescovered, threshold = {percentage_threshold}")
                n_negatives_to_cover = len(remaining_y) - sum(remaining_y)
                percentages_negatives_covered = 1 - (n_negatives_to_cover / n_negatives_at_start)
                self.stream.write(f"\npercentages_negatives_covered = {percentages_negatives_covered}")
                continue_criterion = percentages_negatives_covered < percentage_threshold
                print('n_negatives_to_cover', n_negatives_to_cover)
                print('n_negatives_at_start', n_negatives_at_start)
                print('percentages_negatives_covered', percentages_negatives_covered)
            else:
                raise ValueError("unexpected stopping_method paramter value", self.stopping_method)
            print(f'continue given the stopping method =          {continue_criterion} | (stopping_method = {self.stopping_method}, stopping_method_threshold = {self.stopping_method_threshold})')

            continue_criterion = continue_criterion and there_are_still_space_to_add_rules_in_model and there_are_still_y_to_cover and there_are_still_rules_to_consider and last_rule_covered_at_least_one_negative

            self.stream.write("\nDiscarding all examples that the rule classifies as negative")

            iteration_callback(self.model_)

        self.stream.write("\nTraining completed")

        self.stream.write("\nCalculating rule importances")
        # Definition: how often each rule outputs a value that causes the value of the model to be final
        final_outcome = 0 if self.model_type == "conjunction" else 1
        total_outcome = (self.model_.predict(X) == final_outcome).sum()  # n times the model outputs the final outcome
        self.rule_importances = np.array(
            [
                (r.classify(X) == final_outcome).sum() / total_outcome
                for r in self.model_.rules
            ]
        )  # contribution of each rule
        self.stream.write("\nDone.")
 
        return self

    def predict(self, X):
        """
        Predict class

        Parameters:
        -----------
        X: array-like, shape=[n_examples, n_features]
            The feature of the input examples.

        Returns:
        --------
        predictions: numpy_array, shape=[n_examples]
            The predicted class for each example.

        """
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
        X = check_array(X)
        return self.classes_[self.model_.predict(X)]

    def predict_proba(self, X):
        """
        Predict class probabilities

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.

        Returns:
        --------
        p : array of shape = [n_examples, 2]
            The class probabilities for each example. Classes are ordered by lexicographic order.

        """
        warn(
            "SetCoveringMachines do not support probabilistic predictions. The returned values will be zero or one.",
            RuntimeWarning,
        )
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
        X = check_array(X)
        pos_proba = self.classes_[self.model_.predict(X)]
        neg_proba = 1.0 - pos_proba
        return np.hstack((neg_proba.reshape(-1, 1), pos_proba.reshape(-1, 1)))

    def score(self, X, y):
        """
        Predict classes of examples and measure accuracy

        Parameters:
        -----------
        X: array-like, shape=(n_examples, n_features)
            The feature of the input examples.
        y : array-like, shape = [n_samples]
            The labels of the input examples.

        Returns:
        --------
        accuracy: float
            The proportion of correctly classified examples.

        """
        check_is_fitted(self, ["model_", "rule_importances", "classes_"])
        X, y = check_X_y(X, y)
        return accuracy_score(y_true=y, y_pred=self.predict(X))

    def _append_conjunction_model(self, new_rule):
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _append_disjunction_model(self, new_rule):
        new_rule = new_rule.inverse()
        self.model_.add(new_rule)
        logging.debug("Attribute added to the model: " + str(new_rule))
        return new_rule

    def _get_example_idx_by_class_conjunction(self, y):
        positive_example_idx = np.where(y == 1)[0]
        negative_example_idx = np.where(y == 0)[0]
        return positive_example_idx, negative_example_idx

    def _get_example_idx_by_class_disjunction(self, y):
        positive_example_idx = np.where(y == 0)[0]
        negative_example_idx = np.where(y == 1)[0]
        return positive_example_idx, negative_example_idx

    def __str__(self):
        return _class_to_string(self)
