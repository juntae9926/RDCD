# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Original source: https://github.com/facebookresearch/isc2021

from dataclasses import astuple, dataclass
from typing import List, Optional, Tuple
from collections import defaultdict
import numpy as np


@dataclass
class GroundTruthMatch:
    query: str
    db: str


@dataclass
class PredictedMatch:
    query: str
    db: str
    score: float


@dataclass
class Metrics:
    average_precision: float
    precisions: np.ndarray
    recalls: np.ndarray
    thresholds: np.ndarray
    recall_at_p90: float
    threshold_at_p90: float
    recall_at_rank1: float
    recall_at_rank10: float


def argsort(seq):
    # from https://stackoverflow.com/a/3382369/3853462
    return sorted(range(len(seq)), key=seq.__getitem__)


def precision_recall(
    y_true: np.ndarray, probas_pred: np.ndarray, num_positives: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute precisions, recalls and thresholds.

    Parameters
    ----------
    y_true : np.ndarray
        Binary label of each prediction (0 or 1). Shape [n, k] or [n*k, ]
    probas_pred : np.ndarray
        Score of each prediction (higher score == images more similar, ie not a distance)
        Shape [n, k] or [n*k, ]
    num_positives : int
        Number of positives in the groundtruth.

    Returns
    -------
    precisions, recalls, thresholds
        ordered by increasing recall.
    """
    probas_pred = probas_pred.flatten()
    y_true = y_true.flatten()
    # to handle duplicates scores, we sort (score, NOT(jugement)) for predictions
    # eg,the final order will be (0.5, False), (0.5, False), (0.5, True), (0.4, False), ...
    # This allows to have the worst possible AP.
    # It prevents participants from putting the same score for all predictions to get a good AP.
    order = argsort(list(zip(probas_pred, ~y_true)))
    order = order[::-1]  # sort by decreasing score
    probas_pred = probas_pred[order]
    y_true = y_true[order]

    ntp = np.cumsum(y_true)  # number of true positives <= threshold
    nres = np.arange(len(y_true)) + 1  # number of results

    precisions = ntp / nres
    recalls = ntp / num_positives
    return precisions, recalls, probas_pred


def average_precision_old(recalls: np.ndarray, precisions: np.ndarray):
    """
    Compute the micro average-precision score (uAP).

    Parameters
    ----------
    recalls : np.ndarray
        Recalls, can be in any order.
    precisions : np.ndarray
        Precisions for each recall value.

    Returns
    -------
    uAP: float
    """

    # Order by increasing recall
    order = np.argsort(recalls)
    recalls = recalls[order]
    precisions = precisions[order]
    return ((recalls[1:] - recalls[:-1]) * precisions[:-1]).sum()


# Jay Qi's version
def average_precision(recalls: np.ndarray, precisions: np.ndarray):
    # Order by increasing recall
    # order = np.argsort(recalls)
    # recalls = recalls[order]
    # precisions = precisions[order]

    # Check that it's ordered by increasing recall
    if not np.all(recalls[:-1] <= recalls[1:]):
        raise Exception("recalls array must be sorted before passing in")

    return ((recalls - np.concatenate([[0], recalls[:-1]])) * precisions).sum()


def find_operating_point(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, required_x: float
) -> Tuple[float, Optional[float], Optional[float]]:
    """
    Find the highest y with x at least `required_x`.

    Returns
    -------
    x, y, z
        The best operating point (highest y) with x at least `required_x`.
        If we can't find a point with the required x value, return
        x=required_x, y=None, z=None
    """
    valid_points = x >= required_x
    if not np.any(valid_points):
        return required_x, None, None

    valid_x = x[valid_points]
    valid_y = y[valid_points]
    valid_z = z[valid_points]
    best_idx = np.argmax(valid_y)
    return valid_x[best_idx], valid_y[best_idx], valid_z[best_idx]


def check_duplicates(predictions: List[PredictedMatch]) -> List[PredictedMatch]:
    """
    Raise an exception if predictions contains duplicates
    (ie several predictions for the same (query, db) pair).
    """
    unique_pairs = set((p.query, p.db) for p in predictions)
    if len(unique_pairs) != len(predictions):
        raise ValueError("Predictions contains duplicates.")


def sanitize_predictions(predictions: List[PredictedMatch]) -> List[PredictedMatch]:
    # TODO(lowik) check for other possible loopholes
    check_duplicates(predictions)
    return predictions

def extract_positive_and_nearest_negative_scores(probas_pred, y_true):
    positive_scores = []
    nearest_negative_scores = []
    Negative_scores = []

    # 각 쿼리에 대한 점수를 처리합니다.
    for i in range(0, len(probas_pred), 5):  # 각 쿼리당 10개씩 점수가 있다고 가정
        scores_for_query = probas_pred[i:i+5]  # 현재 쿼리에 대한 10개의 점수
        true_labels_for_query = y_true[i:i+5]  # 현재 쿼리에 대한 10개의 정답 레이블

        # Positive sample의 점수를 찾습니다.
        if np.any(true_labels_for_query):
            positive_score = scores_for_query[true_labels_for_query.argmax()]
            positive_scores.append(positive_score)

            # 가장 높은 Negative sample의 점수를 찾습니다. (정렬되어 있으므로, positive를 제외한 첫 번째 점수가 됩니다.)
            negative_scores = np.delete(scores_for_query, true_labels_for_query.argmax())
            if negative_scores.size > 0:  # Negative sample이 존재하는 경우
                nearest_negative_scores.append(negative_scores.max())
                Negative_scores.append(negative_scores.mean())
            else:  # 모든 후보가 Positive인 드문 경우 처리
                nearest_negative_scores.append(np.nan)
                Negative_scores.append(np.nan)
            #import pdb;pdb.set_trace()
        else:
            # 정답이 없는 경우는 무시하거나, 다른 방식으로 처리할 수 있습니다.
            continue

    return np.array(positive_scores), np.array(nearest_negative_scores), np.array(Negative_scores)

def make_npy(positive_sim, nearest_neg_sim, neg_sim):
    # 결과를 npy 파일로 저장
    np.save('/user/appendix/positive_scores.npy', positive_sim)
    np.save('/user/appendix/nearest_negative_scores.npy', nearest_neg_sim)
    np.save('/user/appendix/neg_scores.npy', neg_sim)

def to_arrays(gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]):
    """Convert from list of matches to arrays"""
    predictions = sanitize_predictions(predictions)
    gt_set = {astuple(g) for g in gt_matches}
    probas_pred = np.array([p.score for p in predictions])
    y_true = np.array([(p.query, p.db) in gt_set for p in predictions], dtype=bool)
    positive_sim, nearest_neg_sim, neg_sim = extract_positive_and_nearest_negative_scores(probas_pred, y_true)
    import pdb;pdb.set_trace()
    make_npy(positive_sim, nearest_neg_sim, neg_sim)
    return y_true, probas_pred


def find_tp_ranks(
    gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]
):
    q_to_res = defaultdict(list)
    for p in predictions:
        q_to_res[p.query].append(p)
    ranks = []
    not_found = int(1 << 35)
    for m in gt_matches:
        if m.query not in q_to_res:
            ranks.append(not_found)
            continue
        res = q_to_res[m.query]
        res = np.array([(p.score, m.db == p.db) for p in res])
        (i,) = np.where(res[:, 1] == 1)
        if i.size == 0:
            ranks.append(not_found)
        else:
            i = i[0]
            rank = (res[:, 0] >= res[i, 0]).sum() - 1
            ranks.append(rank)
    return np.array(ranks)

def save_predictions(predictions):
    import pandas as pd
    data = {
    "query": [pm.query for pm in predictions],
    "db": [pm.db for pm in predictions],
    "score": [pm.score for pm in predictions]
    }

    df = pd.DataFrame(data)

    # DataFrame을 CSV 파일로 저장
    df.to_csv("/user/appendix/predictions.csv", index=False)


def evaluate(
    gt_matches: List[GroundTruthMatch], predictions: List[PredictedMatch]
) -> Metrics:
    predictions = sanitize_predictions(predictions)
    import pdb;pdb.set_trace()
    save_predictions(predictions)
    import pdb;pdb.set_trace()
    y_true, probas_pred = to_arrays(gt_matches, predictions)
    p, r, t = precision_recall(y_true, probas_pred, len(gt_matches))
    ap = average_precision(r, p)
    pp90, rp90, tp90 = find_operating_point(p, r, t, required_x=0.9)  # @Precision=90%
    ranks = find_tp_ranks(gt_matches, predictions)
    recall_at_rank1 = (ranks == 0).sum() / ranks.size
    recall_at_rank10 = (ranks < 10).sum() / ranks.size

    return Metrics(
        average_precision=ap,
        precisions=p,
        recalls=r,
        thresholds=t,
        recall_at_p90=rp90,
        threshold_at_p90=tp90,
        recall_at_rank1=recall_at_rank1,
        recall_at_rank10=recall_at_rank10,
    )


def print_metrics(metrics: Metrics):
    print(f"Average Precision: {metrics.average_precision:.5f}")
    if metrics.recall_at_p90 is None:
        print("Does not reach P90")
    else:
        print(f"Recall at P90    : {metrics.recall_at_p90:.5f}")
        print(f"Threshold at P90 : {metrics.threshold_at_p90:g}")
    print(f"Recall at rank 1:  {metrics.recall_at_rank1:.5f}")
    print(f"Recall at rank 10: {metrics.recall_at_rank10:.5f}")
