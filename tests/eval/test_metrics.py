from adaptive.eval.metrics import (
    answer_correctness,
    citation_completeness,
    groundedness,
    mrr,
    ndcg,
    paired_bootstrap_delta,
    recall_at_k,
    route_confusion_matrix,
    sql_result_set_accuracy,
)

EXPECTED_HALF_SCORE = 0.5
EXPECTED_NDCG_SCORE = 0.6934264036172708
BOOTSTRAP_SAMPLES = 200
BOOTSTRAP_SEED = 7


def test_retrieval_metrics_use_ranked_ids_and_relevance() -> None:
    ranked = ["c3", "c1", "c2"]
    relevant = {"c1", "c2"}

    assert recall_at_k(ranked, relevant, 2) == EXPECTED_HALF_SCORE
    assert mrr(ranked, relevant) == EXPECTED_HALF_SCORE
    assert ndcg(ranked, relevant, 3) == EXPECTED_NDCG_SCORE


def test_answer_and_citation_metrics_are_normalized_and_grounded() -> None:
    assert answer_correctness("The window is 30 days.", "30 days") == 1.0
    assert groundedness("The window is 30 days.", ["Refunds are available within 30 days."]) == 1.0
    assert citation_completeness([{"chunk_id": "c1"}, {"chunk_id": "c2"}], {"c1", "c2"}) == 1.0


def test_sql_accuracy_and_route_confusion_are_exact() -> None:
    assert sql_result_set_accuracy([{"id": 1}, {"id": 2}], [{"id": 2}, {"id": 1}]) == 1.0
    matrix = route_confusion_matrix(
        ["parametric", "single_hop", "multi_hop"],
        ["parametric", "multi_hop", "multi_hop"],
    )
    assert matrix == {
        "parametric": {"parametric": 1, "single_hop": 0, "multi_hop": 0},
        "single_hop": {"parametric": 0, "single_hop": 0, "multi_hop": 1},
        "multi_hop": {"parametric": 0, "single_hop": 0, "multi_hop": 1},
    }


def test_paired_bootstrap_is_deterministic_and_reports_delta_interval() -> None:
    result = paired_bootstrap_delta(
        [1.0, 0.0, 1.0], [0.0, 0.0, 1.0], samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED
    )

    assert result.delta == 1 / 3
    assert result.lower <= result.delta <= result.upper
    assert result.samples == BOOTSTRAP_SAMPLES
    assert result.seed == BOOTSTRAP_SEED
