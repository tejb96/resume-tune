"""Tests for score-driven selection (pure logic, no LLM)."""

from __future__ import annotations

from scoring import (
    ContentRatings,
    SelectionPolicy,
    build_selection_from_scores,
    compute_content_composites,
    experience_bullet_composite,
    heuristic_ratings,
    normalize_rating,
    selection_trim_exhausted,
    trim_selection_by_lowest_score,
)


def test_normalize_rating_maps_1_to_5() -> None:
    assert normalize_rating(5) == 100.0
    assert normalize_rating(4) == 75.0
    assert normalize_rating(3) == 50.0
    assert normalize_rating(2) == 25.0
    assert normalize_rating(1) == 10.0
    assert normalize_rating(99) == 100.0


def test_chem_role_caps_high_strength_bullet() -> None:
    composite = experience_bullet_composite(role_relevance=1, bullet_strength=5)
    assert composite == 0.7 * 10.0 + 0.3 * 100.0
    swe = experience_bullet_composite(role_relevance=5, bullet_strength=4)
    assert swe == 0.7 * 100.0 + 0.3 * 75.0
    assert composite < swe


def test_build_excludes_irrelevant_role() -> None:
    background = {
        "experience": [
            {
                "company": "Acme",
                "title": "Software Engineer",
                "bullets": ["Built React apps", "Led API migration"],
            },
            {
                "company": "Plant",
                "title": "Chemical Engineer",
                "bullets": ["Reduced costs by 40%"],
            },
        ],
        "projects": [],
        "education": [],
    }
    ratings = ContentRatings(
        roles={0: 5, 1: 1},
        experience_bullets={(0, 0): 4, (0, 1): 5, (1, 0): 5},
    )
    selection = build_selection_from_scores(
        background,
        ratings,
        policy=SelectionPolicy(max_experience_entries=3, min_role_relevance_rating=2),
    )
    role_indices = [e["role_index"] for e in selection["experience_selections"]]
    assert 0 in role_indices
    assert 1 not in role_indices


def test_build_omits_low_scoring_project() -> None:
    background = {
        "experience": [
            {"company": "Co", "title": "Dev", "bullets": ["Shipped features"]},
        ],
        "projects": [{"name": "Side", "bullets": ["Built a toy app"]}],
        "education": [],
    }
    ratings = ContentRatings(
        roles={0: 5},
        experience_bullets={(0, 0): 5},
        projects={0: 3},
        project_bullets={(0, 0): 3},
    )
    selection = build_selection_from_scores(
        background,
        ratings,
        policy=SelectionPolicy(min_project_composite=50.0),
    )
    assert selection["project_selections"] == []


def test_build_project_floor_tops_up_best_gated_project() -> None:
    background = {
        "experience": [
            {"company": "Co", "title": "Dev", "bullets": ["Shipped features"]},
        ],
        "projects": [
            {"name": "Weak", "bullets": ["Toy app"]},
            {"name": "Better", "bullets": ["Slightly less toy app"]},
        ],
        "education": [],
    }
    ratings = ContentRatings(
        roles={0: 5},
        experience_bullets={(0, 0): 5},
        projects={0: 1, 1: 2},
        project_bullets={(0, 0): 2, (1, 0): 3},
    )
    selection = build_selection_from_scores(
        background,
        ratings,
        policy=SelectionPolicy(min_project_composite=50.0, min_project_entries=1, min_project_bullets=1),
    )
    assert len(selection["project_selections"]) == 1
    assert selection["project_selections"][0]["project_index"] == 1
    assert selection["project_selections"][0]["bullet_indices"] == [0]


def test_trim_protects_project_at_floor() -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1]}],
        "project_selections": [{"project_index": 0, "bullet_indices": [0]}],
        "education_indices": [0],
        "content_composites": {
            "exp:0:0": 92.5,
            "exp:0:1": 85.0,
            "proj:0:0": 44.0,
            "edu:0": 75.0,
        },
    }
    trimmed, event = trim_selection_by_lowest_score(
        selection,
        policy=SelectionPolicy(min_project_entries=1, min_project_bullets=1),
    )
    assert event is not None
    assert event["removed"] != "proj:0:0"
    assert trimmed["project_selections"] == [{"project_index": 0, "bullet_indices": [0]}]


def test_build_experience_floor_when_all_roles_below_gate() -> None:
    background = {
        "experience": [
            {"company": "Old", "title": "Chem Eng", "bullets": ["Metric bullet"]},
        ],
        "projects": [],
        "education": [],
    }
    ratings = ContentRatings(roles={0: 1}, experience_bullets={(0, 0): 5})
    selection = build_selection_from_scores(background, ratings)
    assert len(selection["experience_selections"]) == 1
    assert selection["experience_selections"][0]["bullet_indices"]


def test_trim_removes_lowest_composite_project_first() -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1]}],
        "project_selections": [{"project_index": 0, "bullet_indices": [0]}],
        "education_indices": [0],
        "content_composites": {
            "exp:0:0": 92.5,
            "exp:0:1": 85.0,
            "proj:0:0": 44.0,
            "edu:0": 75.0,
        },
    }
    trimmed, event = trim_selection_by_lowest_score(selection)
    assert event is not None
    assert event["removed"] == "proj:0:0"
    assert trimmed["project_selections"] == []


def test_trim_policy_prefers_project_within_gap() -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0, 1]}],
        "project_selections": [{"project_index": 0, "bullet_indices": [0]}],
        "education_indices": [0],
        "content_composites": {
            "exp:0:0": 92.5,
            "exp:0:1": 80.0,
            "proj:0:0": 78.0,
            "edu:0": 75.0,
        },
    }
    trimmed, event = trim_selection_by_lowest_score(
        selection,
        policy=SelectionPolicy(prefer_experience_within_gap=10.0, min_bullets_per_role=1),
    )
    assert event is not None
    assert event["removed"] == "proj:0:0"


def test_trim_never_removes_last_anchor_bullet() -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0]}],
        "project_selections": [],
        "education_indices": [0, 1],
        "content_composites": {
            "exp:0:0": 90.0,
            "edu:0": 75.0,
            "edu:1": 20.0,
        },
    }
    trimmed, event = trim_selection_by_lowest_score(selection)
    assert event is not None
    assert event["removed"] == "edu:1"
    assert trimmed["experience_selections"][0]["bullet_indices"] == [0]


def test_selection_trim_exhausted_at_floors() -> None:
    selection = {
        "experience_selections": [{"role_index": 0, "bullet_indices": [0]}],
        "project_selections": [],
        "education_indices": [0],
        "content_composites": {"exp:0:0": 90.0, "edu:0": 75.0},
    }
    assert selection_trim_exhausted(selection) is True


def test_build_excludes_low_relevance_education_for_software_role() -> None:
    background = {
        "experience": [{"company": "Co", "title": "Dev", "bullets": ["work"]}],
        "projects": [],
        "education": [
            {"degree": "M.Eng. Software Engineering", "institution": "U"},
            {"degree": "B.Sc. Chemical Engineering", "institution": "U"},
        ],
    }
    ratings = ContentRatings(
        roles={0: 5},
        experience_bullets={(0, 0): 4},
        education={0: 5, 1: 2},
    )
    selection = build_selection_from_scores(background, ratings)
    assert selection["education_indices"] == [0]


def test_heuristic_ratings_produces_valid_selection() -> None:
    background = {
        "experience": [
            {"company": "Co", "title": "Eng", "bullets": ["a", "b"]},
        ],
        "projects": [{"name": "P", "bullets": ["c"]}],
        "education": [{"degree": "BS CS", "institution": "U"}],
    }
    ratings = heuristic_ratings(background)
    selection = build_selection_from_scores(background, ratings)
    assert selection["experience_selections"]
    assert "content_composites" in selection
    assert compute_content_composites(background, ratings)
