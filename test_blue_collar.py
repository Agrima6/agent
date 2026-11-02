"""Blue-collar roles are recognised, and each family is asked its own questions."""
import pytest
import blue_collar
import planner
from retrieval import metadata_filter
from roles import detect_role_type, is_plain_language_role


@pytest.mark.parametrize("role,family", [
    ("Watchman", "security"), ("Security Guard (Night)", "security"), ("Chowkidar", "security"),
    ("Truck Driver", "driving"), ("Delivery Boy", "delivery"), ("Warehouse Loader", "warehouse"),
    ("Housekeeping Attendant", "housekeeping"), ("Electrician", "electrical"), ("Plumber", "plumbing"),
    ("Welder", "welding"), ("Mason", "construction"), ("Machine Operator", "factory"), ("Cook", "kitchen"),
    ("Tailor", "tailoring"), ("Gardener", "agriculture"), ("Nanny", "domestic"), ("Helper", "general"),
])
def test_role_is_recognised_as_blue_collar_with_its_family(role, family):
    assert detect_role_type(role) == "blue_collar"
    assert blue_collar.family_for(role) == family
    assert is_plain_language_role(role)


@pytest.mark.parametrize("role", ["Backend Engineer", "Product Manager", "Customer Support Executive", "Data Analyst"])
def test_office_roles_are_not_blue_collar(role):
    assert detect_role_type(role) != "blue_collar"


def test_every_family_has_enough_own_questions():
    for family in blue_collar.FAMILIES:
        own = [q for q in blue_collar.FAMILY_QUESTIONS if f"bc_{family}" in q["role_tags"]]
        assert len(own) >= 2, family


def test_watchman_gets_security_questions_not_loader_ones():
    picked = metadata_filter(planner.QUESTION_BANK, "blue_collar:security")
    assert picked and all("bc_security" in q["role_tags"] or "bc_" not in " ".join(q["role_tags"]) for q in picked)
    assert not any("bc_warehouse" in q["role_tags"] for q in picked)
    assert not any("general" in q["role_tags"] and "blue_collar" not in q["role_tags"] for q in picked)


def test_question_ids_unique():
    ids = [q["id"] for q in planner.QUESTION_BANK]
    assert len(ids) == len(set(ids))
