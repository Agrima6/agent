"""Maps a free-text role name (whatever HR types in) to a role *type* used to pick both
competency weights and which question-bank questions are eligible — so "Frontend Developer"
and "Product Manager" actually get different interviews, not the same generic bank."""

ROLE_TYPE_KEYWORDS = {
    "product_manager": ["product manager", "product owner", " pm ", "^pm$", "product lead"],
    "frontend": ["frontend", "front-end", "front end", "ui engineer", "ux engineer", "react ",
                 "angular ", "vue ", "web developer", "javascript developer", "ui/ux"],
    "fullstack": ["fullstack", "full-stack", "full stack", "mern", "mean stack"],
    "backend": ["backend", "back-end", "back end", "server-side", "api engineer", "java developer",
                "python developer", "golang", "go developer", "node developer", "node.js"],
    "data": ["data scientist", "data engineer", "ml engineer", "machine learning", "ai engineer",
             "data analyst"],
    "devops": ["devops", "sre", "site reliability", "platform engineer", "infrastructure engineer",
               "cloud engineer"],
    "support": ["support", "customer success", "customer service"],
    "blue_collar": [
        "factory worker", "machine operator", "assembly line", "warehouse", "labourer", "laborer",
        "driver", "delivery boy", "delivery partner", "electrician", "plumber", "welder",
        "technician", "mechanic", "fitter", "helper", "loader", "packer", "housekeeping",
        "security guard", "watchman", "carpenter", "mason", "field worker", "site worker",
        "construction worker", "forklift operator", "tailor", "cook", "kitchen staff",
    ],
}

# Role types where the candidate is more likely to be more comfortable with simple, spoken,
# everyday language than corporate/technical English or formal Hindi — the interviewer's
# vocabulary and sentence complexity should adapt accordingly (see agent.py PLAIN_LANGUAGE_TYPES).
PLAIN_LANGUAGE_ROLE_TYPES = {"blue_collar"}

ROLE_COMPETENCY_PRESETS = {
    "backend": [
        {"key": "technical_depth", "weight": 0.30},
        {"key": "problem_solving", "weight": 0.25},
        {"key": "practical_application", "weight": 0.20},
        {"key": "communication", "weight": 0.15},
        {"key": "ownership", "weight": 0.10},
    ],
    "frontend": [
        {"key": "technical_depth", "weight": 0.25},
        {"key": "practical_application", "weight": 0.25},
        {"key": "problem_solving", "weight": 0.20},
        {"key": "communication", "weight": 0.15},
        {"key": "ownership", "weight": 0.15},
    ],
    "fullstack": [
        {"key": "technical_depth", "weight": 0.28},
        {"key": "problem_solving", "weight": 0.22},
        {"key": "practical_application", "weight": 0.22},
        {"key": "communication", "weight": 0.15},
        {"key": "ownership", "weight": 0.13},
    ],
    "product_manager": [
        {"key": "prioritization", "weight": 0.25},
        {"key": "stakeholder_communication", "weight": 0.25},
        {"key": "problem_solving", "weight": 0.20},
        {"key": "execution_ownership", "weight": 0.15},
        {"key": "strategic_thinking", "weight": 0.15},
    ],
    "data": [
        {"key": "technical_depth", "weight": 0.30},
        {"key": "problem_solving", "weight": 0.30},
        {"key": "practical_application", "weight": 0.20},
        {"key": "communication", "weight": 0.20},
    ],
    "devops": [
        {"key": "technical_depth", "weight": 0.30},
        {"key": "problem_solving", "weight": 0.25},
        {"key": "ownership", "weight": 0.20},
        {"key": "communication", "weight": 0.15},
        {"key": "practical_application", "weight": 0.10},
    ],
    "support": [
        {"key": "customer_handling", "weight": 0.35},
        {"key": "communication", "weight": 0.30},
        {"key": "problem_solving", "weight": 0.20},
        {"key": "ownership", "weight": 0.15},
    ],
    "general": [
        {"key": "technical_depth", "weight": 0.30},
        {"key": "problem_solving", "weight": 0.25},
        {"key": "practical_application", "weight": 0.20},
        {"key": "communication", "weight": 0.15},
        {"key": "ownership", "weight": 0.10},
    ],
    "blue_collar": [
        {"key": "reliability", "weight": 0.30},
        {"key": "safety_awareness", "weight": 0.25},
        {"key": "teamwork", "weight": 0.20},
        {"key": "following_instructions", "weight": 0.15},
        {"key": "practical_skill", "weight": 0.10},
    ],
}


def is_plain_language_role(role_name: str) -> bool:
    return detect_role_type(role_name) in PLAIN_LANGUAGE_ROLE_TYPES


def detect_role_type(role_name: str) -> str:
    name = f" {role_name.lower()} "
    for role_type, keywords in ROLE_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw.startswith("^") and kw.endswith("$"):
                if name.strip() == kw.strip("^$"):
                    return role_type
            elif kw in name:
                return role_type
    return "general"


def competencies_for_role(role_name: str) -> list[dict]:
    role_type = detect_role_type(role_name)
    return ROLE_COMPETENCY_PRESETS.get(role_type, ROLE_COMPETENCY_PRESETS["general"])
