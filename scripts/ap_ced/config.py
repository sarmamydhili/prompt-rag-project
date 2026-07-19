"""Subject configuration for AP Course and Exam Description (CED) extraction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UnitMeta:
    name: str
    class_periods: Optional[int] = None


@dataclass
class ExtractOptions:
    """Toggle optional CED sections. Other AP subjects may omit some of these."""

    include_skill_categories: bool = True
    include_essential_knowledge: bool = True
    include_unit_scenarios: bool = True
    include_topic_scenario_links: bool = True
    include_weightage: bool = True
    units: Optional[List[int]] = None  # None = all configured units


@dataclass
class SubjectConfig:
    """Per-subject CED extraction settings."""

    subject: str
    slug: str
    units: Dict[int, UnitMeta]
    topic_titles: Dict[str, str]
    skill_categories: Optional[List[dict]] = None
    manual_los: Dict[str, str] = field(default_factory=dict)
    unit_header_names: List[str] = field(default_factory=list)
    footer_prefix: str = ""
    scenario_title_overrides: Dict[str, str] = field(default_factory=dict)
    first_topic_id: str = "1.1"
    last_topic_id: str = ""

    def unit_numbers(self) -> List[int]:
        return sorted(self.units)


def get_subject_config(slug: str) -> SubjectConfig:
    from .subjects import SUBJECT_REGISTRY

    key = slug.strip().lower().replace(" ", "-").replace("_", "-")
    if key not in SUBJECT_REGISTRY:
        known = ", ".join(sorted(SUBJECT_REGISTRY))
        raise KeyError(f"Unknown subject '{slug}'. Known: {known}")
    return SUBJECT_REGISTRY[key]


def list_subjects() -> List[str]:
    from .subjects import SUBJECT_REGISTRY

    return sorted(SUBJECT_REGISTRY)
