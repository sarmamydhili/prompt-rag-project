"""Reusable AP Course and Exam Description (CED) framework extractor."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

from .config import ExtractOptions, SubjectConfig

LO_LINE = re.compile(r"^(?P<code>\d+\.\d+\.[A-Z])\s*$")
EK_LINE = re.compile(r"^(?P<code>\d+\.\d+\.[A-Z]\.\d+)\s*$")
VERB_START = re.compile(
    r"^(Identify|Explain|Describe|Assess|Configure|Determine|Apply|Document|Evaluate|"
    r"Implement|Detect|Calculate|Compare|Create|Analyze|Select|Justify|Represent|"
    r"Interpret|Define|Estimate|Provide|Use|Make|Develop|Complete|Work)\b"
)


def normalize(text: str) -> str:
    text = text.replace("\u2002", " ").replace("\u2001", " ").replace("\xa0", " ")
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_noise_line(line: str, config: SubjectConfig, *, allow_bullets: bool = False) -> bool:
    line = line.strip()
    if not line:
        return True
    if re.fullmatch(r"\d{1,3}", line):
        return True
    if line in {
        "ESSENTIAL KNOWLEDGE",
        "LEARNING OBJECTIVE",
        "Required Course Content",
        "SUGGESTED SKILLS",
        "ILLUSTRATIVE EXAMPLES",
        "UNIT",
        "return to contents",
    }:
        return True
    if config.footer_prefix and line.startswith(config.footer_prefix):
        return True
    if line in set(config.unit_header_names):
        return True
    if re.fullmatch(r"\d\.[A-Z]", line):
        return True
    if line.startswith("Bullet ") or line.startswith("§"):
        return not allow_bullets
    return False


def is_unit_glance_page(text: str) -> bool:
    has_header = "UNIT AT A GLANCE" in text or (
        "UNIT AT A G" in text and "Scenario Connections" in text
    )
    return has_header and "Scenario Connections" in text


def get_framework_slice(doc: fitz.Document, config: SubjectConfig) -> str:
    full = "\n".join(page.get_text() for page in doc)
    first = re.escape(config.first_topic_id)
    starts = [m.start() for m in re.finditer(rf"TOPIC\s*{first}\b", full)]
    if not starts:
        raise ValueError(f"Could not find TOPIC {config.first_topic_id} in PDF")
    start = starts[0]
    last_id = config.last_topic_id or max(config.topic_titles)
    lasts = list(re.finditer(rf"TOPIC\s*{re.escape(last_id)}\b", full))
    if not lasts:
        raise ValueError(f"Could not find TOPIC {last_id} in PDF")
    framework = full[start : lasts[-1].start() + 30000]
    cut = re.search(r"\nANSWER KEY AND QUESTION ALIGNMENT|\nSCORING GUIDELINES\n", framework)
    if cut:
        framework = framework[: cut.start()]
    return framework


def parse_lo_counts_from_glance(doc: fitz.Document) -> Dict[str, int]:
    letters_by_topic: Dict[str, set] = defaultdict(set)
    for page in doc:
        text = page.get_text()
        if not is_unit_glance_page(text):
            continue
        for match in re.finditer(r"(\d+\.\d+)\.([A-Z])\b", text):
            letters_by_topic[match.group(1)].add(match.group(2))
    return {
        topic_id: ord(max(letters)) - ord("A") + 1
        for topic_id, letters in letters_by_topic.items()
        if letters
    }


def parse_lo_counts_from_framework(framework: str) -> Dict[str, int]:
    letters_by_topic: Dict[str, set] = defaultdict(set)
    for match in re.finditer(r"(?m)^(\d+\.\d+)\.([A-Z])\s*$", framework):
        letters_by_topic[match.group(1)].add(match.group(2))
    return {
        topic_id: ord(max(letters)) - ord("A") + 1
        for topic_id, letters in letters_by_topic.items()
        if letters
    }


def get_course_at_a_glance_pages(doc: fitz.Document) -> List[fitz.Page]:
    pages: List[fitz.Page] = []
    for page in doc:
        text = page.get_text()
        if ("Course at" in text and "COURSE SKILLS" in text) or (
            "Skill" in text and re.search(r"\t3\.1\t|\t4\.1\t|\t5\.1\t", text)
        ):
            pages.append(page)
    for page in doc:
        text = page.get_text()
        if "Securing Networks" in text and "\t3.1\t" in text and "Skill" in text:
            if page not in pages:
                pages.append(page)
    return pages


def parse_topic_skill_map(doc: fitz.Document) -> Dict[str, int]:
    topic_skill: Dict[str, int] = {}
    for page in get_course_at_a_glance_pages(doc):
        events = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                y = line["bbox"][1]
                for span in line.get("spans", []):
                    label = span.get("text", "").strip()
                    x = span["bbox"][0]
                    size = span.get("size", 0)
                    if re.fullmatch(r"[1-9]\.\d+", label) and size >= 6:
                        events.append((y, x, "topic", label))
                    if label in {"1", "2", "3", "4"} and 6 <= size <= 8 and 50 <= x <= 450:
                        events.append((y, x, "skill", int(label)))
        events.sort(key=lambda item: (item[0], item[1]))
        topics = [(y, x, value) for y, x, kind, value in events if kind == "topic"]
        skills = [(y, x, value) for y, x, kind, value in events if kind == "skill"]
        for topic_y, topic_x, topic_id in topics:
            nearby = [
                skill
                for skill_y, skill_x, skill in skills
                if abs(skill_x - (topic_x - 20)) < 40 and abs(skill_y - topic_y) < 20
            ]
            if nearby:
                topic_skill[str(topic_id)] = int(nearby[0])
            else:
                column_skills = [
                    (skill_y, skill)
                    for skill_y, skill_x, skill in skills
                    if abs(skill_x - (topic_x - 20)) < 40 and skill_y <= topic_y + 5
                ]
                if column_skills:
                    column_skills.sort(key=lambda item: abs(item[0] - topic_y))
                    topic_skill[str(topic_id)] = int(column_skills[0][1])
    return topic_skill


def parse_course_at_a_glance_skills(
    doc: fitz.Document, config: SubjectConfig, lo_counts: Dict[str, int]
) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    text = "\n".join(page.get_text() for page in get_course_at_a_glance_pages(doc))
    start = text.find("Teach")
    end = text.find("UNIT \n2")
    if end < 0 and config.unit_header_names:
        # Fall back to second configured unit name
        second = list(config.units.values())[1].name if len(config.units) > 1 else ""
        end = text.find(second) if second else -1
    if start >= 0 and end > start:
        chunk = text[start:end]
        idx = chunk.find("1.1")
        if idx >= 0:
            unit1_skills = [
                int(value) for value in re.findall(r"^\s*([1-4])\s*$", chunk[idx:], re.M)
            ]
            unit1_codes: List[str] = []
            for tid in config.topic_titles:
                if not tid.startswith("1."):
                    continue
                for index in range(lo_counts.get(tid, 0)):
                    unit1_codes.append(f"{tid}.{chr(ord('A') + index)}")
            for code, skill in zip(unit1_codes, unit1_skills):
                mapping[code] = skill

    topic_skills = parse_topic_skill_map(doc)
    for tid, count in lo_counts.items():
        if tid.startswith("1.") and any(c.startswith("1.") for c in mapping):
            continue
        skill = topic_skills.get(tid)
        if skill is None:
            continue
        for index in range(count):
            mapping[f"{tid}.{chr(ord('A') + index)}"] = skill
    return mapping


def parse_unit_glance_metadata(
    doc: fitz.Document, config: SubjectConfig, unit_num: int
) -> Tuple[Dict[str, str], Dict[str, str]]:
    scenarios: Dict[str, str] = {}
    lo_descriptions: Dict[str, str] = {}
    unit_name = config.units[unit_num].name

    for page in doc:
        text = page.get_text()
        if not is_unit_glance_page(text):
            continue
        if unit_name not in text and f"UNIT\n{unit_num}" not in text.replace(" ", ""):
            continue

        has_readable_lo = any(
            span.get("size", 0) >= 7
            for block in page.get_text("dict").get("blocks", [])
            if block.get("type") == 0
            for line in block.get("lines", [])
            for span in line.get("spans", [])
            if re.fullmatch(r"\d+\.\d+\.[A-Z]", span.get("text", "").strip())
        )
        if not has_readable_lo and "UNIT AT A GLANCE" in text:
            continue

        unit_topics = [tid for tid in config.topic_titles if tid.startswith(f"{unit_num}.")]
        topic_boundary = rf"(?=\n{unit_num}\.\d+(?:[\t ]|\n)|$)"

        for topic_id in unit_topics:
            block_match = re.search(
                rf"{re.escape(topic_id)}(?:[\t ]|\n)[\s\S]*?{topic_boundary}",
                text,
            )
            if not block_match:
                continue
            scenario_match = re.search(rf"\n({unit_num}[A-Z])[\t ]", block_match.group(0))
            if scenario_match:
                scenarios[topic_id] = scenario_match.group(1)

        for topic_id in unit_topics:
            block_match = re.search(
                rf"{re.escape(topic_id)}(?:[\t ]|\n)[\s\S]*?{topic_boundary}",
                text,
            )
            if not block_match:
                continue
            block = block_match.group(0)
            codes = re.findall(rf"({re.escape(topic_id)}\.[A-Z])\b", block)
            ordered_codes: List[str] = []
            for code in codes:
                if code not in ordered_codes:
                    ordered_codes.append(code)
            collapsed = normalize(block)
            descs = re.findall(
                r"((?:Identify|Explain|Describe|Assess|Configure|Determine|Apply|Document|Evaluate|"
                r"Implement|Detect|Calculate|Compare|Create|Analyze|Select|Justify|Represent|"
                r"Interpret|Define|Estimate|Provide|Use|Make|Develop|Complete|Work)\b[^0-9]*?)"
                rf"(?=\s*(?:Identify|Explain|Describe|Assess|Configure|Determine|Apply|Document|Evaluate|"
                r"Implement|Detect|Calculate|Compare|Create|Analyze|Select|Justify|Represent|"
                r"Interpret|Define|Estimate|Provide|Use|Make|Develop|Complete|Work)\b|"
                rf"\s*{unit_num}[A-Z]\b|\s*Class Periods|\s*\d+\s*$|$)",
                collapsed,
            )
            cleaned = []
            for desc in descs:
                desc = re.split(
                    r"\b(?:Read Scenario|Reflect on|Analyze the|Research how|Produce a|Review physical|Recommend )\b",
                    normalize(desc),
                )[0]
                desc = normalize(desc)
                if len(desc) >= 10:
                    cleaned.append(desc)
            if len(ordered_codes) == len(cleaned):
                for code, desc in zip(ordered_codes, cleaned):
                    lo_descriptions[code] = desc

        if len(scenarios) < len(unit_topics):
            scenario_ids = re.findall(rf"\n({unit_num}[A-Z])[\t ]", text)
            ordered_scenarios: List[str] = []
            for sid in scenario_ids:
                if sid not in ordered_scenarios:
                    ordered_scenarios.append(sid)
            if len(ordered_scenarios) == len(unit_topics):
                for topic_id, sid in zip(unit_topics, ordered_scenarios):
                    scenarios[topic_id] = sid
            elif unit_num == 1 and len(ordered_scenarios) == 5:
                for topic_id, sid in zip(unit_topics, ordered_scenarios):
                    scenarios[topic_id] = sid
            elif unit_num == 2 and ordered_scenarios:
                for topic_id in ["2.2", "2.3", "2.4"]:
                    if topic_id in config.topic_titles:
                        scenarios[topic_id] = ordered_scenarios[0]

        collapsed = normalize(text)
        for match in re.finditer(
            rf"({unit_num}\.\d+\.[A-Z])\s+([A-Z][^0-9]*?)"
            rf"(?=\s*{unit_num}\.\d+\.[A-Z]\b|\s*[1-9][A-Z]\s|\s*Class Periods|$)",
            collapsed,
        ):
            code, desc = match.group(1), normalize(match.group(2))
            desc = re.split(
                r"\b(?:Read Scenario|Reflect on|Analyze the|Research how|Produce a)\b",
                desc,
            )[0]
            desc = normalize(desc)
            if len(desc) >= 10:
                if code not in lo_descriptions or len(desc) > len(lo_descriptions[code]):
                    lo_descriptions[code] = desc

    return scenarios, lo_descriptions


def parse_topic_pages(
    framework: str, config: SubjectConfig, *, include_ek: bool
) -> Tuple[Dict[str, str], Dict[str, List[dict]]]:
    lo_descriptions: Dict[str, str] = {}
    essential_knowledge: Dict[str, List[dict]] = defaultdict(list)
    lines = framework.splitlines()
    i = 0
    while i < len(lines):
        lo_match = LO_LINE.match(lines[i].strip()) if lines[i].strip() else None
        if lo_match:
            code = lo_match.group("code")
            j = i + 1
            desc_parts: List[str] = []
            while j < len(lines):
                raw = lines[j]
                if LO_LINE.match(raw.strip()) or EK_LINE.match(raw.strip()):
                    break
                if raw.strip() in {"ESSENTIAL KNOWLEDGE", "LEARNING OBJECTIVE"} and not desc_parts:
                    break
                if raw.strip() in {"ESSENTIAL KNOWLEDGE", "SUGGESTED SKILLS", "ILLUSTRATIVE EXAMPLES"} and desc_parts:
                    break
                if is_noise_line(raw, config) and not desc_parts:
                    j += 1
                    continue
                if is_noise_line(raw, config) and desc_parts:
                    break
                if raw.strip().startswith("ILLUSTRATIVE"):
                    break
                if re.match(r"^\d+\.\d+\s+[A-Z]", raw.strip()) and desc_parts:
                    break
                desc_parts.append(raw.strip())
                j += 1
                joined = normalize(" ".join(desc_parts))
                if joined.endswith(".") and VERB_START.match(joined):
                    break
            desc = normalize(" ".join(desc_parts))
            if desc and VERB_START.match(desc) and len(desc) >= 10:
                sentences = re.split(r"(?<=\.)\s+", desc)
                if len(sentences) > 1 and sum(1 for s in sentences if VERB_START.match(s)) >= 3:
                    desc = sentences[0]
                if code not in lo_descriptions or len(desc) > len(lo_descriptions[code]):
                    lo_descriptions[code] = desc
            i = max(j, i + 1)
            continue

        ek_match = EK_LINE.match(lines[i].strip()) if lines[i].strip() else None
        if ek_match:
            if not include_ek:
                i += 1
                continue
            codes = [ek_match.group("code")]
            j = i + 1
            while j < len(lines) and EK_LINE.match(lines[j].strip() or ""):
                codes.append(EK_LINE.match(lines[j].strip()).group("code"))
                j += 1

            parts: List[str] = []
            raw_parts: List[str] = []
            while j < len(lines):
                raw = lines[j].strip()
                if LO_LINE.match(raw) or EK_LINE.match(raw):
                    break
                if raw in {"ESSENTIAL KNOWLEDGE", "LEARNING OBJECTIVE", "SUGGESTED SKILLS"}:
                    if parts:
                        break
                    j += 1
                    continue
                if raw.startswith("Bullet"):
                    line_text = raw.replace("Bullet ", "").strip()
                    parts.append(line_text)
                    raw_parts.append(line_text)
                    j += 1
                    continue
                if raw.startswith("§"):
                    line_text = raw.lstrip("§ ").strip()
                    parts.append(line_text)
                    raw_parts.append(line_text)
                    j += 1
                    continue
                if is_noise_line(raw, config, allow_bullets=True) and not parts:
                    j += 1
                    continue
                if is_noise_line(raw, config, allow_bullets=True) and parts:
                    break
                if raw.startswith("TOPIC"):
                    break
                parts.append(raw)
                raw_parts.append(raw)
                j += 1

            if parts:
                parent = codes[0].rsplit(".", 1)[0]
                if len(codes) > 1:
                    paragraphs: List[str] = []
                    current: List[str] = []
                    for line in raw_parts:
                        stripped = line.strip()
                        if not stripped:
                            if current:
                                paragraphs.append(normalize(" ".join(current)))
                                current = []
                            continue
                        starts_new = bool(
                            VERB_START.match(stripped)
                            or re.match(
                                r"^(Most|However|Individuals|When|Users|Adversaries|Victims)\b",
                                stripped,
                            )
                        )
                        if starts_new and current and normalize(" ".join(current)).endswith("."):
                            paragraphs.append(normalize(" ".join(current)))
                            current = [stripped]
                        else:
                            current.append(stripped)
                    if current:
                        paragraphs.append(normalize(" ".join(current)))
                    if len(paragraphs) >= len(codes):
                        for code, paragraph in zip(codes, paragraphs[: len(codes)]):
                            bucket = essential_knowledge[parent]
                            if not any(item["code"] == code for item in bucket):
                                bucket.append({"code": code, "description": paragraph})
                    else:
                        description = normalize(" ".join(parts))
                        for code in codes:
                            bucket = essential_knowledge[parent]
                            if not any(item["code"] == code for item in bucket):
                                bucket.append({"code": code, "description": description})
                else:
                    code = codes[0]
                    description = normalize(" ".join(parts))
                    bucket = essential_knowledge[parent]
                    if not any(item["code"] == code for item in bucket):
                        bucket.append({"code": code, "description": description})
            i = max(j, i + 1)
            continue
        i += 1

    for parent in essential_knowledge:
        essential_knowledge[parent].sort(key=lambda item: item["code"])
    return lo_descriptions, essential_knowledge


def compute_weightage(config: SubjectConfig, unit_num: int) -> Optional[int]:
    periods = {
        n: meta.class_periods
        for n, meta in config.units.items()
        if meta.class_periods is not None
    }
    if unit_num not in periods or not periods:
        return None
    total = sum(periods.values())
    if total <= 0:
        return None
    weights = {n: round(value / total * 100) for n, value in periods.items()}
    drift = 100 - sum(weights.values())
    if drift:
        weights[max(weights, key=weights.get)] += drift
    return weights[unit_num]


def parse_unit_scenarios(doc: fitz.Document, config: SubjectConfig, unit_num: int) -> List[dict]:
    full = "\n".join(page.get_text() for page in doc)
    max_unit = max(config.units)
    pattern = re.compile(
        rf"SCENARIO\s+(?P<id>{unit_num}[A-Z]):\s*\n(?P<rest>[\s\S]*?)(?=\nSCENARIO\s+[1-{max_unit}][A-Z]:|\nUNIT AT A|\Z)"
    )

    def clean_body(text: str) -> str:
        cleaned_lines: List[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if re.fullmatch(r"\d{1,3}", stripped):
                continue
            if config.footer_prefix and stripped.startswith(config.footer_prefix):
                continue
            if stripped == "return to contents":
                continue
            if stripped in set(config.unit_header_names) or stripped == "UNIT":
                continue
            if re.fullmatch(r"[1-9]", stripped):
                continue
            cleaned_lines.append(stripped)
        body_text = "\n".join(cleaned_lines)
        body_text = re.sub(r"\n{3,}", "\n\n", body_text)
        body_text = re.sub(r"[ \t]*Bullet[ \t]+", " • ", body_text)
        body_text = re.sub(r"[ \t]*§[ \t]+", " • ", body_text)
        return normalize(body_text)

    def looks_like_title(line: str) -> bool:
        line = normalize(line)
        if not line or len(line) > 80:
            return False
        if line.endswith(".") or line.endswith(":"):
            return False
        if line.startswith("Bullet") or line.startswith("§") or line.startswith("•"):
            return False
        if re.match(
            r"^(You |In |As |Sitting |The company |Submarines |First,|Consider |Most |However )",
            line,
        ):
            return False
        words = line.split()
        return 2 <= len(words) <= 12

    by_id: Dict[str, dict] = {}
    for match in pattern.finditer(full):
        sid = match.group("id")
        rest = match.group("rest").strip("\n")
        lines = rest.splitlines()
        if not lines:
            continue

        title = normalize(lines[0])
        body_lines = lines[1:]
        if not looks_like_title(title):
            found_title = None
            for index, line in enumerate(lines[:20]):
                candidate = normalize(line.replace("\t", " "))
                candidate = re.sub(r"^Bullet\s+", "", candidate)
                if looks_like_title(candidate):
                    found_title = (index, candidate)
                    if re.search(
                        r"\b(Protecting|Securing|Detecting|Configuring|Designing|Analyzing|Sending|Verifying)\b",
                        candidate,
                    ):
                        break
            if found_title:
                index, title = found_title
                body_lines = lines[:index] + lines[index + 1 :]
            else:
                body_lines = lines
                title = sid

        if sid in config.scenario_title_overrides:
            title = config.scenario_title_overrides[sid]

        body_text = clean_body("\n".join(body_lines))
        if len(body_text) < 40:
            continue
        prev = by_id.get(sid)
        if prev is None or len(body_text) > len(prev["body"]):
            by_id[sid] = {"id": sid, "title": title, "body": body_text}

    return [by_id[key] for key in sorted(by_id, key=lambda value: value)]


def extract_unit(
    doc: fitz.Document,
    framework: str,
    config: SubjectConfig,
    options: ExtractOptions,
    unit_num: int,
    skill_map: Dict[str, int],
    lo_counts: Dict[str, int],
) -> dict:
    topic_scenarios, glance_los = parse_unit_glance_metadata(doc, config, unit_num)
    topic_los, ek_map = parse_topic_pages(
        framework, config, include_ek=options.include_essential_knowledge
    )
    for code, desc in config.manual_los.items():
        topic_los[code] = desc

    topics_out = []
    for topic_id, title in config.topic_titles.items():
        if not topic_id.startswith(f"{unit_num}."):
            continue
        count = lo_counts.get(topic_id, 0)
        letters = [chr(ord("A") + index) for index in range(count)]
        objectives = []
        for letter in letters:
            code = f"{topic_id}.{letter}"
            desc = topic_los.get(code) or glance_los.get(code)
            if not desc:
                continue
            if code in glance_los and code not in topic_los and desc.count(". ") >= 2:
                first = desc.split(". ")[0] + "."
                if len(first) >= 10:
                    desc = first
            obj = {"code": code, "description": desc}
            if options.include_skill_categories and skill_map.get(code) is not None:
                obj["skill_category"] = skill_map[code]
            if options.include_essential_knowledge:
                obj["essential_knowledge"] = ek_map.get(code, [])
            objectives.append(obj)

        topic_obj: dict = {"topic": title, "objectives": objectives}
        if options.include_topic_scenario_links and topic_id in topic_scenarios:
            topic_obj["scenario"] = topic_scenarios[topic_id]
        topics_out.append(topic_obj)

    meta = config.units[unit_num]
    unit_obj: dict = {
        "unit": meta.name,
        "unit_code": f"Unit {unit_num}",
        "topics": topics_out,
    }
    if options.include_weightage:
        weight = compute_weightage(config, unit_num)
        if weight is not None:
            unit_obj["weightage_percent"] = weight
    if options.include_unit_scenarios:
        unit_obj["scenarios"] = parse_unit_scenarios(doc, config, unit_num)
    return unit_obj


def build_document(
    doc: fitz.Document,
    config: SubjectConfig,
    options: Optional[ExtractOptions] = None,
) -> dict:
    options = options or ExtractOptions()
    framework = get_framework_slice(doc, config)
    glance_counts = parse_lo_counts_from_glance(doc)
    framework_counts = parse_lo_counts_from_framework(framework)
    lo_counts: Dict[str, int] = {}
    for tid in set(framework_counts) | set(glance_counts):
        lo_counts[tid] = max(framework_counts.get(tid, 0), glance_counts.get(tid, 0))

    skill_map: Dict[str, int] = {}
    if options.include_skill_categories and config.skill_categories:
        skill_map = parse_course_at_a_glance_skills(doc, config, lo_counts)

    unit_nums = options.units or config.unit_numbers()
    payload: dict = {
        "subject": config.subject,
        "units": [
            extract_unit(doc, framework, config, options, unit_num, skill_map, lo_counts)
            for unit_num in unit_nums
            if unit_num in config.units
        ],
    }
    if options.include_skill_categories and config.skill_categories:
        payload["skill_categories"] = config.skill_categories
    return payload


def extract_from_pdf(
    pdf_path: Path,
    config: SubjectConfig,
    options: Optional[ExtractOptions] = None,
) -> dict:
    doc = fitz.open(pdf_path)
    try:
        return build_document(doc, config, options)
    finally:
        doc.close()
