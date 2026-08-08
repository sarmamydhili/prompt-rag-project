"""Simple AP CED PDF → course_framework JSON extractor (no subject registry)."""

from __future__ import annotations

import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz

LO_LINE = re.compile(r"^(?P<code>\d+\.\d+\.[A-Z])\s*$")
EK_LINE = re.compile(r"^(?P<code>\d+\.\d+\.[A-Z]\.\d+)\s*$")
TOPIC_HEADER = re.compile(
    r"TOPIC\s+(?P<id>\d+\.\d+)\s*\n(?P<title>[^\n]+(?:\n(?![A-Z]{3,}|TOPIC|UNIT|Required|LEARNING|ESSENTIAL|SUGGESTED)[^\n]+){0,4})",
    re.M,
)
VERB_START = re.compile(
    r"^(Identify|Explain|Describe|Assess|Configure|Determine|Apply|Document|Evaluate|"
    r"Implement|Detect|Calculate|Compare|Create|Analyze|Select|Justify|Represent|"
    r"Interpret|Define|Estimate|Provide|Use|Make|Develop|Complete|Work|"
    r"Understand|Compare|Discuss|Construct|Propose|Recommend|Calculate)\b"
)


def normalize(text: str) -> str:
    text = text.replace("\u2002", " ").replace("\u2001", " ").replace("\xa0", " ")
    text = re.sub(r"(\w)-\s+(\w)", r"\1-\2", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def full_text(doc: fitz.Document) -> str:
    return "\n".join(page.get_text() for page in doc)


def derive_subject(doc: fitz.Document) -> str:
    """Derive 'AP …' subject name from the cover pages."""
    cover = "\n".join(page.get_text() for page in doc[:3])
    # Normalize trademark / odd spaces
    cover_n = cover.replace("®", " ").replace("\u2002", " ").replace("\u2001", " ")
    m = re.search(
        r"AP\s+(.+?)\s+COURSE\s+AND\s+EXAM\s+DESCRIPTION",
        cover_n,
        re.I | re.S,
    )
    if not m:
        raise ValueError("Could not derive AP subject from PDF cover")
    name = normalize(m.group(1))
    # Drop trailing junk
    name = re.split(r"\bINCLUDES\b", name, maxsplit=1)[0].strip()
    if not name.lower().startswith("ap"):
        name = f"AP {name}"
    return normalize(name)


def is_unit_glance_page(text: str) -> bool:
    has_header = "UNIT AT A GLANCE" in text or (
        "UNIT AT A G" in text and ("Scenario Connections" in text or "Learning Objectives" in text)
    )
    return has_header and ("Learning Objectives" in text or "Scenario Connections" in text)


def discover_topics(doc: fitz.Document) -> OrderedDict:
    """Discover topic_id → title from TOPIC headers across the PDF."""
    text = full_text(doc)
    topics: OrderedDict[str, str] = OrderedDict()
    for m in TOPIC_HEADER.finditer(text):
        tid = m.group("id")
        title = normalize(m.group("title"))
        title = re.sub(r"\bRequired Course Content\b.*$", "", title).strip()
        title = re.sub(r"\bLEARNING OBJECTIVE\b.*$", "", title).strip()
        if not title or len(title) < 3:
            continue
        # Prefer first good title; longer is usually better if we re-see it
        if tid not in topics or len(title) > len(topics[tid]):
            topics[tid] = title
    if not topics:
        # Fallback: glance tables "1.1 Title"
        for page in doc:
            page_text = page.get_text()
            if not is_unit_glance_page(page_text) and "Course at" not in page_text:
                continue
            for m in re.finditer(r"(?m)^(\d+\.\d+)\s+([A-Z][^\n]{3,80})", page_text):
                tid, title = m.group(1), normalize(m.group(2))
                if tid not in topics:
                    topics[tid] = title
    if not topics:
        raise ValueError("Could not discover any TOPIC X.Y entries in PDF")
    return OrderedDict(sorted(topics.items(), key=lambda kv: tuple(map(int, kv[0].split(".")))))


def _clean_unit_title(raw_title: str) -> str:
    junk = re.compile(
        r"AP EXAM WEIGHTING|CLASS PERIOD|Developing Understanding|FIGURE|"
        r"FINANCIAL ADVISOR|IoCs|controls\.|Applies to Unit|indicator|"
        r"UNIT AT A GLANCE|return to",
        re.I,
    )
    lines = []
    for line in raw_title.splitlines():
        line = line.strip()
        if not line or junk.search(line):
            continue
        if re.match(r"^\d{1,2}[–-]\d{1,2}%", line):
            continue
        if re.fullmatch(r"\d{1,3}", line):
            continue
        if line.endswith(".") and len(line) > 40:
            continue
        lines.append(line)
    title = normalize(" ".join(lines))
    title = re.sub(r"\s+\d{1,2}[–-]\d{1,2}%.*$", "", title).strip()
    if len(title) > 55:
        chunks = re.split(r"(?<=[.!?])\s+", title)
        title = chunks[-1].strip()
    return title


def _looks_like_unit_title(title: str) -> bool:
    if not title or len(title) < 3 or len(title) > 60:
        return False
    words = title.split()
    if not (1 <= len(words) <= 10):
        return False
    if not title[0].isupper():
        return False
    if title.endswith(".") or title.lower().startswith("students "):
        return False
    return True


def _unit_meta(doc: fitz.Document, unit_nums: List[int]) -> Tuple[Dict[int, str], Dict[int, int]]:
    """Discover unit display names and class-period counts."""
    text = full_text(doc)
    allowed = set(unit_nums)
    title_votes: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    period_vals: Dict[int, List[int]] = defaultdict(list)

    # Footer used on most topic pages: "Title\nUNIT\nN"
    footer = re.compile(
        r"(?P<title>[A-Z][^\n]{2,55}(?:\n(?:and |to |[a-z])[^\n]{1,40})?)\n"
        r"UNIT\s*\n\s*(?P<num>\d+)\b"
    )
    for m in footer.finditer(text):
        n = int(m.group("num"))
        if n not in allowed:
            continue
        title = _clean_unit_title(m.group("title"))
        if _looks_like_unit_title(title):
            title_votes[n][title] += 1

    openers = [
        re.compile(
            r"UNIT\s*\n\s*(?P<num>\d+)\s*\n"
            r"(?P<title>(?:[^\n]+\n){1,4}?)"
            r"(?:[~∼]|Tilde)\s*(?P<periods>\d+)[\u2002\s]*\n?CLASS\s+PERIODS",
            re.I,
        ),
        re.compile(
            r"UNIT\s*\n"
            r"(?P<title>(?:[A-Za-z][^\n]*\n){1,4})"
            r"\s*(?P<num>\d+)\s*\n"
            r"(?:Tilde\s*)?(?P<periods>\d+)[\u2002\s]*\n?CLASS\s+PERIODS",
            re.I,
        ),
        re.compile(
            r"UNIT[\u2003\s–-]*\n?\s*(?P<num>\d+)\s*\n"
            r"(?:Part\s+\d+\s*\n)?"
            r"(?P<title>(?:[A-Za-z][^\n]*\n){1,4})"
            r"(?:[~∼]|Tilde)\s*(?P<periods>\d+)[\u2002\s]*\n?CLASS\s+PERIODS",
            re.I,
        ),
    ]
    for pattern in openers:
        for m in pattern.finditer(text):
            n = int(m.group("num"))
            if n not in allowed:
                continue
            title = _clean_unit_title(m.group("title"))
            if _looks_like_unit_title(title):
                title_votes[n][title] += 3
            period_vals[n].append(int(m.group("periods")))

    names: Dict[int, str] = {}
    for n in unit_nums:
        votes = title_votes.get(n) or {}
        if votes:
            names[n] = sorted(votes.items(), key=lambda kv: (kv[1], len(kv[0])))[-1][0]
        else:
            names[n] = f"Unit {n}"

    # Title-adjacent periods fill gaps only: "Securing Spaces\n~21 CLASS PERIODS"
    for n, title in names.items():
        if n in period_vals:
            continue
        flex = re.escape(title).replace(r"\ ", r"[\s\n]+")
        title_period = re.compile(
            flex
            + r"[\s\S]{0,100}?(?:[~∼]|Tilde)\s*(\d+)[\u2002\s]*CLASS\s+PERIODS",
            re.I,
        )
        found = [int(m.group(1)) for m in title_period.finditer(text)]
        if found:
            counts: Dict[int, int] = defaultdict(int)
            for v in found:
                counts[v] += 1
            period_vals[n] = [sorted(counts.items(), key=lambda kv: kv[1])[-1][0]]

    periods: Dict[int, int] = {}
    for n in unit_nums:
        vals = period_vals.get(n) or []
        if not vals:
            continue
        unique: List[int] = []
        for v in vals:
            if v not in unique:
                unique.append(v)
        # Same value repeated → once; distinct part totals (6+29) → sum
        periods[n] = unique[0] if len(unique) == 1 else sum(unique)

    return names, periods


def discover_unit_names(doc: fitz.Document, unit_nums: List[int]) -> Dict[int, str]:
    return _unit_meta(doc, unit_nums)[0]


def discover_class_periods(doc: fitz.Document, unit_nums: List[int]) -> Dict[int, int]:
    return _unit_meta(doc, unit_nums)[1]


def get_framework_slice(doc: fitz.Document, topic_ids: List[str]) -> str:
    text = full_text(doc)
    first, last = topic_ids[0], topic_ids[-1]
    starts = [m.start() for m in re.finditer(rf"TOPIC\s*{re.escape(first)}\b", text)]
    if not starts:
        raise ValueError(f"Could not find TOPIC {first}")
    lasts = list(re.finditer(rf"TOPIC\s*{re.escape(last)}\b", text))
    if not lasts:
        raise ValueError(f"Could not find TOPIC {last}")
    framework = text[starts[0] : lasts[-1].start() + 30000]
    cut = re.search(r"\nANSWER KEY AND QUESTION ALIGNMENT|\nSCORING GUIDELINES\n", framework)
    if cut:
        framework = framework[: cut.start()]
    return framework


def is_noise_line(line: str, subject: str, unit_names: List[str], *, allow_bullets: bool = False) -> bool:
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
    if subject and line.startswith(subject.split()[0]) and "·" in line:
        return True
    if line.startswith(subject):
        return True
    if line in set(unit_names):
        return True
    if re.fullmatch(r"\d\.[A-Z]", line):
        return True
    if line.startswith("Bullet ") or line.startswith("§"):
        return not allow_bullets
    return False


def parse_lo_counts(doc: fitz.Document, framework: str) -> Dict[str, int]:
    letters: Dict[str, set] = defaultdict(set)
    for page in doc:
        text = page.get_text()
        if is_unit_glance_page(text):
            for m in re.finditer(r"(\d+\.\d+)\.([A-Z])\b", text):
                letters[m.group(1)].add(m.group(2))
    for m in re.finditer(r"(?m)^(\d+\.\d+)\.([A-Z])\s*$", framework):
        letters[m.group(1)].add(m.group(2))
    return {tid: ord(max(ls)) - ord("A") + 1 for tid, ls in letters.items() if ls}


def parse_topic_pages(
    framework: str, subject: str, unit_names: List[str]
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
                if is_noise_line(raw, subject, unit_names) and not desc_parts:
                    j += 1
                    continue
                if is_noise_line(raw, subject, unit_names) and desc_parts:
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
                if is_noise_line(raw, subject, unit_names, allow_bullets=True) and not parts:
                    j += 1
                    continue
                if is_noise_line(raw, subject, unit_names, allow_bullets=True) and parts:
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
                            or re.match(r"^(Most|However|Individuals|When|Users|Adversaries|Victims)\b", stripped)
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


def parse_glance_los_and_scenarios(
    doc: fitz.Document, unit_num: int
) -> Tuple[Dict[str, str], Dict[str, str]]:
    scenarios: Dict[str, str] = {}
    lo_descriptions: Dict[str, str] = {}
    for page in doc:
        text = page.get_text()
        if not is_unit_glance_page(text):
            continue
        if f"UNIT\n{unit_num}" not in text.replace(" ", "") and not re.search(
            rf"UNIT\s*\n\s*{unit_num}\b", text
        ):
            # still allow if topic ids for this unit appear
            if not re.search(rf"\b{unit_num}\.\d+\b", text):
                continue

        unit_topics = sorted(
            {m.group(1) for m in re.finditer(rf"\b({unit_num}\.\d+)\b", text)},
            key=lambda t: tuple(map(int, t.split("."))),
        )
        topic_boundary = rf"(?=\n{unit_num}\.\d+(?:[\t ]|\n)|$)"
        for topic_id in unit_topics:
            block_match = re.search(
                rf"{re.escape(topic_id)}(?:[\t ]|\n)[\s\S]*?{topic_boundary}", text
            )
            if not block_match:
                continue
            scenario_match = re.search(rf"\n({unit_num}[A-Z])[\t ]", block_match.group(0))
            if scenario_match:
                scenarios[topic_id] = scenario_match.group(1)

        collapsed = normalize(text)
        for match in re.finditer(
            rf"({unit_num}\.\d+\.[A-Z])\s+([A-Z][^0-9]*?)"
            rf"(?=\s*{unit_num}\.\d+\.[A-Z]\b|\s*[1-9][A-Z]\s|\s*Class Periods|$)",
            collapsed,
        ):
            code, desc = match.group(1), normalize(match.group(2))
            desc = re.split(r"\b(?:Read Scenario|Reflect on|Analyze the|Research how|Produce a)\b", desc)[0]
            desc = normalize(desc)
            if len(desc) >= 10:
                if code not in lo_descriptions or len(desc) > len(lo_descriptions[code]):
                    lo_descriptions[code] = desc

        # Fallback scenario order zip when per-topic miss
        if unit_topics:
            scenario_ids = []
            for sid in re.findall(rf"\n({unit_num}[A-Z])[\t ]", text):
                if sid not in scenario_ids:
                    scenario_ids.append(sid)
            if len(scenario_ids) == len(unit_topics):
                for topic_id, sid in zip(unit_topics, scenario_ids):
                    scenarios.setdefault(topic_id, sid)

    return scenarios, lo_descriptions


def parse_unit_scenarios(doc: fitz.Document, unit_num: int, subject: str, unit_names: List[str]) -> List[dict]:
    text = full_text(doc)
    if not re.search(rf"SCENARIO\s+{unit_num}[A-Z]:", text):
        return []
    pattern = re.compile(
        rf"SCENARIO\s+(?P<id>{unit_num}[A-Z]):\s*\n(?P<rest>[\s\S]*?)(?=\nSCENARIO\s+[1-9][A-Z]:|\nUNIT AT A|\Z)"
    )

    def clean_body(body: str) -> str:
        cleaned = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped or re.fullmatch(r"\d{1,3}", stripped):
                continue
            if stripped == "return to contents" or stripped == "UNIT":
                continue
            if stripped in set(unit_names) or stripped.startswith(subject.split()[0] + " "):
                continue
            if subject and stripped.startswith(subject):
                continue
            cleaned.append(stripped)
        body_text = "\n".join(cleaned)
        body_text = re.sub(r"[ \t]*Bullet[ \t]+", " • ", body_text)
        body_text = re.sub(r"[ \t]*§[ \t]+", " • ", body_text)
        return normalize(body_text)

    def looks_like_title(line: str) -> bool:
        line = normalize(line)
        if not line or len(line) > 80 or line.endswith(".") or line.endswith(":"):
            return False
        if line.startswith("Bullet") or line.startswith("§") or line.startswith("•"):
            return False
        if re.match(r"^(You |In |As |Sitting |The company |Submarines |First,|Consider )", line):
            return False
        words = line.split()
        return 2 <= len(words) <= 12

    by_id: Dict[str, dict] = {}
    for match in pattern.finditer(text):
        sid = match.group("id")
        lines = match.group("rest").strip("\n").splitlines()
        if not lines:
            continue
        title = normalize(lines[0])
        body_lines = lines[1:]
        if not looks_like_title(title):
            found = None
            for index, line in enumerate(lines[:20]):
                candidate = normalize(re.sub(r"^Bullet\s+", "", line.replace("\t", " ")))
                if looks_like_title(candidate):
                    found = (index, candidate)
                    if re.search(
                        r"\b(Protecting|Securing|Detecting|Configuring|Designing|Analyzing|Sending|Verifying)\b",
                        candidate,
                    ):
                        break
            if found:
                index, title = found
                body_lines = lines[:index] + lines[index + 1 :]
            else:
                body_lines = lines
                title = sid
        # Known layout quirk
        if sid == "3C" and "Naval" not in title:
            title = "Protecting a Network on a Naval Submarine"
        body = clean_body("\n".join(body_lines))
        if len(body) < 40:
            continue
        prev = by_id.get(sid)
        if prev is None or len(body) > len(prev["body"]):
            by_id[sid] = {"id": sid, "title": title, "body": body}
    return [by_id[k] for k in sorted(by_id)]


def try_skill_map(doc: fitz.Document, lo_counts: Dict[str, int]) -> Dict[str, int]:
    """Best-effort skill mapping; empty if not detectable."""
    mapping: Dict[str, int] = {}
    # Unit 1 per-LO stream from Course at a Glance text
    glance_text = ""
    for page in doc:
        t = page.get_text()
        if "Course at" in t and ("COURSE SKILLS" in t or "Skill" in t):
            glance_text += "\n" + t
    if not glance_text:
        return mapping
    start = glance_text.find("Teach")
    end = glance_text.find("UNIT \n2")
    if start >= 0 and end > start:
        chunk = glance_text[start:end]
        idx = chunk.find("1.1")
        if idx >= 0:
            skills = [int(v) for v in re.findall(r"^\s*([1-4])\s*$", chunk[idx:], re.M)]
            codes = []
            for tid in sorted(lo_counts, key=lambda x: tuple(map(int, x.split(".")))):
                if tid.startswith("1."):
                    for i in range(lo_counts[tid]):
                        codes.append(f"{tid}.{chr(ord('A') + i)}")
            for code, skill in zip(codes, skills):
                mapping[code] = skill
    return mapping


def detect_ced_variant(doc: fitz.Document) -> str:
    """
    Two College Board CED layouts:

    - career: AP Cybersecurity / AP Business with Personal Finance style —
      single ~N CLASS PERIODS counts; weightage from period shares.
    - standard: most other AP CEDs (e.g. Physics) — ranged class periods
      ("tilde 12 hyphen 17" / "~12–17 CLASS PERIODS") plus N–M% AP Exam Weighting.
    """
    text = full_text(doc)
    if re.search(r"SCENARIO\s+[1-9][A-Z]:", text):
        return "career"

    single = len(
        re.findall(r"(?:[~∼]|Tilde)\s*\d{1,2}[\u2002\s]*CLASS\s+PERIODS", text, re.I)
    )
    ranged_class = len(
        re.findall(
            r"(?:[~∼]\s*\d+\s*[–-]\s*\d+[\u2002\s]*CLASS\s+PERIODS|"
            r"tilde\s+\d+\s+hyphen\s+\d+)",
            text,
            re.I,
        )
    )

    # Career CEDs use many single-count class-period markers.
    if single >= 3 and ranged_class <= 1:
        return "career"
    if ranged_class >= 2:
        return "standard"
    if single >= 1:
        return "career"
    if re.search(r"\d{1,2}\s*[–-]\s*\d{1,2}%\s*AP\s*Exam\s*Weighting", text, re.I):
        return "standard"
    return "career"


def discover_exam_weight_ranges(
    doc: fitz.Document, unit_nums: List[int]
) -> Dict[int, Tuple[int, int]]:
    """Parse N–M% AP Exam Weighting per unit from Course at a Glance pages."""
    allowed = set(unit_nums)
    chunks: List[str] = []
    for page in doc:
        t = page.get_text()
        if not re.search(r"AP\s*Exam\s*Weighting", t, re.I):
            continue
        if not re.search(r"\d{1,2}\s*[–-]\s*\d{1,2}%", t):
            continue
        if not re.search(r"Class\s*\n?\s*Periods|CLASS\s+PERIODS|Periods\s*\d", t, re.I):
            continue
        # Visual glance pages usually have Progress Check / tilde pacing / several UNITs
        if (
            "Progress Check" in t
            or re.search(r"tilde\s+\d+", t, re.I)
            or t.count("UNIT") >= 2
        ):
            chunks.append(t)

    ranges: Dict[int, Tuple[int, int]] = {}
    glance = "\n".join(chunks)
    if not glance:
        return ranges

    for wm in re.finditer(
        r"(\d{1,2})\s*[–-]\s*(\d{1,2})%\s*AP\s*Exam\s*Weighting", glance, re.I
    ):
        lo, hi = int(wm.group(1)), int(wm.group(2))
        before = glance[max(0, wm.start() - 350) : wm.start()]
        units = list(
            re.finditer(
                r"UNIT\s*\n(?:[A-Za-z][^\n]*(?:\n[A-Za-z/][^\n]*){0,3}\n\s*)?(\d+)\b",
                before,
            )
        )
        if not units:
            continue
        n = int(units[-1].group(1))
        if n in allowed and n not in ranges:
            ranges[n] = (lo, hi)
    return ranges


def compute_weightage(shares: Dict[int, float], unit_num: int) -> Optional[int]:
    """Normalize positive share values to integers that sum to 100."""
    if unit_num not in shares or not shares:
        return None
    total = sum(shares.values())
    if total <= 0:
        return None
    weights = {n: round(v / total * 100) for n, v in shares.items()}
    drift = 100 - sum(weights.values())
    if drift:
        weights[max(weights, key=weights.get)] += drift
    return weights[unit_num]


def compute_weightage_from_periods(periods: Dict[int, int], unit_num: int) -> Optional[int]:
    return compute_weightage({n: float(v) for n, v in periods.items()}, unit_num)


def compute_weightage_from_exam_ranges(
    ranges: Dict[int, Tuple[int, int]], unit_num: int
) -> Optional[int]:
    """Use midpoints of College Board exam-weight ranges, normalized to 100%."""
    if unit_num not in ranges:
        return None
    mids = {n: (lo + hi) / 2.0 for n, (lo, hi) in ranges.items()}
    return compute_weightage(mids, unit_num)


def build_document(doc: fitz.Document, units_filter: Optional[List[int]] = None) -> dict:
    subject = derive_subject(doc)
    topics = discover_topics(doc)
    unit_nums = sorted({int(tid.split(".")[0]) for tid in topics})
    if units_filter:
        unit_nums = [n for n in unit_nums if n in set(units_filter)]
        topics = OrderedDict((k, v) for k, v in topics.items() if int(k.split(".")[0]) in set(units_filter))

    unit_names, periods = _unit_meta(doc, unit_nums)
    variant = detect_ced_variant(doc)
    exam_ranges = (
        discover_exam_weight_ranges(doc, unit_nums) if variant == "standard" else {}
    )
    framework = get_framework_slice(doc, list(topics.keys()))
    lo_counts = parse_lo_counts(doc, framework)
    topic_los, ek_map = parse_topic_pages(framework, subject, list(unit_names.values()))
    skill_map = try_skill_map(doc, lo_counts)

    # Known firewall stacked-LO fix when descriptions mis-aligned
    manual = {
        "3.4.A": "Configure a firewall to manage the flow of network traffic.",
        "3.4.B": "Identify types of network-based firewalls.",
        "3.4.C": "Explain how a firewall uses an access control list to allow or deny traffic entering or leaving a network.",
        "3.4.D": "Determine the effective placement of firewalls in a network.",
    }
    for code, desc in manual.items():
        if code.rsplit(".", 1)[0] in topics:
            topic_los[code] = desc

    units_out = []
    for n in unit_nums:
        topic_scenarios, glance_los = parse_glance_los_and_scenarios(doc, n)
        unit_topics = [(tid, title) for tid, title in topics.items() if tid.startswith(f"{n}.")]
        topics_out = []
        for tid, title in unit_topics:
            count = lo_counts.get(tid, 0)
            # also include any discovered LO letters under this topic
            if count == 0:
                letters = sorted({c.split(".")[-1] for c in topic_los if c.startswith(tid + ".")})
                count = len(letters)
            objectives = []
            for i in range(count):
                letter = chr(ord("A") + i)
                code = f"{tid}.{letter}"
                desc = topic_los.get(code) or glance_los.get(code)
                if not desc:
                    continue
                obj = {"code": code, "description": desc}
                if code in skill_map:
                    obj["skill_category"] = skill_map[code]
                ek = ek_map.get(code, [])
                if ek:
                    obj["essential_knowledge"] = ek
                objectives.append(obj)
            topic_obj: dict = {"topic": title, "objectives": objectives}
            if tid in topic_scenarios:
                topic_obj["scenario"] = topic_scenarios[tid]
            topics_out.append(topic_obj)

        unit_obj: dict = {
            "unit": unit_names.get(n, f"Unit {n}"),
            "unit_code": f"Unit {n}",
            "topics": topics_out,
        }
        if variant == "standard" and exam_ranges:
            weight = compute_weightage_from_exam_ranges(exam_ranges, n)
        else:
            weight = compute_weightage_from_periods(periods, n)
        if weight is not None:
            unit_obj["weightage_percent"] = weight
        scenarios = parse_unit_scenarios(doc, n, subject, list(unit_names.values()))
        if scenarios:
            unit_obj["scenarios"] = scenarios
        units_out.append(unit_obj)

    payload: dict = {"subject": subject, "units": units_out}
    # Only include skill_categories legend if we mapped any skills
    if skill_map:
        payload["skill_categories"] = [
            {"id": 1, "name": "Analyze Risk", "color": "#B12D89"},
            {"id": 2, "name": "Mitigate Risk", "color": "#006EAF"},
            {"id": 3, "name": "Detect Attacks", "color": "#C98E71"},
            {"id": 4, "name": "Collaborate", "color": "#E9A612"},
        ]
    return payload


def extract_from_pdf(pdf_path: Path, units: Optional[List[int]] = None) -> dict:
    doc = fitz.open(pdf_path)
    try:
        return build_document(doc, units_filter=units)
    finally:
        doc.close()
