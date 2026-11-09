"""Sample matched discharge notes and export a human-review redundancy audit."""
import argparse
import csv
import json
import random
import re
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path

# Extend these with the exact concept names used in your measurements structs.
ALIASES = {
    "creatinine": r"creatinine|creat|Cr",
    "hemoglobin": r"hemoglobin|hgb",
    "sodium": r"sodium|Na",
    "glucose": r"glucose|gluc",
    "heart_rate": r"heart[ _]rate|HR",
    "systolic_bp": r"systolic (?:blood pressure|BP)|SBP",
    "diastolic_bp": r"diastolic (?:blood pressure|BP)|DBP",
    "respiratory_rate": r"respiratory[ _-]rate|resp[ ._-]*rate|RR",
    "temperature": r"temperature|temp",
    "spo2": r"SpO[2₂]|oxygen saturation",
}
NUMBER = r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)"
UNIT = r"mg/dL|g/dL|g/L|(?:µ|μ|u)mol/L|mmol/L|mEq/L|mm\s?Hg|bpm|beats/min|breaths/min|resp/min|%|°[CF]|deg[CF]|/min"
TIME = r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}(?::\d{2})?"
MEASURE = re.compile(
    rf"(?:(?P<time>{TIME})[ \t]+)?\b(?P<name>{'|'.join(ALIASES.values())})"
    rf"\b(?:-(?=[\d.])|[ \t]*(?:[:=]|was)?[ \t]*)(?P<value>{NUMBER})"
    rf"(?:(?![\w./:-])|(?=/[ \t]*(?:{'|'.join(ALIASES.values())})\b))"
    rf"(?:[ \t]*(?P<unit>{UNIT})(?![\w/]))?",
    re.I,
)
PANEL = re.compile(
    r"^[ \t]*(?P<date>\d{4}-\d{2}-\d{2}|_{3,})[ \t]+"
    r"(?P<clock>\d{1,2}:\d{2}(?::\d{2})?[ \t]*(?:AM|PM)?)"
    r"[ \t]+BLOOD\b", re.I,
)
PANEL_ANCHOR = re.compile(
    rf"\b(?P<name>WBC|RBC|Hct)\b(?:-(?=[\d.])|[ \t]*[:=]?[ \t]*)"
    rf"(?P<value>{NUMBER})(?![\w./:-])", re.I,
)
BP = re.compile(
    rf"(?:(?P<time>{TIME})[ \t]+)?\b(?:BP|blood pressure)\b[ \t]*[:=]?[ \t]*"
    rf"(?P<sbp>\d{{2,3}})/(?P<dbp>\d{{2,3}})\b(?:[ \t]*(?P<unit>mm\s?Hg))?",
    re.I,
)
AGE = re.compile(r"\b(?P<value>\d{1,3})(?:[- ]year[- ]old|\s+y/o\b)", re.I)
SEX = re.compile(r"\b(?:sex|gender)[ \t]*:[ \t]*(?P<value>female|male|[MF])\b"
                 r"|\b\d{1,3}[- ]year[- ]old[ \t]+(?P<adjacent>female|male|woman|man)\b", re.I)
ADMIN = re.compile(
    r"^[ \t]*(?P<name>admission type|admission location|insurance)[ \t]*:"
    r"[ \t]*(?P<value>[^\n\r]+)", re.I | re.M,
)
IDS = ["subject_id", "hadm_id", "stay_id", "note_id", "split"]
COLUMNS = IDS + ["discharge_note", "age", "gender", "admission_type",
                 "admission_location", "insurance", "admittime", "dischtime",
                 "diagnoses", "measurements"]
FIELDS = IDS + ["concept", "start", "end", "span", "context", "note_value",
                "note_unit", "note_time", "structured_candidates", "suggested_label",
                "reason", "note_time_source", "value_vs_range",
                "event_linkage", "event_id", "specimen_id", "linked_event_time",
                "linked_event_unit", "linked_feature", "link_evidence", "review_label", "review_feature", "review_comment",
                "context_start", "context_end", "context_marked"]


def first(record, *keys):
    return next((record[k] for k in keys if record.get(k) is not None), None)


def concept_for(name):
    name = str(name).strip()
    for concept, aliases in ALIASES.items():
        if name.casefold() == concept or re.fullmatch(aliases, name, re.I):
            return concept
    return None


def nested(value, field):
    if isinstance(value, str):
        value = json.loads(value)
    if value is not None and not isinstance(value, (list, dict)):
        raise ValueError(f"Expected a list or object for {field}, got {type(value).__name__}")
    return value


def measurements(value):
    """Adapt either a list of structs or a mapping from concept to struct."""
    value = nested(value, "measurements") or []
    items = value.items() if isinstance(value, dict) else enumerate(value)
    result = {}
    for key, item in items:
        if item is None:
            continue
        if not isinstance(item, dict):
            raise ValueError("Edit measurements() to match your measurement schema.")
        name = first(item, "name", "measurement", "label", "concept", "feature")
        concept = concept_for(name if name is not None else key)
        if concept:
            path = f"measurements[{key!r}]"
            for statistic in ("last", "mean", "min", "max"):
                aliases = {"min": "minimum", "max": "maximum"}
                keys = (statistic, f"{statistic}_value", aliases.get(statistic, statistic))
                value_key = next((k for k in keys if item.get(k) is not None), keys[0])
                result.setdefault(concept, []).append({
                    "feature": f"{path}.{value_key}",
                    "statistic": statistic,
                    "value": item.get(value_key),
                    "unit": first(item, "unit", "valueuom"),
                    "time": first(item, "last_timestamp", "last_charttime", "last_time", "timestamp", "charttime")
                    if statistic == "last" else None,
                    "missing": first(item, "missing", "is_missing", "missingness"),
                })
    return result


def diagnoses(value):
    """Use the recorded descriptions and codes as a literal seed dictionary."""
    value = nested(value, "diagnoses") or []
    if isinstance(value, dict):
        value = [value] if any(k in value for k in ("diagnosis", "description", "long_title", "icd_code", "code")) else [
            {"icd_code": code, "description": description} for code, description in value.items()
        ]
    for index, item in enumerate(value):
        if isinstance(item, str):
            yield f"diagnoses[{index}]", item
        elif isinstance(item, dict):
            for key in ("diagnosis", "description", "long_title", "icd_code", "code"):
                if isinstance(item.get(key), str):
                    yield f"diagnoses[{index}].{key}", item[key]


def decimal(value):
    try:
        number = Decimal(str(value))
        return number if number.is_finite() else None
    except InvalidOperation:
        return None


def time_value(value):
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def unit_key(value):
    if not value:
        return None
    value = str(value).lower().replace(" ", "")
    return {"beats/min": "bpm", "resp/min": "breaths/min", "degc": "°c", "degf": "°f", "μmol/l": "umol/l", "µmol/l": "umol/l"}.get(value, value)


# Allow only explicit, concept-specific units. The EHR unit never fills a blank note unit.
PINT_UNITS = {
    "g/dl": "gram / deciliter",
    "g/l": "gram / liter",
    "mmol/l": "millimole / liter",
    "umol/l": "micromole / liter",
    "°c": "degC",
    "°f": "degF",
}
CONVERTIBLE_UNITS = {
    "hemoglobin": {"g/dl", "g/l"},
    "temperature": {"°c", "°f"},
    "creatinine": {"mmol/l", "umol/l"},
}


@lru_cache(maxsize=1)
def unit_registry():
    try:
        from pint import UnitRegistry
    except ImportError:
        return None
    return UnitRegistry(non_int_type=Decimal)


def comparable_value(concept, value, note_unit, ehr_unit):
    source, target = unit_key(note_unit), unit_key(ehr_unit)
    number = decimal(value)
    if number is None or source is None or target is None:
        return None
    if source == target:
        return number
    if {source, target} <= CONVERTIBLE_UNITS.get(concept, set()):
        registry = unit_registry()
        if registry is not None:
            return registry.Quantity(number, PINT_UNITS[source]).to(PINT_UNITS[target]).magnitude
    return None


def range_status(concept, value, unit, candidates):
    bounds = {}
    for statistic in ("min", "max"):
        options = [c for c in candidates if c["statistic"] == statistic
                   and c["missing"] in (None, False, 0) and decimal(c["value"]) is not None]
        if len(options) != 1:
            return "unknown"
        bounds[statistic] = options[0]
    low, high = bounds["min"], bounds["max"]
    if unit_key(low["unit"]) != unit_key(high["unit"]):
        return "unknown"
    converted = comparable_value(concept, value, unit, low["unit"])
    if converted is None or decimal(low["value"]) > decimal(high["value"]):
        return "unknown"
    if converted < decimal(low["value"]):
        return "below"
    if converted > decimal(high["value"]):
        return "above"
    return "within"


def measurement_time(note, match):
    if match["time"]:
        return match["time"], "inline"
    line_start = note.rfind("\n", 0, match.start()) + 1
    panel = PANEL.match(note[line_start:match.start()])
    if panel is None:
        return None, None
    clock = panel["clock"].strip().upper()
    meridiem = clock[-2:] if clock.endswith(("AM", "PM")) else ""
    clock = clock[:-2].strip() if meridiem else clock
    fmt = "%H:%M:%S" if clock.count(":") == 2 else "%H:%M"
    if meridiem:
        fmt = fmt.replace("%H", "%I") + "%p"
    try:
        parsed = datetime.strptime(clock + meridiem, fmt)
    except ValueError:
        return None, None
    precision = "seconds" if clock.count(":") == 2 else "minutes"
    clock = parsed.time().isoformat(timespec=precision)
    if panel["date"].startswith("_"):
        return clock, "panel_time_only"
    return f'{panel["date"]}T{clock}', "panel_datetime"


def load_events(path, sample):
    """Read normalized source events, including events outside the model window."""
    admissions = {(str(row["subject_id"]), str(row["hadm_id"])) for row in sample}
    events = defaultdict(list)
    required = {"subject_id", "hadm_id", "source", "event_id", "specimen_id", "stay_id",
                "concept", "charttime", "value", "unit"}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if missing := required.difference(reader.fieldnames or []):
            raise ValueError(f"Event CSV missing columns: {sorted(missing)}")
        seen = set()
        for event in reader:
            key = event["subject_id"], event["hadm_id"]
            if key not in admissions:
                continue
            event["concept"] = concept_for(event["concept"]) or event["concept"].strip().lower()
            event_key = event["source"], event["event_id"]
            if (event["source"] not in {"lab", "chart"} or not event["event_id"]
                    or event_key in seen or time_value(event["charttime"]) is None):
                raise ValueError(f"Invalid or duplicate event at CSV line {reader.line_num}")
            seen.add(event_key)
            events[key].append(event)
    return events


def event_agrees(event, value, unit):
    number = decimal(value)
    recorded = decimal(event["value"])
    if number is None or recorded is None or not unit_key(event["unit"]):
        return False
    # An omitted note unit can be linked through corroborating panel values.
    if not unit:
        return number == recorded
    return comparable_value(event["concept"], value, unit, event["unit"]) == recorded


def same_display_time(event_time, note_time):
    timestamp = time_value(event_time)
    if timestamp is None or not note_time:
        return False
    clock = note_time.split("T")[-1].split(" ")[-1]
    precision = "minutes" if clock.count(":") == 1 else "seconds"
    date = time_value(note_time)
    if date is not None and timestamp.date() != date.date():
        return False
    return timestamp.time().isoformat(timespec=precision) == clock


def panel_links(line, concept, value, unit, time, events):
    anchors = defaultdict(set)
    for pattern in (MEASURE, PANEL_ANCHOR):
        for match in pattern.finditer(line):
            name = concept_for(match["name"]) or match["name"].lower()
            if name != concept:
                anchors[name].add((match["value"], match.groupdict().get("unit")))
    specimens = defaultdict(list)
    for event in events:
        if (event["source"] == "lab" and event["specimen_id"]
                and same_display_time(event["charttime"], time)):
            specimens[event["specimen_id"]].append(event)
    links = []
    for specimen in specimens.values():
        supporting = sorted(
            name for name, readings in anchors.items()
            if len(readings) == 1 and sum(
                event["concept"] == name and event_agrees(event, *next(iter(readings)))
                for event in specimen
            ) == 1
        )
        if len(supporting) >= 2:
            links.extend((event, supporting) for event in specimen
                         if event["concept"] == concept and event_agrees(event, value, unit))
    return links


def link_measurement(note, match, row, concept, value, unit, time, source, candidates, events):
    if source in {"panel_datetime", "panel_time_only"}:
        start = note.rfind("\n", 0, match.start()) + 1
        end = note.find("\n", match.end())
        line = note[start:] if end == -1 else note[start:end]
        links = panel_links(line, concept, value, unit, time, events)
    elif time_value(time) is not None and unit:
        is_lab = concept in {"creatinine", "hemoglobin", "sodium", "glucose"}
        links = [
            (event, []) for event in events
            if event["concept"] == concept and same_display_time(event["charttime"], time)
            and event_agrees(event, value, unit)
            and (event["source"] == "lab" if is_lab else
                 event["source"] == "chart" and row.get("stay_id") is not None
                 and event["stay_id"] == str(row["stay_id"]))
        ]
    else:
        links = []

    result = {"event_linkage": "unavailable", "suggested_label": "uncertain"}
    if len(links) != 1:
        if links:
            result["event_linkage"] = "ambiguous"
        result["reason"] = ("Multiple raw events match." if links else
                            "No qualifying raw-event link." if events else
                            "No raw events available for this admission.")
        return result

    event, supporting = links[0]
    result.update(
        event_linkage="unique", event_id=event["event_id"], specimen_id=event["specimen_id"],
        linked_event_time=event["charttime"], linked_event_unit=event["unit"],
        link_evidence=", ".join(supporting) if supporting else "Explicit timestamp and unit",
        reason="Raw event linked. Provenance for the model feature is unverified.",
    )
    feature = event.get("feature_path")
    matching_features = [
        c for c in candidates if feature and c["feature"] == feature
        and c["statistic"] == "last" and c["missing"] in (None, False, 0)
        and comparable_value(concept, event["value"], event["unit"], c["unit"]) is not None
        and comparable_value(concept, event["value"], event["unit"], c["unit"]) == decimal(c["value"])
        and (not c["time"] or time_value(c["time"]) == time_value(event["charttime"]))
    ]
    if feature and event.get("in_feature_window") == "0":
        result["reason"] = "Source provenance conflicts with the feature-window flag."
    elif len(matching_features) == 1:
        result.update(suggested_label="redundant", linked_feature=feature,
                      reason="Unique raw event supplies the verified structured last feature.")
    elif not feature and event.get("in_feature_window") == "0":
        result.update(suggested_label="complementary",
                      reason="Unique raw event is outside the verified feature window.")
    return result


def review_context(note, start, end):
    """Show complete matching lines plus one neighboring line on each side."""
    line_start = note.rfind("\n", 0, start) + 1
    context_start = note.rfind("\n", 0, max(0, line_start - 1)) + 1
    line_end = note.find("\n", end)
    context_end = note.find("\n", line_end + 1) if line_end != -1 else -1
    if context_end == -1:
        context_end = len(note)
    return {
        "context": note[context_start:context_end],
        "context_start": context_start, "context_end": context_end,
        "context_marked": (note[context_start:start] + "[[" + note[start:end]
                           + "]]" + note[end:context_end]),
    }


def audit_row(row, events=()):
    note = row["discharge_note"]
    indexed = measurements(row.get("measurements"))

    def entry(match, concept, value, candidates, label, reason, unit=None, time=None):
        start, end = match.span()
        return {
            **{key: row.get(key) for key in IDS},
            "concept": concept, "start": start, "end": end, "span": note[start:end],
            **review_context(note, start, end),
            "note_value": value, "note_unit": unit, "note_time": time,
            "structured_candidates": json.dumps(candidates, default=str, ensure_ascii=False),
            "suggested_label": label, "reason": reason,
        }

    for pattern in (MEASURE, BP):
        for match in pattern.finditer(note):
            if pattern is MEASURE and re.match(r"[ \t]*(?:[–—-][ \t]*\d|to[ \t]+\d)",
                                                 note[match.end():], re.I):
                continue
            if pattern is BP:
                pairs = [("systolic_bp", match["sbp"]), ("diastolic_bp", match["dbp"])]
            else:
                pairs = [(concept_for(match["name"]), match["value"])]
            for concept, value in pairs:
                candidates = indexed.get(concept, [])
                time, source = measurement_time(note, match)
                unit = match["unit"]
                yield {
                    **entry(match, concept, value, candidates, "uncertain", "", unit, time),
                    **link_measurement(note, match, row, concept, value, unit, time,
                                       source, candidates, events),
                    "note_time_source": source,
                    "value_vs_range": range_status(concept, value, unit, candidates),
                }

    for pattern in (AGE, SEX, ADMIN):
        for match in pattern.finditer(note):
            if pattern is AGE:
                field, value = "age", match["value"]
            elif pattern is SEX:
                field, value = "gender", match["value"] or match["adjacent"]
            else:
                field, value = match["name"].lower().replace(" ", "_"), match["value"].strip()
            expected = row.get(field)
            sex_map = {"m": "male", "f": "female", "man": "male", "woman": "female"}
            actual_text, expected_text = value.casefold(), str(expected).strip().casefold()
            if field == "age":
                agrees = decimal(value) == decimal(expected)
            elif field == "gender":
                agrees = sex_map.get(actual_text, actual_text) == sex_map.get(expected_text, expected_text)
            else:
                agrees = actual_text == expected_text
            # Age has an admission-time reference and can be approximate.
            label = "redundant" if agrees and field != "age" else "uncertain"
            reason = ("Value agrees. Verify this refers to the patient and encounter."
                      if agrees else "Check definition, timing, and patient reference.")
            yield entry(match, field, value, [{"feature": field, "value": expected}], label, reason)

    by_term = {}
    for feature, term in diagnoses(row.get("diagnoses")):
        term = term.strip()
        if len(term) >= 3:
            by_term.setdefault(term.casefold(), []).append({"feature": feature, "value": term})
    for term, candidates in by_term.items():
        for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", note, re.I):
            yield entry(match, "diagnosis", match[0], candidates, "uncertain",
                        "Literal diagnosis hit. Verify assertion, experiencer, and encounter.")


def sample_rows(files, count, seed, split):
    """Reservoir sample one eligible note per admission across all input files."""
    rng = random.Random(seed)
    sample, seen = [], set()
    for path in files:
        columns = [key for key in COLUMNS if key in path.schema_arrow.names]
        required = {"subject_id", "hadm_id", "note_id", "discharge_note", "measurements", "diagnoses", "split"}
        if missing := required.difference(columns):
            raise ValueError(f"Missing columns: {sorted(missing)}")
        for batch in path.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                if split != "all" and str(row["split"]).casefold() != split.casefold():
                    continue
                note = row["discharge_note"]
                admission = row["hadm_id"]
                if admission is None or row["note_id"] is None or not isinstance(note, str) or not note.strip():
                    continue
                if admission in seen:
                    continue
                seen.add(admission)
                if len(sample) < count:
                    sample.append(row)
                else:
                    index = rng.randrange(len(seen))
                    if index < count:
                        sample[index] = row
    return sample


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parquet", nargs="+", type=Path)
    parser.add_argument("--schema", action="store_true", help="Show nested schemas without printing row data")
    parser.add_argument("--count", type=int, choices=range(50, 101), default=75, metavar="50–100")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", default="train", help="Exact split label, or all")
    parser.add_argument("--out", type=Path, default=Path("redundancy_audit"), help="New output directory")
    parser.add_argument("--events", type=Path, help="Normalized source-event CSV with optional feature provenance")
    args = parser.parse_args()

    import pyarrow.parquet as pq

    files = [pq.ParquetFile(path) for path in args.parquet]
    if args.schema:
        for path, parquet in zip(args.parquet, files):
            print(path, parquet.schema_arrow, sep="\n")
        return

    sample = sample_rows(files, args.count, args.seed, args.split)
    if len(sample) < args.count:
        parser.error(f"Only {len(sample)} eligible admissions for split {args.split!r}. Check the split label.")
    events = load_events(args.events, sample) if args.events else {}
    mentions = [
        mention for row in sample
        for mention in audit_row(row, events.get((str(row.get("subject_id")), str(row["hadm_id"])), ()))
    ]
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "mentions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(mentions)
    with (args.out / "sample.jsonl").open("w", encoding="utf-8") as stream:
        for row in sample:
            stream.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    print(f"Exported {len(sample)} notes and {len(mentions)} candidate mentions to {args.out}")


if __name__ == "__main__":
    main()
