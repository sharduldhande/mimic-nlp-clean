import duckdb
import pandas as pd
import re
import os
from pprint import pformat
import json
import numpy as np

df = duckdb.sql("""
    SELECT *
    FROM read_parquet('data/dataset_splits_*.parquet')
    WHERE discharge_note IS NOT NULL
    ORDER BY hash(note_id)
    LIMIT 75
""").df()

print(df.columns.tolist())
print(len(df))

os.makedirs("data", exist_ok=True)

with open("data/redundancy_sample_75.txt", "w", encoding="utf-8") as f:
    for i, row in df.reset_index(drop=True).iterrows():
        f.write(f"\n{'=' * 80}\n")
        f.write(f"RECORD {i + 1} | note_id: {row['note_id']}\n")
        f.write(f"{'=' * 80}\n")

        for col in df.columns:
            f.write(f"\n{col}:\n")
            f.write(f"{pformat(row[col], width=120)}\n")

print("Saved 75 records to data/redundancy_sample_75.txt")

"""
    
    Use regex or clinical NLP to extract entities and numeric values from the text.
[ ] Match each mention to the structured record using clinical concept, value, unit and time.
[ ] Classify mentions as redundant, complementary, conflicting or uncertain.
[ ] Record exact note spans and the corresponding structured feature for auditing.
"""

patterns = {
    # Demographics
    "age": r"\b(?P<value>\d{1,3})[\s-]*year[\s-]*old\b",
    "sex": r"\bSex:\s*(?P<value>Male|Female|M|F)\b",

    # Vitals
    "heart_rate": r"\bHR\s*[:=\- ]?\s*(?P<value>\d{2,3})\b",
    "blood_pressure": r"\bBP\s*[:=\- ]?\s*(?P<value>\d{2,3}/\d{2,3})\b",
    "respiratory_rate": r"\bRR\s*[:=\- ]?\s*(?P<value>\d{1,3})\b",
    "oxygen_saturation": r"\b(?P<value>\d{2,3})\s*%\s*oxygen saturation\b",

    # Diagnoses
    "hypertension": r"\b(?P<value>HTN|Hypertension)\b",
    "coronary_artery_disease": r"\b(?P<value>CAD|Coronary artery disease)\b",
    "chronic_kidney_disease": r"\b(?P<value>CKD|Chronic kidney disease)\b",
    "gi_bleed": r"\b(?P<value>GI bleed|GIB|gastrointestinal bleed)\b",
    "anemia": r"\b(?P<value>Anemia)\b",
    "dvt": r"\b(?P<value>DVTs?|deep vein thrombosis)\b",
}

lab_aliases = {
    "hemoglobin": ["HGB", "Hgb", "hemoglobin"],
    "hematocrit": ["HCT", "Hct", "calcHCT", "hematocrit"],
    "creatinine": ["CREAT", "Creat", "Cr", "creatinine"],
    "sodium": ["Na", "SODIUM"],
    "potassium": ["K", "POTASSIUM"],
    "glucose": ["GLUCOSE", "Glu"],
    "wbc": ["WBC"],
    "bun": ["BUN", "UREA N", "UreaN"],
}

for concept, aliases in lab_aliases.items():
    names = "|".join(re.escape(a).replace(r"\ ", r"\s+")
                     for a in aliases)

    patterns[concept] = (
        rf"\b(?:{names})\s*[:=\- ]\s*"
        rf"(?P<value>\d+(?:\.\d+)?)\b"
    )

mentions = []

for _, row in df.iterrows():
    text = row["discharge_note"]

    if not isinstance(text, str):
        continue

    for concept, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):

            mentions.append({
                "note_id": row["note_id"],
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "concept": concept,
                "value": match.group("value"),
                "note_span": match.group(0),
                "span_start": match.start(),
                "span_end": match.end(),
                "context": text[max(0, match.start()-60):
                                min(len(text), match.end()+60)].replace("\n", " ")
            })

mentions_df = pd.DataFrame(mentions)

mentions_df.to_csv(
    "data/extracted_mentions.csv",
    index=False
)

print(f"Extracted {len(mentions_df)} mentions")


import json
import numpy as np

# Structured feature names / aliases
feature_map = {
    "hemoglobin": ["hemoglobin", "hgb"],
    "hematocrit": ["hematocrit", "hct"],
    "creatinine": ["creatinine", "creat"],
    "sodium": ["sodium", "na"],
    "potassium": ["potassium", "k"],
    "glucose": ["glucose", "glu"],
    "wbc": ["wbc", "white blood cell count"],
    "bun": ["bun", "urea nitrogen"],
    "heart_rate": ["heart_rate", "heart rate", "hr"],
    "respiratory_rate": ["respiratory_rate", "rr"],
    "oxygen_saturation": ["spo2", "oxygen saturation"],
    "systolic_bp": ["sbp", "systolic_bp", "systolic blood pressure"],
    "diastolic_bp": ["dbp", "diastolic_bp", "diastolic blood pressure"],
}

diagnosis_map = {
    "hypertension": ["hypertension"],
    "coronary_artery_disease": [
        "coronary artery disease", "atherosclerotic heart disease"
    ],
    "chronic_kidney_disease": ["chronic kidney disease"],
    "gi_bleed": [
        "gastrointestinal bleed", "gastrointestinal hemorrhage"
    ],
    "anemia": ["anemia"],
    "dvt": ["deep vein thrombosis", "venous thrombosis"],
}

def get_records(value):
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    return value if isinstance(value, list) else []

def normalize(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())

def compare_numeric(note_value, measurement):
    try:
        n = float(note_value)
    except (ValueError, TypeError):
        return "not_comparable"

    for field in ["last_value", "min_value", "max_value", "mean_value"]:
        value = measurement.get(field)
        if value is None:
            continue
        try:
            v = float(value)
            if np.isfinite(v) and abs(n - v) <= max(0.05, abs(v) * 0.001):
                return f"matches_{field}"
        except (ValueError, TypeError):
            continue

    try:
        low = float(measurement["min_value"])
        high = float(measurement["max_value"])
        if low <= n <= high:
            return "within_recorded_range"
        return "outside_recorded_range"
    except (ValueError, TypeError, KeyError):
        return "not_comparable"


matches = []
records_by_note = {
    str(row["note_id"]): row
    for _, row in df.iterrows()
}

for _, mention in mentions_df.iterrows():
    record = records_by_note[str(mention["note_id"])]
    concept = mention["concept"]
    value = str(mention["value"])
    base = mention.to_dict()
    found = False

    # Demographics
    if concept in ["age", "sex"]:
        feature = "age" if concept == "age" else "gender"
        structured_value = record[feature]

        if concept == "sex":
            agrees = normalize(value) in {
                normalize(structured_value),
                "male" if normalize(structured_value) == "m" else "female"
            }
        else:
            agrees = value == str(structured_value)

        matches.append({
            **base,
            "structured_feature": feature,
            "structured_value": structured_value,
            "comparison": "equal" if agrees else "different",
        })
        found = True

    # Numerical measurements
    elif concept in feature_map or concept == "blood_pressure":

        targets = [(concept, value)]

        if concept == "blood_pressure":
            systolic, diastolic = value.split("/")
            targets = [
                ("systolic_bp", systolic),
                ("diastolic_bp", diastolic)
            ]

        for target, note_value in targets:
            aliases = {normalize(a) for a in feature_map[target]}

            for measurement in get_records(record["measurements"]):
                if not isinstance(measurement, dict):
                    continue
                if measurement.get("is_missing"):
                    continue
                if normalize(measurement.get("feature")) not in aliases:
                    continue

                matches.append({
                    **base,
                    "structured_feature": measurement["feature"],
                    "structured_value": measurement.get("last_value"),
                    "structured_min": measurement.get("min_value"),
                    "structured_max": measurement.get("max_value"),
                    "structured_mean": measurement.get("mean_value"),
                    "structured_unit": measurement.get("unit"),
                    "structured_time": measurement.get("last_time"),
                    "comparison": compare_numeric(note_value, measurement),
                })
                found = True

    # Diagnoses
    elif concept in diagnosis_map:
        for diagnosis in get_records(record["diagnoses"]):
            if not isinstance(diagnosis, dict):
                continue

            description = str(diagnosis.get("diagnosis", "")).lower()

            if any(term in description for term in diagnosis_map[concept]):
                matches.append({
                    **base,
                    "structured_feature": "diagnoses",
                    "structured_value": diagnosis.get("diagnosis"),
                    "icd_code": diagnosis.get("icd_code"),
                    "comparison": "diagnosis_candidate",
                })
                found = True

    if not found:
        matches.append({
            **base,
            "structured_feature": None,
            "structured_value": None,
            "comparison": "no_candidate_found",
        })

matches_df = pd.DataFrame(matches)
matches_df.to_csv("data/mention_matches.csv", index=False)

print(f"Saved {len(matches_df)} candidate matches")
print(matches_df["comparison"].value_counts())

features = set()

for measurements in df["measurements"]:
    for measurement in get_records(measurements):
        if isinstance(measurement, dict):
            features.add(measurement.get("feature"))

print(sorted(str(f) for f in features if f is not None))