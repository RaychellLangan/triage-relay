"""DDXPlus patient rows -> short intake text + routing labels, one JSON line per patient.

usage: pyd.sh -I convert_ddxplus.py <patients.csv> <release_evidences.json> <routing_table.json> <out.jsonl> [limit]

TEXT (what the models read). The reason for the visit first, then symptoms, then history, in the dataset's own
English wording. DDXPlus lists mostly POSITIVE answers (an unlisted question was answered no or never asked);
a few categorical questions also appear at their default value (e.g. travel "N"), and those are kept as given.
The dataset's English labels include some poor translations (V_112 "lancinante" -> "haunting", V_71 "déchirante" ->
"heartbreaking"); training keeps them as they are, and any display fix belongs in the demo, not here.

    Age 49. Sex F.
    Reason for visit: Do you have a cough? yes
    Symptoms:
    - Characterize your pain: haunting, sensitive, tugging, burning
    - How intense is the pain? 6
    History:
    - Have you traveled out of the country in the last 4 weeks? N

LABELS (what the models are scored against). From the routing table (labels/routing_table.py):
    dept_true / urgency_true : the department and urgency of the recorded PATHOLOGY
    dept_mass                : the ground-truth differential spread over departments (sums to 1)
    severity_mass            : the differential spread over DDXPlus severity 1..5 (sums to 1)
The PATHOLOGY name is kept for analysis only; nothing downstream is trained to say it.
"""
import ast
import csv
import json
import sys
from collections import OrderedDict, defaultdict


def load_evidences(path):
    return json.load(open(path, encoding="utf-8"))


def answer_text(ev, value):
    """One answer value -> words. V_ codes use the dataset's English meaning; numbers stay numbers."""
    if isinstance(value, str) and value.startswith("V_"):
        return ev["value_meaning"].get(value, {}).get("en", value)
    return str(value)


def patient_text(row, evidences):
    groups = OrderedDict()                       # evidence name -> list of answers, in the row's order
    for item in ast.literal_eval(row["EVIDENCES"]):
        name, _, value = item.partition("_@_")
        groups.setdefault(name, [])
        if value:
            groups[name].append(value)

    def line(name):
        ev = evidences[name]
        q = ev["question_en"].strip()
        if ev["data_type"] == "B" or not groups[name]:
            return f"{q} yes"
        return f"{q} " + ", ".join(answer_text(ev, v) for v in groups[name])

    initial = row["INITIAL_EVIDENCE"]
    symptoms = [line(n) for n in groups if n != initial and not evidences[n]["is_antecedent"]]
    history = [line(n) for n in groups if n != initial and evidences[n]["is_antecedent"]]
    parts = [f"Age {row['AGE']}. Sex {row['SEX']}.",
             f"Reason for visit: {line(initial) if initial in groups else evidences[initial]['question_en'].strip() + ' yes'}"]
    if symptoms:
        parts.append("Symptoms:\n" + "\n".join("- " + s for s in symptoms))
    if history:
        parts.append("History:\n" + "\n".join("- " + h for h in history))
    return "\n".join(parts)


def labels(row, table):
    diff = ast.literal_eval(row["DIFFERENTIAL_DIAGNOSIS"])
    total = sum(p for _, p in diff)
    dept, sev = defaultdict(float), defaultdict(float)
    for name, p in diff:
        r = table[name]
        dept[r["department"]] += p / total
        sev[str(r["severity"])] += p / total
    truth = table[row["PATHOLOGY"]]
    return {
        "dept_true": truth["department"],
        "urgency_true": truth["urgency"],
        "severity_true": truth["severity"],
        "dept_mass": {k: round(v, 5) for k, v in sorted(dept.items(), key=lambda kv: -kv[1])},
        "severity_mass": {k: round(sev.get(k, 0.0), 5) for k in "12345"},
        "pathology": row["PATHOLOGY"],
    }


def main(patients_path, evidences_path, table_path, out_path, limit=None):
    evidences = load_evidences(evidences_path)
    table = json.load(open(table_path, encoding="utf-8"))
    limit = int(limit) if limit else None
    n = 0
    with open(patients_path, encoding="utf-8", newline="") as f, open(out_path, "w", encoding="utf-8") as out:
        for i, row in enumerate(csv.DictReader(f)):
            if limit and i >= limit:
                break
            rec = {"id": i, "text": patient_text(row, evidences)}
            rec.update(labels(row, table))
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} patients -> {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:6])
