"""Which department and how urgent: the routing labels, built only from what DDXPlus ships.

- URGENCY comes from DDXPlus's own per-condition `severity` (1 = most severe ... 5 = least). Not ours.
- DEPARTMENT comes from the condition's ICD-10 code. The chapter letter is the WHO's body-system grouping;
  the chapter -> department names below are a DEMO MAPPING for a general hospital and must be reviewed by a clinician
  before anyone relies on them. A handful of codes sit in chapters that don't name a ward (T = injury/poisoning,
  R = symptoms, S = injury); those are listed in EXCEPTIONS with the reason.

Nothing here diagnoses. The labels say where a case should be looked at, and how soon.
"""

# ICD-10 chapter letter -> department (demo mapping, clinician review needed)
CHAPTER_TO_DEPARTMENT = {
    "A": "Infectious diseases",   # certain infectious and parasitic diseases (A00-B99)
    "B": "Infectious diseases",
    "C": "Oncology",              # neoplasms (C00-D48)
    "D": "Haematology",           # D50-D89 blood; the only D codes here are D64.9 anaemia and D86 sarcoidosis (exception)
    "F": "Mental health",         # mental and behavioural (F00-F99)
    "G": "Neurology",             # nervous system (G00-G99)
    "H": "ENT",                   # H60-H95 ear (the only H code here is H66.90 otitis media)
    "I": "Cardiology",            # circulatory (I00-I99)
    "J": "Respiratory",           # respiratory (J00-J99)
    "K": "Gastroenterology",      # digestive (K00-K93)
    "M": "Rheumatology",          # musculoskeletal and connective tissue (M00-M99)
}

# Conditions whose chapter does not name the right place, with the reason. Keyed by DDXPlus English name.
EXCEPTIONS = {
    "Anaphylaxis": ("Emergency (resus)", "T78.0 sits in the injury/poisoning chapter; anaphylaxis is managed in resus"),
    "Scombroid food poisoning": ("Emergency (resus)", "T61.1 poisoning chapter; managed in the emergency department"),
    "Localized edema": ("General medicine", "R60.0 is a symptom code, not a body system"),
    "Spontaneous rib fracture": ("Trauma and orthopaedics", "S22.9 injury chapter"),
    "Sarcoidosis": ("Respiratory", "D86 sits with blood/immune disorders; usually followed by respiratory medicine"),
    "Acute laryngitis": ("ENT", "J04.0 is respiratory-chapter but seen by ENT"),
    "Acute rhinosinusitis": ("ENT", "J01 respiratory-chapter, seen by ENT"),
    "Chronic rhinosinusitis": ("ENT", "J32 respiratory-chapter, seen by ENT"),
    "Allergic sinusitis": ("ENT", "J30 respiratory-chapter, seen by ENT"),
    "Epiglottitis": ("ENT", "J05.1 airway emergency, ENT/anaesthetics"),
    "Larygospasm": ("ENT", "J38.5 (DDXPlus spelling kept) airway, ENT/anaesthetics"),
    "Croup": ("Paediatrics", "J05.0 almost always a child"),
    "Bronchiolitis": ("Paediatrics", "J21 almost always an infant"),
    "Acute otitis media": ("ENT", "H66.90 ear"),
}

# DDXPlus severity -> how soon (labels only; the number is DDXPlus's)
SEVERITY_TO_URGENCY = {
    1: "Immediate",
    2: "Very urgent",
    3: "Urgent",
    4: "Standard",
    5: "Non-urgent",
}


def department_of(name, icd10):
    if name in EXCEPTIONS:
        return EXCEPTIONS[name][0]
    letter = (icd10 or "?").strip()[:1].upper()
    return CHAPTER_TO_DEPARTMENT.get(letter, "General medicine")


def build(conditions):
    """conditions = parsed release_conditions.json -> {english name: {department, severity, urgency, icd10}}"""
    table = {}
    for key, c in conditions.items():
        name = c.get("cond-name-eng") or key
        table[key] = {
            "name": name,
            "icd10": c.get("icd10-id"),
            "department": department_of(name, c.get("icd10-id")),
            "severity": c.get("severity"),
            "urgency": SEVERITY_TO_URGENCY.get(c.get("severity")),
        }
    return table


if __name__ == "__main__":
    import json
    import sys
    table = build(json.load(open(sys.argv[1], encoding="utf-8")))
    json.dump(table, open(sys.argv[2], "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    by_dept = {}
    for row in table.values():
        by_dept.setdefault(row["department"], []).append(f'{row["severity"]} {row["name"]}')
    for dept in sorted(by_dept):
        print(f"{dept} ({len(by_dept[dept])}): " + "; ".join(sorted(by_dept[dept])))
