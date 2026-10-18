"""Red flags the kiosk checks first: symptoms that always need a member of staff to look straight away.

Worded for patients from NHS public guidance (nhs.uk): stroke (the FAST test), heart attack, sepsis, severe bleeding.
A tick marks the patient Immediate, puts them top of every list, tells staff why, and sends them to the flag's own department
(stroke -> Neurology, heart attack -> Cardiology, sepsis or heavy bleeding -> Emergency (resus)), never a model's guess:
the training data has no stroke at all. Flags pointing to different departments go to Emergency (resus).
"""
RED_FLAGS = [
    {"id": "face",   "text": "One side of my face has drooped, or I can't smile properly", "from": "Stroke (FAST: face)", "route": "Neurology"},
    {"id": "arms",   "text": "I can't lift both arms and keep them up", "from": "Stroke (FAST: arms)", "route": "Neurology"},
    {"id": "speech", "text": "My speech is slurred or mixed up, or I'm confused", "from": "Stroke (FAST: speech) / sepsis", "route": "Neurology"},
    {"id": "chest",  "text": "Chest pain like pressure, tightness or squeezing, that may spread to my arms, neck, jaw or back", "from": "Heart attack", "route": "Cardiology"},
    {"id": "skin",   "text": "My skin, lips or tongue look blue, grey, pale or blotchy", "from": "Sepsis", "route": "Emergency (resus)"},
    {"id": "rash",   "text": "A rash that does not fade when I press a glass on it", "from": "Sepsis", "route": "Emergency (resus)"},
    {"id": "breath", "text": "I'm breathing very fast or can hardly breathe", "from": "Sepsis", "route": "Emergency (resus)"},
    {"id": "bleed",  "text": "I'm bleeding heavily and it won't stop", "from": "Severe bleeding", "route": "Emergency (resus)"},
]
BY_ID = {f["id"]: f for f in RED_FLAGS}
