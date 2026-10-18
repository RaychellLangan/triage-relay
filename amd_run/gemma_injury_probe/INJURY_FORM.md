# Injury form: the fixed answers a wider reader (Gemma) may fill, for patients the tiny translator cannot read
NEXICOG's idea (14 Oct): Gemma knows the wider world (horses, kicks, ovens), so it could translate an unseen case into a FIXED form.
The rules on the form decide, never the model. Same principle as the rest of Triage Relay: translate, don't decide.

**The form holds only what a patient can tell us in words.** Anything that needs an examination is listed separately for the nurse, never filled by a model.

## Fields (each answer is chosen from a closed list; "unknown" is always allowed and is the honest default)
| field | answers | why it is on the form (source checked 14 Oct) |
|---|---|---|
| injury | yes / no | the kiosk's injury door |
| how | fall · blow or kick · cut · burn or scald · bite or sting · crush · road traffic · other · unknown | mechanism, for the nurse; "road traffic" and "fall" feed the dangerous-mechanism item below |
| where | head · face · neck · chest · belly · back · arm · hand or wrist · hip or leg · knee · ankle or foot · other · unknown | body area (the kiosk's tiles say the same thing) |
| knocked_out_or_memory_gap | yes / no / unknown | NICE NG232 rec 1.5.9: loss of consciousness or amnesia since the injury opens the 8-hour CT criteria (with a risk factor) |
| vomited_more_than_once | yes / no / unknown | NICE NG232 rec 1.5.8 (people 16+): "more than 1 episode of vomiting" is a 1-hour CT criterion |
| seizure_since | yes / no / unknown | NICE NG232 rec 1.5.8: post-traumatic seizure, a 1-hour CT criterion |
| dangerous_mechanism | yes / no / unknown | NICE NG232 rec 1.5.9 definition: a pedestrian or cyclist struck by a motor vehicle, an occupant ejected from a motor vehicle, or a fall from more than 1 m or 5 stairs |
| blood_thinners | yes / no / unknown | NICE NG232 rec 1.5.13: anticoagulants (incl. warfarin-type, DOACs, heparins) or antiplatelets (not aspirin alone) → consider CT within 8 hours. Rec 1.5.9 also lists current bleeding or clotting disorders as an 8-hour risk factor |
| can_walk_four_steps | yes / no / unknown | Ottawa ankle/foot and knee rules: unable to bear weight both immediately and in the ED (4 steps; a limp still counts as able) |
| heavy_bleeding | yes / no / unknown | the kiosk's NHS red flag: heavy bleeding that won't stop |

**Age 65 or over** (a NICE 1.5.9 risk factor) comes from the kiosk's age field, never from the words.

## Never filled by a model: for the nurse to examine
- Glasgow Coma Scale (rec 1.5.8: 12 or less on first assessment; less than 15 at 2 hours).
- Suspected open or depressed skull fracture; signs of a basal skull fracture; focal neurological deficit (all rec 1.5.8).
- Ottawa bone-tenderness points: the posterior edge or tip of either malleolus; the navicular; the base of the 5th metatarsal; isolated patellar or fibular-head tenderness; knee flexion to 90°.

## What the rules would do with a filled form (design note; nothing here is built into the app yet)
- **Any 1.5.8 item** (vomited more than once, seizure): the card says "NICE head-injury criteria for urgent imaging may apply: clinician to assess now".
- **Knocked out or memory gap, plus a 1.5.9 risk factor** (65+, a bleeding or clotting disorder, a dangerous mechanism): "NICE 8-hour imaging criteria may apply".
- **can_walk_four_steps = no** with a leg, knee or ankle injury: "Ottawa rule: weight-bearing criterion met; clinician to examine".
- **Heavy bleeding:** the red-flag path (Immediate).
- Every message is a prompt for a clinician, never an order, and never a diagnosis.

## Sources
- NICE NG232 *Head injury: assessment and early management* (2023), recommendations 1.5.8, 1.5.9 and 1.5.13. Read on nice.org.uk on 14 Oct 2026, paraphrased here.
- Ottawa ankle and knee rules (Stiell et al.), as summarised by MDCalc / RACGP / Physitrack. Check against the original publications before any clinical use.
