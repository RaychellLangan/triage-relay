"""Ordinary English the router never trains on. The fine-tuned GPT-2 is scored on these passages before step 1 and at every
measurement; the rise in its loss is the English it has spent to learn routing (the "Under Pressure" canary).

Written for this project (no third-party text), deliberately unmedical and varied in register.
"""

PASSAGES = [
    "The bus was late again, so she walked the last mile along the canal. The water was flat and brown, and a heron stood "
    "on the far bank as if it had been placed there by someone who wanted the view to look finished. By the time she reached "
    "the bakery the queue had gone, and the man behind the counter was already sweeping the floor.",

    "To repot a houseplant, water it the day before so the roots hold together. Choose a pot only a little wider than the old "
    "one, add a layer of fresh compost, and lower the plant in so that the soil sits at the same height on the stem as before. "
    "Fill the gaps, press gently, and leave it somewhere bright but out of direct sun for a week.",

    "The committee met on Thursday to discuss the new parking arrangements. Several residents argued that the proposed permits "
    "would push cars into the side streets, while the shop owners wanted more short stays near the high street. After an hour "
    "the chair suggested a trial of three months, and the motion passed by a narrow margin.",

    "When the river froze that winter, the children of the village tested it every morning with long sticks, tapping further "
    "out each day. Their grandparents told them stories of the year the ice held a horse and cart, and of the year it did not. "
    "Nobody was quite sure which story was true, but everyone stayed close to the reeds.",

    "A good loaf needs very little: flour, water, salt and time. The dough should rest until it has nearly doubled, then be "
    "folded over itself a few times to give it strength. Bake it hot at first so the crust sets, then lower the heat and let "
    "the inside finish slowly. Leave it to cool before cutting, however tempting it smells.",

    "The train pulled out of the station just after six. Most of the passengers were asleep or pretending to be, and the only "
    "sound was the rattle of a loose window and a phone buzzing somewhere further down the carriage. Outside, the fields went "
    "from grey to green as the sun came up behind the hills.",
]
