"""Attribution v3 (docs/DESIGN.md): every quotation gets one of three voices, male, female or
narrator. No roster, no relation graph: speech tags decide by rule, a name -> gender table gives the
gender of a name, and the model is asked only about quotes without a tag.

    tags.py     speech tags and "is this dialogue" (rules)
    genders.py  the name -> gender table (rules, one model call per batch of leftover names, genders.yaml)
    decide.py   combine rule, model and scene evidence into a voice and a confidence
"""

MALE, FEMALE, NARRATOR = "_generic_male", "_generic_female", "narrator"
VOICE_OF = {"male": MALE, "female": FEMALE}
GENDER_OF = {MALE: "male", FEMALE: "female"}
