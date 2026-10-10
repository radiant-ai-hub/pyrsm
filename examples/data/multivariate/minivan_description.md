## Minivan choice study (choice-based conjoint)

### Description

A choice-based conjoint (CBC) study for a new minivan. 200 respondents each answered 15 choice questions; every question showed three minivans and the respondent chose the one they would buy. The data are in long format: one row per minivan shown in each question.

The study design follows the minivan example in Chapman and Feit, *R for Marketing Research and Analytics* (Springer), chapter 13. The data are simulated from known preferences (`scripts/generate_minivan_example.py` in pyrsm): every respondent has their own part-worths around the averages below, carpoolers prefer extra seats (+1.5 for 7 seats, +2.0 for 8 seats), and choices follow a logit model.

### Variables

- resp = Respondent id
- ques = Question number for the respondent (1-15)
- task = Choice task id: one value per question per respondent (1-3,000)
- alt = Position of the minivan in the question (1-3)
- carpool = Does the respondent carpool? (yes, no)
- seat = Number of seats (6, 7, 8)
- cargo = Cargo space (2ft, 3ft)
- eng = Engine (gas, hyb = hybrid, elec = electric)
- price = Price in $1,000 (30, 35, 40)
- choice = 1 if the respondent chose this minivan, else 0

Average part-worths used to simulate the choices: seat (6: 0, 7: -1, 8: -1), cargo (2ft: 0, 3ft: 0.5), engine (gas: 0, hybrid: -1, electric: -2), price -0.2 per $1,000.
