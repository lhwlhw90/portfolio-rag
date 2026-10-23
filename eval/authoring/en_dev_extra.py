"""M5: extra *dev-only* questions for the English eval set (written from scratch for M5).

They extend the dev split used to choose and calibrate the answerability judge (M4's dev had
only 13 unanswerable questions). They are never mixed into test: scripts/build_dev_extra.py
writes them to eval/datasets/en.dev_extra.jsonl after checking that no question duplicates or
closely paraphrases a test (or existing dev) question, and that the test files are unchanged.

Every question below was checked by hand against sample_docs/en: the documents do NOT state the
asked-for fact. Most are near misses (a known entity + an attribute the documents don't give),
a few are off-domain.
Format as in en_questions.py: (type, question, evidence, answer_contains).
"""
Q = [
    # --- employee handbook: entity known, attribute missing
    ("unanswerable", "Is sabbatical leave available after several years at Acme?", [], []),
    ("unanswerable", "What is the notice period when an employee resigns?", [], []),
    ("unanswerable", "Can I work remotely from another country for a month?", [], []),
    ("unanswerable", "How are overtime hours compensated?", [], []),
    ("unanswerable", "Who is the current VP of People?", [], []),
    ("unanswerable", "What is the mileage allowance for using a private car on company business?", [], []),
    ("unanswerable", "How many days of bereavement leave are granted?", [], []),
    ("unanswerable", "Which laptop brand and model do new hires receive?", [], []),
    ("unanswerable", "On which day of the month are salaries paid?", [], []),
    ("unanswerable", "Up to what value may employees give presents to customers?", [], []),
    # --- benefits
    ("unanswerable", "Does the group health insurance also cover spouses?", [], []),
    ("unanswerable", "Which clinic carries out the yearly check-up?", [], []),
    ("unanswerable", "Is life insurance part of the benefits package?", [], []),
    ("unanswerable", "How large is the wedding gift Acme gives to employees who marry?", [], []),
    # --- office
    ("unanswerable", "Is the headquarters building open on public holidays?", [], []),
    ("unanswerable", "What does a cup of coffee cost in the canteen?", [], []),
    ("unanswerable", "On which floor is reception located?", [], []),
    ("unanswerable", "Are there showers for staff who cycle to work?", [], []),
    ("unanswerable", "How long do people usually wait for a garage spot?", [], []),
    ("unanswerable", "When exactly is the next evacuation exercise scheduled?", [], []),
    # --- IT
    ("unanswerable", "Which vendor makes the VPN client we use?", [], []),
    ("unanswerable", "What is the phone number of the front desk?", [], []),
    ("unanswerable", "Which authenticator app is mandatory, Microsoft or Google?", [], []),
    ("unanswerable", "Is there a monthly page quota for printing?", [], []),
    ("unanswerable", "Does Acme reimburse home internet costs?", [], []),
    # --- security
    ("unanswerable", "How often must staff complete security awareness training?", [], []),
    ("unanswerable", "Which cloud storage services are on the approved list?", [], []),
    ("unanswerable", "How long are server logs retained?", [], []),
    ("unanswerable", "Who is Acme's data protection officer?", [], []),
    # --- RoboArm X2 product / manual / release notes / support
    ("unanswerable", "What is the number of joints on this robot arm?", [], []),
    ("unanswerable", "Does the X2 support EtherCAT?", [], []),
    ("unanswerable", "What is the capacity in watt-hours of the X2 battery?", [], []),
    ("unanswerable", "Which ISO flange pattern does the X2 tool mount follow?", [], []),
    ("unanswerable", "How fast can the base joint rotate in degrees per second?", [], []),
    ("unanswerable", "Can the X2 be leased instead of bought?", [], []),
    ("unanswerable", "What colour is the arm's housing?", [], []),
    ("unanswerable", "How many weeks does delivery of a newly ordered X2 take?", [], []),
    ("unanswerable", "What changed in firmware 1.9?", [], []),
    ("unanswerable", "Which bugs did release 2.1 fix?", [], []),
    ("unanswerable", "What is the turnaround for fixing a robot whose guarantee has lapsed?", [], []),
    ("unanswerable", "What does Standard support cost once the warranty has expired?", [], []),
    ("unanswerable", "Is on-site training at the customer's plant available?", [], []),
    ("unanswerable", "Does the open-source driver work with ROS 1 Noetic?", [], []),
    ("unanswerable", "Does the controller box have a fan filter that needs cleaning?", [], []),
    # --- off-domain
    ("unanswerable", "What will the weather be like in Berlin tomorrow?", [], []),
    ("unanswerable", "Who won the 2022 football World Cup?", [], []),
    ("unanswerable", "How do I bake sourdough bread?", [], []),
]
