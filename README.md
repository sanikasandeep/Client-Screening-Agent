# Client Screening Agent

An AI assistant that helps compliance teams review sanctions-screening alerts, focusing first on telling genuine matches apart from false alarms when two people happen to share a name.

This README explains, in plain language, what each part of this repository does: the two files that build test data, the agent that makes the decisions, and what happened when the agent was first tested.

## The problem this solves

When a bank, fintech, or similar regulated company takes on a new customer, it is legally required to check that person against sanctions lists: government-published lists of individuals and organisations that companies are forbidden from doing business with. The check is done by computer, and because names can be spelled, transliterated, or recorded in many ways, the computer flags anyone whose details look close to someone on a list.

However, most of these flags are false alarms. Thousands of ordinary people share a name with someone on a sanctions list but are not that person. Annoyingly, a human compliance officer has to open each flagged case and investigate it by hand, comparing dates of birth, nationalities, and other details to decide whether it is a real match or a coincidence. This is slow, expensive, and repetitive, and the vast majority of the work ends in "not the same person."

The agent in this project reads each flagged case, weighs the identifying details, and judges whether it is a genuine match, a false alarm, or something that still needs a human, along with a written rationale as to why it came to the conclusion it did. This helps provide AI transparency, allowing regulators to verify AI-enabled decisions more reliably. The aim is to clear the obvious false alarms so officers spend their time only on the cases that genuinely need a person.

## Why this repository builds test data

Before anyone can trust an agent like this, it has to be proven accurate, and one mistake matters far more than the others: it must never wave through a person who really is on a sanctions list. To measure accuracy you need a large set of example cases where the correct answer is already known, so you can run the agent over them and check how often it gets each one right.

There is no public collection of real, already-decided screening cases, because those contain real customers' personal data and are confidential. So this repository creates a realistic test set instead. That is what the two generator files do: they manufacture example cases, with known correct answers, that look like the cases the real system produces.

- **`sanctions_fp_generator.py`** builds *false alarms* (false positives): people who share a name with someone on the UK sanctions list but are demonstrably a different person, for example because their date of birth or nationality clearly differs. Because the file constructs each case deliberately, it knows for certain that the answer is "not the same person."

- **`true_positive_generator.py`** builds *genuine matches* (true positives): customers who really are the listed person, but recorded the way a real sign-up form would capture them, with a name spelled a little differently, or one detail missing. The match is built in, so the known answer is "yes, the same person."

A good test needs both. An agent that simply cleared every case would look perfect on a test made only of false alarms, so the genuine matches are what prove it can still catch the people it must never miss.

## How the data is formatted, and why

Each generated case is written to look like the data the real screening system produces, so that an agent which does well on the test will also do well in production rather than on a made-up format. A few deliberate choices shape that data.

**Every case has two sides.** One side is the *customer*, the person being checked, shaped like the record a verification session produces. The other side is the *candidate*, the sanctions-list entry the customer was flagged against, shaped like the record the screening database returns. The agent's whole job is to compare these two sides and decide if they are the same person, so the test gives it exactly those two sides and nothing it would not have in reality.

**The cases carry the details an officer actually uses.** Beyond the name, each side includes date of birth, nationality, place of birth, and an identity-document number where available, because those are the details that settle whether two same-named people are actually one person. The list entry also records what kind of target it is (for example a sanctioned individual versus a politically exposed person) since that changes both the decision and how serious it is.

**Dates of birth are kept in full.** An earlier version used only the birth year, but day and month often decide a case, so the full date is preserved. This lets the test include realistic situations such as a customer's full date of birth being checked against a list entry that only records a year.

**The customers are entirely invented.** The sanctions-list side uses real, public UK designations, but every "customer" is fabricated, so the data files contain no real person's private information and are safe to share and to send to an AI service for testing.

**The file is stored as JSONL.** JSON is a standard structured text format. JSONL simply means one JSON record per line, so the file is a list of cases, one per line. This makes it easy to process the cases one at a time and to add more later without reshaping the file.

**A realism filter decides what counts.** A false alarm is only useful in the test if the screening system would actually have raised it. So each constructed case is scored on how similar the name and date of birth are, and only cases that would genuinely cross a screening system's alert threshold are kept. This stops the test from being padded with cases so far-fetched that no real system would ever surface them.

Finally, each case is labelled with a difficulty (easy, medium, or hard) and a type, so results can be read in detail rather than as a single number. The hard cases, such as an identical name and a similar age where only the nationality differs, are the ones that genuinely test the agent.

## The agent

**`screening_agent.py`** contains the agent. It takes one flagged case, shows it to a large language model, and returns a structured decision.

### What it decides

For each case the agent returns four things: a **disposition** (genuine match, false alarm, or needs human review), a **confidence** score, the **key evidence** it actually relied on, and a short written **rationale**. The three-way choice matters: the agent is allowed to say "I cannot tell," and an honest abstention is treated differently from a wrong answer. In use, this means clear false alarms are cleared automatically with a recorded explanation, while genuine and ambiguous cases go to a human with the agent's summary attached.

### What it is allowed to see

The agent is shown only the two records a real alert carries, the customer and the candidate, plus the screening score. It is deliberately not shown the correct answer, the difficulty rating, or any of the other bookkeeping the generators record. Without this, the agent would effectively be marking its own homework, so stripping those fields is the single most important safeguard in the whole test.

### How it reasons

The agent is instructed to decide identity, not guilt: the question is never whether the listed person is dangerous, only whether the customer *is* that listed person. It is told which evidence counts and how much:

- A different date of birth, nationality, place of birth, or document number is strong evidence of **different people**.
- A matching full date of birth, nationality, place of birth, or document number is strong evidence of the **same person**. A matching document number is close to conclusive.
- Names vary legitimately, so a different spelling is not proof of a different person when other details agree, and, equally, a shared name on its own is not proof of a match. The name is why the alert exists; it cannot also be the reason to confirm it.
- Clearing a genuine match is the most serious possible error, so when the evidence genuinely does not settle the question, the agent must abstain rather than guess.

### Guardrails against invented evidence

The agent is also explicitly told what it must **not** do, and these rules were added in response to a real failure found in testing (described below):

- A field counts as evidence only when it is present on *both* records and can be directly compared. A detail present on one side and missing on the other is not evidence, it is missing information.
- Identity-document numbers carry no weight unless the list entry actually has a number to compare against, and the agent must never reason about the *format* or *structure* of an identifier.
- The agent must not infer anyone's nationality, ethnicity, or origin from the style of their name or the shape of an identifier. Such inferences are unreliable, and in a compliance record they are exactly the kind of reasoning that must not appear.
- The agent must not invent explanations that smooth over conflicting details, such as speculating that someone might have changed nationality. It adjudicates the facts as recorded; if the records conflict, that conflict is the finding.
- It may use no information beyond the two records in front of it.

### Reliability

Two different things can go wrong when calling an AI model, and the agent handles them separately. If the model replies with something unreadable, the agent asks again with a correction, and if it still cannot get a usable answer it abstains rather than guessing. If the request itself fails because the service is busy or rate-limited, the agent simply waits and retries, waiting longer each time. Neither failure is allowed to crash a run partway through a long test.

## First test results

The agent was run against a small hand-built sample of cases. 

An easy true positive match A customer whose name, full date of birth, and nationality all matched a listed person exactly. The agent correctly identified this as a genuine match with high confidence, citing precisely those three fields. This is the baseline: the agent must never miss a match this clear.

A harder false positive flag was  a customer with an *identical* name to a listed person, born only a year apart, differing only in nationality. On the first attempt, the agent flagged the case as "needing review" claiming the customer's personal identification number "is consistent with the format used in Georgia" and that the surname was "distinctly Georgian," and speculated the customer might be a naturalised citizen of Georgian origin. However, these claims were hallucinated - that number was a randomly generated string and the list entry had no identifier to compare it to. 

This is exactly the failure that would destroy trust in an agent's written rationale, and it is why the guardrails above were added. With the evidence rules tightened, the agent re-ran the same case and correctly identified it as a false alarm.

## Where the data comes from

The sanctions-list side is drawn from the UK Sanctions List, the official list maintained by the UK Foreign, Commonwealth and Development Office, accessed through OpenSanctions, which publishes that list in a clean, machine-readable form. OpenSanctions is free for non-commercial use; commercial use requires a licence. The customer side is synthetic and created by the code in this repository.
