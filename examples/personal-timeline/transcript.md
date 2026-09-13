# Transcript: examples/personal-timeline/run.sh

One run through the local writer (r23, Q8_0 GGUF under llama.cpp on a 16 GB M-series Mac), captured 2026-09-14. About three seconds per statement and per question. The `added:` lines are exactly what the writer extracted; the answers are assembled by code from those facts, their source sentences and the computed notes.

```

Fresh memory root: $TMP/rembero-timeline   (writer: rembero-writer at http://127.0.0.1:8081/v1)

Remembering nine statements

> My dentist is Dr Chen at Collins Street Dental.
added: dentist(user, 'Dr Chen').; works_at(user, 'Collins Street Dental').
duplicates: 0
retracted: 0

> My last dental check-up was on 3 August 2026.
added: last_dentist_visit(user, '3 August 2026').
duplicates: 0
retracted: 0

> My next dental check-up is due in February 2027.
added: next_dentist_visit(user, 'February 2027').
duplicates: 0
retracted: 0

> I am training for the Melbourne Marathon.
added: training_for(user, 'Melbourne Marathon').
duplicates: 0
retracted: 0

> My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August.
added: long_run(user, '12 km').; long_run(user, '16 km').; long_run(user, '21 km').
duplicates: 0
retracted: 0

> My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to be $65.
added: monthly_cost(user, 79).; monthly_cost(user, 65).; started_at(user, '1 September 2026').
duplicates: 0
retracted: 0

> I promised Maya a status update on the Atlas project by 18 September 2026.
added: deadline(atlas, '18 September 2026').
duplicates: 0
retracted: 0

> Our office moved to 12 Collins Street on 15 June 2026.
added: moved_to(office, '12 Collins Street', '15 June 2026').
duplicates: 0
retracted: 0

> Sam's birthday is 2 October. He likes single-origin coffee.
added: birthday(sam, '2 October').; likes(sam, single_origin_coffee).
duplicates: 0
retracted: 0

What the memory now holds
predicates:
  dentist/2()
  works_at/2()
  last_dentist_visit/2()
  next_dentist_visit/2()
  training_for/2()
  long_run/2()
  monthly_cost/2()
  started_at/2()
  deadline/2()
  moved_to/3()
  birthday/2()
  likes/2()

Asking

? When was my last dental check-up?
Evidence for last_dentist_visit(user, Date):
1. Date = '3 August 2026'
   Claims: last_dentist_visit(user, '3 August 2026')
   Sources: personal/29aa575f-b4c8-457e-9938-4fdef3106c16@2026-09-13T15:54:19.289Z "My last dental check-up was on 3 August 2026."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-08-03: "My last dental check-up was on 3 August 2026." [said 2026-09-13, "3 August 2026"] — about 6 weeks (41 days) before the question date
  (status: answered, query: last_dentist_visit(user, Date), matches: 1, trust: accepted)

? What long runs have I done so far?
Evidence for long_run(user, Distance):
1. Distance = '12 km'
   Claims: long_run(user, '12 km')
   Sources: personal/c81c8b3f-eccc-4dff-a0d8-76d814c0e343@2026-09-13T15:54:21.902Z "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-07-26: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "26 July"; year assumed from the session date] — about 7 weeks (49 days) before the question date
     - 2026-08-09: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "9 August"; year assumed from the session date] — about 5 weeks (35 days) before the question date
     - 2026-08-23: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "23 August"; year assumed from the session date] — about 3 weeks (21 days) before the question date
     Gaps between dated events (the pair whose sentences best match the question is listed first):
     - 2026-07-26 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-09 ("My long runs so far were 12 km on 26 Ju…"): 14 days (2 weeks)
     - 2026-07-26 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-23 ("My long runs so far were 12 km on 26 Ju…"): 28 days (4 weeks)
     - 2026-08-09 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-23 ("My long runs so far were 12 km on 26 Ju…"): 14 days (2 weeks)
2. Distance = '16 km'
   Claims: long_run(user, '16 km')
   Sources: personal/c81c8b3f-eccc-4dff-a0d8-76d814c0e343@2026-09-13T15:54:21.902Z "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-07-26: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "26 July"; year assumed from the session date] — about 7 weeks (49 days) before the question date
     - 2026-08-09: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "9 August"; year assumed from the session date] — about 5 weeks (35 days) before the question date
     - 2026-08-23: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "23 August"; year assumed from the session date] — about 3 weeks (21 days) before the question date
     Gaps between dated events (the pair whose sentences best match the question is listed first):
     - 2026-07-26 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-09 ("My long runs so far were 12 km on 26 Ju…"): 14 days (2 weeks)
     - 2026-07-26 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-23 ("My long runs so far were 12 km on 26 Ju…"): 28 days (4 weeks)
     - 2026-08-09 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-23 ("My long runs so far were 12 km on 26 Ju…"): 14 days (2 weeks)
3. Distance = '21 km'
   Claims: long_run(user, '21 km')
   Sources: personal/c81c8b3f-eccc-4dff-a0d8-76d814c0e343@2026-09-13T15:54:21.902Z "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-07-26: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "26 July"; year assumed from the session date] — about 7 weeks (49 days) before the question date
     - 2026-08-09: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "9 August"; year assumed from the session date] — about 5 weeks (35 days) before the question date
     - 2026-08-23: "My long runs so far were 12 km on 26 July, 16 km on 9 August and 21 km on 23 August." [said 2026-09-13, "23 August"; year assumed from the session date] — about 3 weeks (21 days) before the question date
     Gaps between dated events (the pair whose sentences best match the question is listed first):
     - 2026-07-26 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-09 ("My long runs so far were 12 km on 26 Ju…"): 14 days (2 weeks)
     - 2026-07-26 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-23 ("My long runs so far were 12 km on 26 Ju…"): 28 days (4 weeks)
     - 2026-08-09 ("My long runs so far were 12 km on 26 Ju…") to 2026-08-23 ("My long runs so far were 12 km on 26 Ju…"): 14 days (2 weeks)
  (status: answered, query: long_run(user, Distance), matches: 3, trust: accepted)

? What is my monthly gym cost?
Evidence for monthly_cost(user, Gym):
1. Gym = 79
   Claims: monthly_cost(user, 79)
   Sources: personal/ab58a601-cf85-4f47-8650-1d41115db127@2026-09-13T15:54:23.250Z "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to be $65."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-09-01: "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to b…" [said 2026-09-13, "1 September 2026"] — 12 days before the question date
     Quantities stated in the history (with the sentence each comes from):
     - 79 dollars: "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to b…"
     - 65 dollars: "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to b…"
       difference between them: 14 dollars
       ratio: 65 of 79 = 82%
2. Gym = 65
   Claims: monthly_cost(user, 65)
   Sources: personal/ab58a601-cf85-4f47-8650-1d41115db127@2026-09-13T15:54:23.250Z "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to be $65."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-09-01: "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to b…" [said 2026-09-13, "1 September 2026"] — 12 days before the question date
     Quantities stated in the history (with the sentence each comes from):
     - 79 dollars: "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to b…"
     - 65 dollars: "My gym membership at Fitness First costs $79 a month since 1 September 2026; it used to b…"
       difference between them: 14 dollars
       ratio: 65 of 79 = 82%
  (status: answered, query: monthly_cost(user, Gym), matches: 2, trust: accepted)

? When is the Atlas status update due?
Evidence for deadline(atlas, DueDate):
1. DueDate = '18 September 2026'
   Claims: deadline(atlas, '18 September 2026')
   Sources: personal/a494a688-5688-45a3-98b2-77d4dc5a53dd@2026-09-13T15:54:23.955Z "I promised Maya a status update on the Atlas project by 18 September 2026."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-09-18: "I promised Maya a status update on the Atlas project by 18 September 2026." [said 2026-09-13, "18 September 2026"] — 5 days after the question date
  (status: answered, query: deadline(atlas, DueDate), matches: 1, trust: accepted)

? When did our office move to Collins Street?
Evidence for moved_to(office, '12 Collins Street', Date):
1. Date = '15 June 2026'
   Claims: moved_to(office, '12 Collins Street', '15 June 2026')
   Sources: personal/98748ea7-3b90-4d88-81a8-d7ce3fa4df8c@2026-09-13T15:54:24.837Z "Our office moved to 12 Collins Street on 15 June 2026."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-06-15: "Our office moved to 12 Collins Street on 15 June 2026." [said 2026-09-13, "15 June 2026"] — about 3 months (13 weeks, 90 days) before the question date
  (status: answered, query: moved_to(office, '12 Collins Street', Date), matches: 1, trust: accepted)

? What could I get Sam for his birthday?
Evidence for likes(sam, Gift):
1. Gift = single_origin_coffee
   Claims: likes(sam, single_origin_coffee)
   Sources: personal/c4e8b73c-875e-4cfe-b42f-75d07e691ef4@2026-09-13T15:54:25.673Z "Sam's birthday is 2 October. He likes single-origin coffee."
   Computed (deterministic, from the sources above):
     Dated events (each temporal expression resolved against the date of the session it was said in):
     - 2026-10-02: "Sam's birthday is 2 October." [said 2026-09-13, "2 October"; year assumed from the session date] — about 3 weeks (19 days) after the question date
  (status: answered, query: likes(sam, Gift), matches: 1, trust: accepted)

? Who is my accountant?
I have no relevant memories to answer that.
  (status: unanswerable, query: n/a, matches: 0, trust: accepted)
Related knowledge (discovery only; not an answer or proof):
  No local lexical matches.

Memory for this run is in $TMP/rembero-timeline (safe to delete).
```
