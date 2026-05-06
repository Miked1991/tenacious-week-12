# Morning Call Summary — Pair Day 2

**Pair:** Mikias Dagem & Nahom Desalegn  
**Written by:** Mikias Dagem  
**Date:** 2026-05-06

---

My question going into the call was vague — something along the lines of "what is retry logic and how do I add it." Nahom pulled it apart: which exact call fails, in which file, and what does the system do when it fails? Once I had to answer that, the question became concrete. I focused on three files where a single timeout means silent data loss — `booking_handler.py` dropping a Cal.com booking, `email_outreach.py` losing a Resend send, and `enrichment_pipeline.py` silently skipping an enrichment. Adding "would all three need the same strategy, and why?" was what made it a real research question rather than an implementation checklist, since it forced the answer to deal with idempotency differences between the files.

Nahom's starting point was broad too — he wanted to understand how function-calling works at the token level, but hadn't identified what specifically he was unsure about. I asked him to pick one: encoding, selection, or argument generation. We kept narrowing until we landed on a single falsifiable claim — does the description string shape the probability distribution that determines which tool name gets generated, or does the model only consult the description afterward? Framing it as a causal-ordering question gave us something precise enough to actually research and answer.

Both questions were locked in before the call ended.
