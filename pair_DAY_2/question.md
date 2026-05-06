# Pair Day 2 — Research Question

**Author:** Mikias Dagem  
**Date:** 2026-05-06

---

## Question

The Tenacious conversion engine calls Cal.com, Resend, Crunchbase, PDL, and Africa's Talking with no retry logic in place — a single timeout means permanent, silent failure. Research and explain the following:

1. What distinguishes a **transient** failure from a **permanent** API failure, and why does that distinction matter before retrying?
2. How does **exponential backoff with jitter** work, and why does it prevent retry storms when multiple clients fail simultaneously?
3. What is a **retry budget**, why is a ceiling necessary, and what happens to your system without one?
4. Walk through how you would apply a correct retry strategy to each of the three affected files — `booking_handler.py`, `email_outreach.py`, and `enrichment_pipeline.py`. Would each file require the same approach? Why or why not?
