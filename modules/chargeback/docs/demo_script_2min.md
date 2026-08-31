# 2-minute demo script

Timed for a live run-through or a single-take screen recording. Total: 120s, with
~8s of slack built in. Each line is what you say; each bracket is what you click,
timed to land right before you say the line after it.

**Before you hit record:** run one Live Simulation round to completion (Sony-style
attack, or whatever's loaded) so a finished round card is already on screen, and have
the Example Gallery tab pre-scrolled to the Sony ULT WEAR case. Never trigger a live
vision-inspection call on camera -- it's 5 API calls and can take 15-30s, which alone
blows a third of your budget. Everything you narrate over should already be rendered.

---

### 0:00–0:12 — The problem (12s)

> "GenAI just made chargeback fraud trivial: type a prompt, get a photo of 'damaged'
> merchandise good enough to fool a support agent. Mastercard reason code 4853 --
> item not as described -- is the fastest-growing dispute category, and it's about
> to get a lot worse."

*(No click yet -- title screen or Live Simulation tab, static.)*

### 0:12–0:22 — What Aegis is (10s)

> "Aegis is a red-team-vs-blue-team lab: we generate the attacks ourselves, and we
> build the defense against them, side by side, in one pipeline."

**[Click: Attack Taxonomy tab]**

### 0:22–0:50 — Attack diversity (28s)

> "The Red Team isn't one canned fraud script -- it's five social-engineering
> tactics crossed with six image-forgery techniques, thirty combinations, sampled or
> swept automatically."

**[Click: Live Simulation tab, scroll to the pre-run round]**

> "Here's one: a customer citing a fabricated return policy, with two AI-generated
> photos of a cracked ear cup from different angles -- built img2img so the two
> angles actually look consistent, which is exactly what makes this attack class
> hard."

### 0:50–1:25 — The defense, and the novel part (35s)

**[Click: Example Gallery tab, Sony ULT WEAR card, scroll to Evidence section]**

> "Naive detection just asks a vision model 'do these match?' -- and it says yes,
> because a single holistic glance misses subtle shape drift. So we don't trust one
> opinion."

**[Click: expand "Angle consistency breakdown"]**

> "We run three independent checks: holistic reasoning, a deterministic geometry
> check that measures damage position off two landmarks in Python -- not model
> judgment -- and a focused check on cropped, isolated close-ups. Here, holistic
> says 0.85, consistent. Geometry says 0.09. They disagree by 0.8. That disagreement
> alone is a red flag we escalate on, even before averaging anything."

**[Point to decision badge: ESCALATE]**

> "Result: escalated to a human, not silently approved."

### 1:25–1:45 — Proof it works at scale (20s)

**[Click: Batch Evaluation tab, pre-run results]**

> "Across all thirty attack combinations plus genuine claims, that's where our
> precision, recall, and AUC numbers come from -- not cherry-picked, the full sweep."

*(Let the metrics/ROC chart sit on screen for a beat -- don't over-narrate numbers.)*

### 1:45–2:00 — Human-in-the-loop, and close (15s)

**[Click: Try It Yourself tab]**

> "And for the ambiguous cases, there's a real chat you can try yourself -- upload
> your own photos, watch the risk signals update live, and the human always makes
> the final call. That's Aegis: attack it, defend it, and know exactly when to ask
> for help."

---

## Fallback cuts (if you're running long)

- Cut Attack Taxonomy entirely (0:12–0:22) -- fold "five tactics, six techniques"
  into the 0:00–0:12 hook instead. Saves ~10s.
- Cut Try It Yourself (1:45–2:00) -- end on the Batch Evaluation numbers with one
  closing line: "That's Aegis -- attack it, defend it, prove it." Saves ~10s.
- Never cut the consistency-breakdown beat (0:50–1:25) -- it's the one moment that's
  actually novel rather than "another LLM wrapper," and it's what separates this
  from every other fraud-detection pitch in the room.
