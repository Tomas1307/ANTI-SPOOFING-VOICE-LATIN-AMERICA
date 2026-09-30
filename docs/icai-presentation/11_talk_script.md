# ICAI 2026 — Talk Script

**Paper:** Accent-Based Evaluation of Speech Anti-spoofing Countermeasures Across Multiple Languages
**Speaker:** Tomás Acosta Bernal · FLAG Lab, Universidad de los Andes
**Session:** Natural Language Processing & Virtual and Augmented Reality — Fri 16 Oct 2026, 19:00–21:00
**Slot position:** first paper of the session

---

## 0. The situation, and what it implies

Three facts from the program shape every decision below.

**You open the session.** The first speaker sets the register for the next two hours and gets the cleanest clock — no accumulated overrun. It also means the room is still filling at 19:00. Your first 60 seconds have to work for someone who sits down late and hasn't read a word of the abstract.

**The session is 120 minutes and mixes NLP with VR/AR.** Six papers are visible on the program page; two hours across six slots is ~20 minutes each, which in practice means **15 minutes of talk, 3–4 of questions, and a minute of handover**. If the session actually holds eight papers, the chair will hold you to 15 minutes flat with no mercy. So: **build for 15:00, rehearse a 12:00 cut** (marked below). Confirm the count with the chair before the session starts — it takes ten seconds and it decides whether you keep §10.

**Your audience is not a speech audience.** You are sharing a session with browser-based AR route planners, NPC-guided VR learning paths, and L2 comprehension assessment in VR. Maybe two people in that room know what an EER is. Nobody knows what LCNN, CQCC or P2S-Grad mean, and nobody needs to. Every number you say out loud needs a unit of meaning attached to it — "43 %" means nothing, "worse than flipping a coin" means everything.

**Decisions taken** (you left these open, so these are the defaults the script is written on — easy to reverse):
- Open in Spanish for ~20 seconds, then switch to English and make the switch itself the argument.
- Audio A/B demo early (minute ~2), with a callback at minute ~8 when the finding explains the demo.
- Methodology gets ~90 seconds and one diagram; the statistics stay in your pocket for Q&A.

---

## 1. Timing map

| # | Block | Runs | Ends at | Cut to 12:00? |
|---|---|---|---|---|
| 1 | Hook — Spanish, then English | 1:15 | 1:15 | trim to 1:00 |
| 2 | Audio A/B + reveal | 1:30 | 2:45 | keep |
| 3 | Who I am, whose paper this is | 0:30 | 3:15 | keep |
| 4 | The problem and the gap | 1:15 | 4:30 | keep |
| 5 | What we tested (48 / 864) | 1:15 | 5:45 | trim to 0:45 |
| 6 | It works in English. It dies in Spanish. | 1:15 | 7:00 | keep |
| 7 | **The culprit — voice conversion** | 2:30 | 9:30 | keep, never rush |
| 8 | The fix — bilingual training | 1:15 | 10:45 | keep |
| 9 | What survives — gender, not accent | 1:30 | 12:15 | keep |
| 10 | Filter-bank scale | 0:45 | 13:00 | **cut first** |
| 11 | Limitations, before they ask | 0:45 | 13:45 | trim to 0:20 |
| 12 | Takeaway, MARSA, thanks | 1:00 | 14:45 | keep |

Blocks 7 and 9 are the two things anyone will still remember on Saturday. Everything else is scaffolding and may be sacrificed to protect them.

---

## 2. The script

Notation: **[bold brackets]** = stage direction. `»` = say roughly this. Lines in quotes are worth memorising word for word; the rest is cues, not a recitation.

---

### Block 1 · Hook — Spanish, then English `1:15`
*Slide: statement, dark. "Your mother calls. She needs $1,500, now. It is her voice."*

**[Walk to centre. Do not touch the clicker. Start in Spanish, with no warning and no apology.]**

> «Te llama tu mamá.
> Está nerviosa. Necesita mil quinientos dólares, ya — te da un número de cuenta.
> Es su voz. Su forma de hablar. La muletilla que usa cuando está preocupada.
> Tú le transfieres el dinero. Claro que se lo transfieres: es tu mamá.»

**[Pause. Two full seconds. Then, in English:]**

> "If you didn't follow that — you are in good company. Neither does any anti-spoofing system in the ASVspoof benchmark."

**[Beat.]**

> "What I said was this: your mother calls, she needs fifteen hundred dollars, right now, and she gives you an account number. It is her voice. So you send it. Of course you send it — it's your mother."

**[Advance slide.]**

> "Except it isn't her. It is thirty seconds of her voice, scraped from a voice note, run through a model that anyone here could install this evening."

**Why this works:** the language switch is not a flourish. You have just made the room feel, for four seconds, exactly what the paper is about — a system that works perfectly in one language and understands nothing in another. Do not explain the parallel here. Let it sit. You will collect it in Block 12.

**Risk control:** if the room looks lost rather than intrigued at the two-second pause, cut straight to the English translation and drop the "good company" line. You lose the joke, you keep the room.

---

### Block 2 · Audio A/B + reveal `1:30`
*Slide: audio A/B, two cards, play buttons*

> "Before I tell you anything about the research, I want ten seconds of your attention and one honest answer."

**[Play A. Play B. Do not comment between them.]**

> "One of those is a real recording of a person. The other is synthetic. Hands up if you think A was the real one."

**[Count. Out loud. "About a third."]**

> "Hands up for B."

**[Count.]**

> "Right. Keep that split in your head — we come back to it in about six minutes, and the reason you split the way you did is the actual finding of this paper."

**[Advance. Reveal which was which — one line, no gloating.]**

**Setup notes:**
- Test the room's audio **before the session opens**, not at 19:00. Evening slot, tired AV staff.
- If audio fails: say "the clips are in the paper's repository, and the QR at the end takes you there", show the spectrogram fallback slide, and move on inside ten seconds. Never troubleshoot on stage.
- Two clips maximum. A third stops being a demo and becomes a listening test.

---

### Block 3 · Who I am, whose paper this is `0:30`
*Slide: speaker*

> "I'm Tomás Acosta, from the FLAG Lab at Universidad de los Andes in Bogotá. I build voice agents in production for a US startup, and I do research on the other side of that coin — how to tell whether the voice on the line is a person."

> "This paper is first-authored by Jaime Hurtado, with Rubén Manrique. I'm presenting it, and I'll show you at the end where my own work picks it up."

**Thirty seconds. Not forty.** Credibility here comes from the fact that you ship these systems, not from a list of degrees.

---

### Block 4 · The problem and the gap `1:15`
*Slide: 3-up cards — TTS / VC / why it matters*

> "There are exactly two ways to fake a voice."

> "**Text-to-speech** writes the audio from scratch. You type a sentence, a model produces the waveform. The person never said it."

> "**Voice conversion** starts from a real recording — someone actually spoke — and rewrites *who* is speaking. The words are real. The identity is stolen."

> "A detector that sits in front of a speaker-verification system has to catch both. And the benchmark everybody trains on — ASVspoof — is **entirely in English**."

*Slide: compare — 2.67 % vs > 30 %*

> "Tamayo and colleagues showed in 2023 what happens when you take those English-trained detectors to Spanish: error rates above thirty percent. What nobody had asked was the obvious follow-up question. **Which** architectures fail, how badly — and *why*."

---

### Block 5 · What we tested `1:15`
*Slide: pipeline diagram, 4 × 3 × 4*

> "So we tested all of them."

> "Every detector is two pieces: a front-end that turns audio into features, and a back-end that decides bona fide or spoofed. We took four front-ends, three back-end variants, four loss functions — that's forty-eight architectures. Each trained on English, on Spanish, and on both. Each one six times with different random seeds."

> "Eight hundred and sixty-four trained models. The point of the seeds is that nothing here is a single lucky run — when we say one thing beats another, we've checked it isn't noise."

**[If someone is going to ask about statistics, this is where you've bought yourself the right to say "that's in the paper" later.]**

**Cut to 0:45:** drop the front-end/back-end anatomy, keep "forty-eight architectures, three data conditions, six seeds each, eight hundred and sixty-four models."

---

### Block 6 · The gap is real, and it is structural `1:15`
*Slide: compare — 2.67 % vs > 30 %*

> "In English, this is a solved problem. Best configuration: two point six seven percent equal error rate. For our purposes, that means it's right roughly forty-nine times out of fifty."

> "Then we took those same English-trained models and showed them Spanish."

**[Pause.]**

> "All forty-eight configurations exceeded thirty percent. Twenty-eight of them exceeded forty. The best one managed thirty-four."

> "To be clear about what those numbers mean: at fifty percent, the system is a coin flip. We are not describing degraded performance. We are describing a detector that has stopped detecting."

**[This next line is the one that upgrades the result from 'known' to 'ours'. Land it deliberately.]**

> "Now — that English-to-Spanish gap was already reported, by Tamayo and colleagues. What forty-eight architectures buy you is this: there is **no configuration that escapes it.** Not a front-end, not a back-end, not a loss function. If you were hoping the right architecture would save you, it doesn't exist. The gap is structural, not a bad model choice."

> "So yes: you need data in the language you'll be attacked in. That much was known. The question this paper answers is *what exactly* that data is protecting you from."

---

### Block 7 · The culprit `2:30` — **the centre of the talk**
*Slide: data — grouped bars, EN-trained vs ES-trained by attack family*

> "Now. That aggregate number — 'thirty-four percent' — hides the actual result. Because the failure is not uniform."

> "We broke the same evaluation down by *attack family*: which kind of fake was it looking at?"

**[Advance. Let them read the chart for three seconds before you speak. Do not narrate the axes.]**

> "The three tall bars are voice conversion — CycleGAN, DiffVC, StarGAN. Fifty-two. Fifty-four. Sixty-six percent."

> "Sixty-six percent is worse than chance. That model isn't confused; it is *systematically wrong*. You'd do better inverting its answer."

**[Beat.]**

> "And the short bar on the right is text-to-speech. Same model. Never saw a word of Spanish. **One point seven nine percent.**"

*Slide: big number — 1.79 %*

> "That's the finding. The gap is real — but it is **not uniform**. Practically all of it is voice conversion."

**[Say this caveat out loud. It is the difference between a claim that survives Q&A and one that doesn't.]**

> "And I have to be precise about that one point seven nine, because it's the *best* English-trained model. Across back-ends, the text-to-speech column moves by an order of magnitude — one point eight, seventeen, forty-six percent. So the honest statement isn't 'TTS always transfers'. It's: **TTS can transfer across languages, and whether it does depends on your pooling strategy.** Voice conversion never transfers, whatever you do."

> "Here's the reading we offer. A vocoder — the thing text-to-speech uses to make sound — leaves artefacts that are properties of the *generator*. They don't care what language you're speaking, so the detector can still spot them. Voice conversion artefacts are tangled up with speaker identity and accent, which is precisely what changes when you change language. So they don't survive the trip."

**[Audio callback. Back to the A/B slide.]**

> "Which brings us back to your hands. The clip most of you identified correctly was text-to-speech. The one that split the room was a conversion. You failed at the same task, in the same direction, as a neural network trained on twenty-five thousand utterances."

**Delivery:** this block carries the talk. Slow down, and resist the urge to caveat it in real time. The caveats have their own slide at Block 11.

---

### Block 8 · The fix `1:15`
*Slide: compare, or card grid*

> "The fix is almost boring. Merge the corpora — train on English and Spanish together."

> "Spanish error rate drops close to zero. Under two tenths of a percent for the five leading configurations. And it costs you less than one percentage point on English: 2.67 becomes 3.64."

**[This is the practical payoff. Most of the room runs systems, not experiments — this is the line they can act on.]**

> "And note what that means, because it's not the obvious conclusion. The obvious conclusion from a language gap is 'ship an English model and a Spanish model'. You don't have to. **One bilingual model beats two monolingual ones** — it scores better on Spanish than the Spanish-only model does, zero point zero nine against zero point one four, and it keeps English within a point. One model, two languages."

> "So there's no trade-off here to agonise over. You just have to *have* the second language in your training data."

> "One honest note while I'm showing you a near-zero number: our merged corpus is fifty-eight percent Spanish. It is not language-balanced, so we can't cleanly separate 'bilingual training helped' from 'there was simply more Spanish'. That separation is future work, and it's in the paper as such."

**[Saying this yourself, unprompted, is worth more than the result itself. It buys you the room's trust for Block 9.]**

---

### Block 9 · What survives `1:30`
*Slide: table — country × gender*

> "So we trained bilingually, we got error rates near zero, and then we asked what error is *left*. The paper is called 'accent-based', so: does accent explain it?"

**[Advance.]**

> "Five Latin-American varieties — Argentina, Chile, Colombia, Peru, Venezuela. The spread between the best and the worst is under **half a percentage point**. And the ranking of architectures is identical for all five — the same back-end wins in every country."

> "Accent, essentially, does not matter."

**[Beat.]**

> "What does show up is speaker gender. And it shows up with a twist: **the direction reverses between countries.** In Colombia the system is roughly twenty times worse on male speakers than on female ones. In Argentina, Chile and Peru it runs the other way."

> "A single global bias in the front-end can't produce that — it would push the same direction everywhere. So what we're looking at is an interaction between who the speakers are and which attack generators were used on each subset. Which points somewhere useful: if you want to adapt these systems, balance your demographics. Don't build accent-specific architectures."

**Why this block earns its 90 seconds:** it's the slide that surprises people who came expecting an accent paper, and it's honest about a negative result. Audiences remember researchers who report what they didn't find.

---

### Block 10 · Filter-bank scale `0:45` — **cut this first**
*Slide: card grid or small chart*

> "One smaller result, quickly. There's a standing assumption in this literature that you should emphasise low frequencies, because the main vowel formants live down there."

> "Across the full grid: it's true for Mel-scale front-ends — lowest median error, wins eight of twelve configurations. And it's the *worst* of the three options for linear-scale ones. The justification only holds for the first formant; the second one spans six hundred to twenty-five hundred hertz and doesn't cooperate."

> "Test both, or you'll report whichever one your front-end happened to be."

---

### Block 11 · Limitations, before they ask `0:45`
*Slide: card grid, 3-up*

> "Three things I'd rather say than have you ask."

> "**One.** Our Spanish test partition is under five thousand utterances, so a zero-point-one percent error rate is a handful of trials. Treat it as 'very low', not as a precise number."

> "**Two.** The conversion attacks share target identities between training and test. Our Spanish results are an upper bound on performance under a strictly disjoint protocol."

> "**Three.** HABLA covers four attack families. We showed that conclusions depend heavily on which attacks a corpus contains — which means our own conclusions carry that same dependency."

**Cut to 0:20:** keep only number three; it's the one that sets up Block 12.

---

### Block 12 · Takeaway, MARSA, thanks `1:00`
*Slide: takeaway — highlighted*

> "So — three sentences, and they're the whole talk."

> "**The language gap is real** — no architecture escapes it, and we checked forty-eight of them."

> "**It lives in voice conversion** — English-trained models can still catch synthesised Spanish; what they cannot catch is *converted* Spanish. And conversion is exactly what the fraud in my opening story uses."

> "**And one bilingual model closes it** — you don't need one detector per language. You need the languages in the training data."

**[Beat.]**

*Slide: closing*

> "That third limitation is what I'm working on now. MARSA: a multi-accent Spanish corpus with six contemporary TTS systems, an acoustic augmentation layer, and a partial-spoof tier — utterances that are real except for the few words that were swapped in. That last one needs frame-level scoring, because an utterance-level error rate has no way to express 'three words of this sentence were fake'."

> "The paper, the code and those audio clips are behind the QR. Thank you — I'm happy to take questions."

**[Leave the closing slide up for the whole Q&A. It holds your name, your email and the QR.]**

---

## 3. Questions you will get, and the honest answer

| Question | Answer |
|---|---|
| **"Isn't this just Tamayo et al. again?"** | No — that's the premise, not the result. Tamayo showed the gap with a handful of baselines. We show (a) no configuration out of 48 escapes it, so it's structural; (b) it decomposes almost entirely into voice conversion, which was never reported; (c) one bilingual model closes it better than a monolingual one. |
| **"So you're saying language doesn't matter?"** | The opposite. Language mismatch is what triggers the collapse — Spanish-trained models catch Spanish conversion at 0.01 %. Our contribution is that the mismatch doesn't hurt uniformly: it hurts through one attack family. |
| "Isn't 0.1 % suspiciously good?" | Yes, and we flag it. Small test partition plus shared conversion-target identities. It's an upper bound. |
| "Why no single best architecture?" | Six seeds can't support crowning one. The top eight on English are statistically inseparable under seed-matched Wilcoxon, and we refuse to name a winner the data doesn't support. |
| "Doesn't 'accent doesn't matter' contradict your title?" | Directly: yes, and that's the contribution. The intuitive story is that accent drives the error. We tested it. It doesn't. Gender does. |
| "Is the bilingual result just corpus imbalance?" | We can't rule it out. 57.7 % Spanish. Named as future work, not swept under the near-zero number. |
| "Why not wav2vec / transformers?" | Named as future work. This paper deliberately covers the classical DSP + LCNN grid exhaustively rather than one modern model shallowly. |
| "Does this transfer to other language pairs?" | Unknown, and we don't claim it. The mechanism we propose — vocoder artefacts are language-independent, conversion artefacts aren't — predicts it should. That's a hypothesis, not a result. |
| (VR/AR person) "How would I use this?" | If your system authenticates anyone by voice, test it against conversion attacks in the language your users actually speak. English benchmark numbers will not transfer. |

---

## 4. Rehearsal checklist

- [ ] Confirm paper count in the session with the chair → decide 15:00 or 12:00 version
- [ ] Test room audio before 19:00; confirm clip volume from the back row
- [ ] Time a full run. If it lands over 15:30, cut Block 10 and don't negotiate with yourself
- [ ] Rehearse Block 1 in Spanish out loud — the pause is the hard part, not the words
- [ ] Rehearse Block 7 twice as often as everything else
- [ ] Have the spectrogram fallback slide ready in case audio dies
- [ ] QR tested from a phone, at distance, on the projector
