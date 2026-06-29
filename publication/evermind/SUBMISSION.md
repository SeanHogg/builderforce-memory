# Evermind — Publication & Submission Kit

This folder is a complete, submission-ready manuscript package.

## Files

| File | Purpose |
|---|---|
| `evermind-architecture.pdf` | **Read this first.** 17-page rendered paper (high-res vector figures, typeset math). *Generated artifact — rebuild from `.tex` after a source edit (see below).* |
| `evermind-architecture.html` | Self-contained render (MathJax + SVG). Open in any browser; re-print to PDF at any DPI. |
| `evermind-architecture.tex` | **The submission master** — IEEEtran LaTeX. Compiles on Overleaf as-is. |
| `figures/*.svg` | 7 vector figures (infinite resolution). |
| `evermind-techrxiv-supplementary.zip` | **TechRxiv upload bundle** — PDF + `.tex` + HTML + figures, ready to attach as supplementary material. *Generated artifact — re-zip after a source edit.* |
| `PEER-REVIEW.md` | Adversarial scientific referee report on the architecture + implementation (findings logged to the Gap Register). |
| `SUBMISSION.md` | This file. |

> **Rebuild note (2026-06-29).** The `.tex` and `.html` masters were updated to record that the language-model **benchmarking harness now ships** in the open packages and on-device in the Studio — it is no longer a released skeleton (see §IX). `evermind-architecture.pdf` and `evermind-techrxiv-supplementary.zip` are generated artifacts and are **stale** until rebuilt: open the `.tex` on Overleaf to regenerate the PDF, then re-zip PDF + `.tex` + `.html` + `figures/` as the supplementary bundle. (A local LaTeX/Inkscape toolchain was not available to rebuild them in place.)

## How to post to TechRxiv (the step only you can do)
1. Sign in / create an account at **https://www.techrxiv.org** (needs your ORCID).
2. *Submit* → upload **`evermind-architecture.pdf`** as the main manuscript.
3. Attach **`evermind-techrxiv-supplementary.zip`** as supplementary files.
4. Paste the **title, authors, abstract, and category** from the metadata block below; paste the cover letter if prompted.
5. Affirm authorship + that it is not under review elsewhere → **Submit for moderation** (TechRxiv posts within ~1–2 business days and mints a DOI).

I cannot perform steps 1–5 for you: TechRxiv has no public submission API, and posting requires your authenticated account, ORCID, and a personal authorship attestation that I must not make on your behalf.

To regenerate the PDF from source: open `.tex` on [Overleaf](https://overleaf.com) (it auto-converts the SVGs via the `svg` package), or print `.html` to PDF from a browser.

---

## Honest status: I cannot click "submit" for you

Submitting to **IEEE Xplore** (the venue behind the article you referenced) is not something an agent can do autonomously, and not because of a tooling gap — it is a credentialed, outward-facing act that is legally and ethically yours to perform. It requires:

- An **IEEE/ScholarOne (Manuscript Central) account** and login;
- An **ORCID** and a declared **institutional affiliation**;
- **Author consent / authorship attestation** (I must not assert authorship on your behalf);
- A **copyright transfer (eCF)** and, for open access, an **Article Processing Charge** (IEEE OA APCs are typically ~US$1,800–2,800);
- Passage through **human peer review** — IEEE Xplore is *indexed*, not self-publish; a DOI appears only after acceptance.

So I have done everything up to that gate: a clean IEEEtran manuscript, figures, and the cover material below. The single remaining step — creating the account and pressing submit — is yours.

## Important integrity note (please read before submitting)

The manuscript is deliberately written as an **architecture/systems paper**, and Section IX states the "beats frozen frontier LLMs" claims as **falsifiable hypotheses (H1–H3) with a measurement protocol — not as results**. This is intentional: a reviewer-facing "publication of fact" that claimed benchmark wins we have not run would be desk-rejected and would damage credibility. The strong, defensible contributions are the formalization (the monoid-scan proof, the single-incumbent invariant, the O(1) invalidation result) and the open, reproducible implementation. **Do not add unmeasured performance claims before submitting.**

Note that the **measurement instrument is no longer missing**: the language-model benchmarking harness (held-out perplexity, bits-per-token, next-token accuracy, throughput, and pairwise A/B) ships in `memory-engine/src/bench`, is exercised by 14 unit tests, and runs on-device in the Studio. So H2/H3 now reduce to *running* the shipped harness at scale against a frozen baseline (H1 still needs its temporal-edit driver built around the recall API). That makes the comparative numbers cheap to produce — but until they are actually run they stay hypotheses, and the manuscript must keep saying so. If you want the comparative results, run the harness in §IX-C first and I will fold the numbers in.

---

## Recommended path (fastest → most authoritative)

### 1. Preprint now: TechRxiv or arXiv (self-serve, free, gets a DOI/handle today)
- **TechRxiv** (https://www.techrxiv.org) — **IEEE's own preprint server.** Closest thing to "an IEEE venue" you can post yourself, no peer review, citable DOI, moderated within ~1–2 days. Recommended given you referenced IEEE Xplore.
- **arXiv** (cs.LG / cs.CL) — needs an arXiv account; first submission may require endorsement.

Use the metadata block below for either.

### 2. Peer-reviewed venue (for the indexed "publication of fact")
| Venue | Fit | Notes |
|---|---|---|
| **IEEE Access** | ★★★★ | Broad, fast (~4–6 wk first decision), open access (APC). Best IEEE-Xplore-indexed home for a systems+formalization paper. |
| **IEEE Trans. Neural Networks & Learning Systems (TNNLS)** | ★★★ | More prestige, stricter; reviewers will want H1–H3 *measured*. Submit after running §IX-C. |
| **JMLR / TMLR** | ★★★★ | TMLR is fast, rigorous, open, and friendly to well-formalized systems work without SOTA benchmarks. Strong alternative to IEEE. |
| **MLSys** | ★★★ | If you lean into the WebGPU/on-device-training systems angle. |

My suggestion: **post to TechRxiv this week**, then submit to **IEEE Access** (or TMLR) once you've decided whether to run the benchmark protocol first.

---

## arXiv / TechRxiv metadata

- **Title:** Evermind: A Self-Updating, On-Device Cognitive Architecture Unifying Selective State-Space Generation, Write-Through Knowledge, and Trainable Affect
- **Authors:** Sean Hogg (Builderforce.ai)
- **Primary category:** cs.LG · **Cross-list:** cs.CL, cs.AI
- **Abstract:** (use the abstract from the manuscript verbatim)
- **Comments:** 17 pages, 7 figures. Reference implementation: open `builderforce-memory` package family, v2026.6.35.

---

## Cover letter (paste into the submission portal; fill the brackets)

> Dear Editor,
>
> Please consider our manuscript, "Evermind: A Self-Updating, On-Device Cognitive Architecture Unifying Selective State-Space Generation, Write-Through Knowledge, and Trainable Affect," for publication in [VENUE].
>
> Contemporary large language models are frozen at a training cutoff; retrieval augmentation mitigates the symptom with append-only stores in which stale and current facts coexist until reconciled by hand. We present a three-layer architecture that treats knowledge *currency* as the primary design axis: a linear-time selective state-space cortex that trains on-device, a write-through knowledge memory that replaces beliefs on write rather than appending them, and a trainable affective layer.
>
> Our contributions are both formal and systems-oriented. We give a unified mathematical treatment of the selective-scan cortex and prove its recurrence is a parallelizable monoid scan (O(log L) span). We formalize "Write-Through Cognition" and prove a single-incumbent invariant — by construction the store never holds two contradictory live facts under one key — together with an O(1) cache-invalidation result. The complete system is implemented as an open, zero-runtime-dependency WebGPU package family, with ONNX export verified to <1e-5 logit parity against the reference forward pass.
>
> We are explicit that the comparative performance theses against frozen frontier models are stated as falsifiable hypotheses with a reproducible measurement protocol, not as benchmark results. We believe the formalization and the open implementation make the work a useful and easily falsifiable object of study for the community.
>
> This manuscript is original, not under review elsewhere [confirm], and all authors consent to submission.
>
> Sincerely,
> Sean Hogg, Builderforce.ai — seanhogg@gmail.com

---

## What I can do next (just ask)
- Tailor the manuscript + cover letter to one specific venue's template (e.g. IEEE Access `\documentclass` and word/figure limits).
- **Run the §IX-C evaluation protocol** against the real packages — H2/H3 now drive the shipped `bench` harness (`benchmarkModel` / `compareModels` / `trainAndBenchmark`) directly, so this is "run it at scale," not "build it first" — and fold measured H1–H3 numbers into the paper (turns hypotheses into results — this is what makes it competitive at TNNLS).
- Generate a TechRxiv-ready zip, or an arXiv source tarball (`.tex` + figures as PDF).
- Convert the SVGs to PDF/PNG locally if your LaTeX toolchain lacks Inkscape.
