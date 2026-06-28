# Evermind — Publication & Submission Kit

This folder is a complete, submission-ready manuscript package.

## Files

| File | Purpose |
|---|---|
| `evermind-architecture.pdf` | **Read this first.** 17-page rendered paper (high-res vector figures, typeset math). |
| `evermind-architecture.html` | Self-contained render (MathJax + SVG). Open in any browser; re-print to PDF at any DPI. |
| `evermind-architecture.tex` | **The submission master** — IEEEtran LaTeX. Compiles on Overleaf as-is. |
| `figures/*.svg` | 7 vector figures (infinite resolution). |
| `SUBMISSION.md` | This file. |

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

The manuscript is deliberately written as an **architecture/systems paper**, and Section IX states the "beats frozen frontier LLMs" claims as **falsifiable hypotheses (H1–H3) with a measurement protocol — not as results**. This is intentional: a reviewer-facing "publication of fact" that claimed benchmark wins we have not run would be desk-rejected and would damage credibility. The strong, defensible contributions are the formalization (the monoid-scan proof, the single-incumbent invariant, the O(1) invalidation result) and the open, reproducible implementation. **Do not add unmeasured performance claims before submitting.** If you want the comparative results, run the protocol in §IX-C first and I will fold the numbers in.

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
- **Comments:** 17 pages, 7 figures. Reference implementation: open `builderforce-memory` package family, v2026.6.32.

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
- **Run the §IX-C evaluation protocol** against the real packages and fold measured H1–H3 numbers into the paper (turns hypotheses into results — this is what makes it competitive at TNNLS).
- Generate a TechRxiv-ready zip, or an arXiv source tarball (`.tex` + figures as PDF).
- Convert the SVGs to PDF/PNG locally if your LaTeX toolchain lacks Inkscape.
