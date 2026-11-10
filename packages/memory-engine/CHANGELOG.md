# Changelog

## 2026.10.65

- New light subpath `./ml/tasks/check-scheduler` (`CHECK_SCHEDULER_TASK`, `CHECK_SCHEDULER_MODEL_CARD`,
  `checkSchedulerFeatures`, `CHECK_SCHEDULER_FEATURES`), exported like `./ml/classical`: it never loads the engine
  root, a package, a `node:` builtin, the LM or the WGSL kernels (`tests/regression.checkSchedulerSubpath.test.ts`),
  so browser and Worker bundles can read the card. `logScale` / `clampRatio` moved to the leaf `ml/scale.ts`;
  `ml/features.ts` and the root still export them.

## 2026.10.64

- `ml/tasks`: new host row `run.check-scheduler` (`CHECK_SCHEDULER_TASK`, `checkSchedulerFeatures`,
  `CHECK_SCHEDULER_FEATURES`, `CHECK_SCHEDULER_MODEL_CARD`) — will running an optional expensive check
  (benchmark, diff coverage, mutation testing) be decisive for a verdict? Builderforce.ai's PRD 41 D9.1
  dynamic-tier scheduler runs model version `beta-cell.1` (a per-cell Beta–Bernoulli posterior, on device)
  under this task id; the row's logistic engine is its later pooled upgrade (trains from 2000 pooled samples;
  the remaining budget is a decision constraint, not a feature). The card is the ONE source of the host's
  `minObservations` (20) and `explorationShare` (0.1). Not in `BUILTIN_ML_TASKS`: a host registers it.
