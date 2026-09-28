/**
 * Distillation — a teacher's exemplars adapted into a student, behind two ports.
 * Engine-free: the `@seanhogg/builderforce-memory/distillation` subpath.
 */
export { DistillationEngine } from './DistillationEngine.js';
export type {
    DistillOptions,
    DistillResult,
    DistillBatchResult,
    DistillationLog,
    DistillSkipReason,
    QualityGate,
    RehearsalOptions,
} from './DistillationEngine.js';
export { ssmRuntimeStudent, SSM_STUDENT_DEFAULT_ADAPT } from './ports.js';
export type { DistillationTeacher, DistillationStudent } from './ports.js';
