/**
 * experience/schedule.ts — when a skill's routine is due. Pure: the host supplies the
 * clock and the local UTC offset, and ticks as often as it likes.
 */
import type { Schedule, Skill } from './types.js';

const MINUTE = 60_000;
const DAY = 86_400_000;

/** `utcOffsetMinutes` is local minus UTC (e.g. -240 for UTC-4). */
export function isDue(schedule: Schedule, lastRunAt: number | null | undefined, now: number, utcOffsetMinutes: number): boolean {
  if (schedule.every === 'minutes') {
    const every = Math.max(1, schedule.minutes) * MINUTE;
    return lastRunAt == null || now - lastRunAt >= every;
  }
  const offset = utcOffsetMinutes * MINUTE;
  const localNow = now + offset;
  const dayStart = localNow - (((localNow % DAY) + DAY) % DAY);
  const target = dayStart + Math.min(23, Math.max(0, schedule.hour)) * 3_600_000 + Math.min(59, Math.max(0, schedule.minute)) * MINUTE;
  // Due once the time of day has passed, unless it already ran since then.
  return localNow >= target && (lastRunAt == null || lastRunAt + offset < target);
}

/** Skills whose enabled routine is due now, oldest-run first. */
export function dueSkills(skills: Skill[], now: number, utcOffsetMinutes: number): Skill[] {
  return skills
    .filter((s) => s.routine?.enabled && isDue(s.routine.schedule, s.routine.lastRunAt, now, utcOffsetMinutes))
    .sort((a, b) => (a.routine?.lastRunAt ?? 0) - (b.routine?.lastRunAt ?? 0));
}
