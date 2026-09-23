/**
 * Dates as the engine sees them (yyyymmdd integers, so `F <= D` is date order) and as documents
 * write them (several formats, on purpose: a reader must recognise "14/03/2025" and
 * "14 March 2025" as the same day).
 */

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December',
];

export type Ymd = number;

export function ymd(year: number, month: number, day: number): Ymd {
  return year * 10_000 + month * 100 + day;
}

export function parts(date: Ymd): { year: number; month: number; day: number } {
  return { year: Math.floor(date / 10_000), month: Math.floor(date / 100) % 100, day: date % 100 };
}

/** A far-future sentinel for "still in force". */
export const OPEN_END: Ymd = 99_991_231;

const DAYS_IN_MONTH = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

/** Days since 2000-01-01, for adding and ordering. */
export function toDayNumber(date: Ymd): number {
  const { year, month, day } = parts(date);
  let days = 0;
  for (let y = 2000; y < year; y += 1) days += isLeap(y) ? 366 : 365;
  for (let m = 1; m < month; m += 1) days += DAYS_IN_MONTH[m - 1]! + (m === 2 && isLeap(year) ? 1 : 0);
  return days + day - 1;
}

export function fromDayNumber(days: number): Ymd {
  let year = 2000;
  let rest = days;
  while (rest >= (isLeap(year) ? 366 : 365)) {
    rest -= isLeap(year) ? 366 : 365;
    year += 1;
  }
  let month = 1;
  for (;;) {
    const length = DAYS_IN_MONTH[month - 1]! + (month === 2 && isLeap(year) ? 1 : 0);
    if (rest < length) break;
    rest -= length;
    month += 1;
  }
  return ymd(year, month, rest + 1);
}

export function addDays(date: Ymd, days: number): Ymd {
  return fromDayNumber(toDayNumber(date) + days);
}

function isLeap(year: number): boolean {
  return (year % 4 === 0 && year % 100 !== 0) || year % 400 === 0;
}

export type DateStyle = 'long' | 'iso' | 'slash' | 'short';

/** 14 March 2025 · 2025-03-14 · 14/03/2025 · 14 Mar 2025 */
export function formatDate(date: Ymd, style: DateStyle): string {
  const { year, month, day } = parts(date);
  const two = (n: number) => String(n).padStart(2, '0');
  switch (style) {
    case 'long':
      return `${day} ${MONTHS[month - 1]} ${year}`;
    case 'iso':
      return `${year}-${two(month)}-${two(day)}`;
    case 'slash':
      return `${two(day)}/${two(month)}/${year}`;
    case 'short':
      return `${day} ${MONTHS[month - 1]!.slice(0, 3)} ${year}`;
  }
}

/** Every way a correct answer may write this date. */
export function dateSpellings(date: Ymd): string[] {
  const { year, month, day } = parts(date);
  return [
    formatDate(date, 'long'),
    formatDate(date, 'iso'),
    formatDate(date, 'slash'),
    formatDate(date, 'short'),
    `${MONTHS[month - 1]} ${day}, ${year}`,
    `${day}.${month}.${year}`,
  ];
}
