/** A small seeded generator (mulberry32), so a world is the same world every time it is built. */
export class Rng {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** A float in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  int(min: number, maxInclusive: number): number {
    return min + Math.floor(this.next() * (maxInclusive - min + 1));
  }

  pick<T>(values: readonly T[]): T {
    if (values.length === 0) throw new Error('cannot pick from an empty list');
    return values[Math.floor(this.next() * values.length)]!;
  }

  /** A shuffled copy (Fisher-Yates). */
  shuffle<T>(values: readonly T[]): T[] {
    const out = [...values];
    for (let i = out.length - 1; i > 0; i -= 1) {
      const j = Math.floor(this.next() * (i + 1));
      [out[i], out[j]] = [out[j]!, out[i]!];
    }
    return out;
  }

  /** `count` distinct items. */
  sample<T>(values: readonly T[], count: number): T[] {
    if (count > values.length) throw new Error(`cannot sample ${count} from ${values.length}`);
    return this.shuffle(values).slice(0, count);
  }

  chance(probability: number): boolean {
    return this.next() < probability;
  }
}
