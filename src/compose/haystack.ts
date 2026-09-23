/**
 * Hide a world's pages inside real document pages.
 *
 * Tiers are nested: the 100-page haystack's filler is the first filler pages of the 500-page one,
 * and every world page sits at the same *relative* depth in each tier. So between tiers nothing
 * changes but the amount of text around the evidence — which is the variable the sweep is about.
 */

import { Rng } from './rng.js';
import type { RenderedPage } from './render.js';
import type { World } from './world.js';

export interface Haystack {
  pages: string[];
  /** Evidence key → page numbers (1-based) holding it. */
  pagesOf: Map<string, number[]>;
  /** Where each world page landed, for reports. */
  worldPageNumbers: number[];
}

/** Words that must not appear in filler, or a filler page could answer (or confuse) a question. */
export function worldVocabulary(world: World): string[] {
  const words = new Set<string>();
  for (const p of world.people) words.add(p.last);
  for (const s of world.suppliers) words.add(s.name.split(' ')[0]!);
  for (const s of world.sites) words.add(s.name.split(' ')[0]!);
  for (const c of world.contracts) words.add(c.ref);
  words.add(world.organisation);
  return [...words];
}

export function cleanFiller(pages: readonly string[], vocabulary: readonly string[], minChars = 400): string[] {
  const patterns = vocabulary.map((word) => new RegExp(`\\b${word.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b`, 'i'));
  return pages.filter((page) => page.trim().length >= minChars && !patterns.some((pattern) => pattern.test(page)));
}

/** Relative depths in (0, 1), one per world page, seeded and sorted so document order holds. */
export function relativeDepths(count: number, seed: number): number[] {
  const rng = new Rng(seed ^ 0xdeca);
  return Array.from({ length: count }, () => 0.01 + rng.next() * 0.98).sort((a, b) => a - b);
}

export function buildHaystack(
  worldPages: readonly RenderedPage[],
  filler: readonly string[],
  totalPages: number,
  depths: readonly number[],
): Haystack {
  if (worldPages.length >= totalPages) throw new Error(`a ${totalPages}-page haystack cannot hold ${worldPages.length} world pages`);
  const needed = totalPages - worldPages.length;
  if (filler.length < needed) throw new Error(`need ${needed} filler pages, have ${filler.length}`);

  const slots: Array<number | undefined> = new Array(totalPages).fill(undefined);
  const worldPageNumbers: number[] = [];
  worldPages.forEach((_page, i) => {
    let at = Math.min(totalPages - 1, Math.round(depths[i]! * (totalPages - 1)));
    while (slots[at] !== undefined) at = (at + 1) % totalPages;
    slots[at] = i;
    worldPageNumbers.push(at + 1);
  });

  const pages: string[] = [];
  const pagesOf = new Map<string, number[]>();
  let next = 0;
  slots.forEach((slot, index) => {
    if (slot === undefined) {
      pages.push(filler[next++]!);
      return;
    }
    const page = worldPages[slot]!;
    pages.push(page.text);
    for (const key of page.keys) pagesOf.set(key, [...(pagesOf.get(key) ?? []), index + 1]);
  });
  return { pages, pagesOf, worldPageNumbers };
}
