/**
 * A long document, read the way the product reads a chat history.
 *
 * The page-scale benchmark needs one thing the chat harness never needed: provenance in pages.
 * A 1000-page report is ingested once with `pdftotext` (form feed between pages, so page numbers
 * survive), then cut into windows of adjacent pages. Each window becomes a `RetrievableSession`
 * whose id carries its page range, so a retrieval can be scored against an evidence page number:
 * a window is a hit when the gold page falls inside its range.
 *
 * Windows are dated in page order from a fixed epoch. Documents have no timestamps, and the
 * retrieval's date headers are written for chat, so a synthetic monotonic date keeps "in range
 * first" ordering inert while leaving the prompt shape byte-identical to the product's.
 */

import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import type { RetrievableSession } from '../knowledge/session-retrieval.js';

/** One page of a document, numbered from 1 as the PDF numbers it. */
export interface DocumentPage {
  page: number;
  text: string;
}

/** A window of adjacent pages: what a retrieval can choose and a reader can be shown. */
export interface PageWindow {
  id: string;
  firstPage: number;
  lastPage: number;
  text: string;
}

export const PAGE_BREAK = '\f';

/** Pages from a PDF, via poppler's pdftotext. Layout mode keeps tables legible. */
export function pagesFromPdf(path: string, options: { layout?: boolean } = {}): DocumentPage[] {
  const args = [...(options.layout === false ? [] : ['-layout']), path, '-'];
  const text = execFileSync('pdftotext', args, {
    encoding: 'utf8',
    maxBuffer: 512 * 1024 * 1024,
    // poppler warns per malformed glyph; a 1000-page report produces thousands of lines of it
    stdio: ['ignore', 'pipe', 'ignore'],
  });
  return pagesFromPageBreaks(text);
}

/** Pages from text that already carries form feeds (pdftotext output saved to disk). */
export function pagesFromTextFile(path: string): DocumentPage[] {
  return pagesFromPageBreaks(readFileSync(path, 'utf8'));
}

/**
 * Split on form feeds. A trailing break produces no page, and a page that is blank after
 * extraction still gets a number: dropping it would shift every later page number.
 */
export function pagesFromPageBreaks(text: string): DocumentPage[] {
  const parts = text.split(PAGE_BREAK);
  if (parts.length > 1 && parts[parts.length - 1]!.trim() === '') parts.pop();
  return parts.map((body, index) => ({ page: index + 1, text: body.replace(/\r\n/g, '\n') }));
}

/** How many pages a PDF has, as the file itself reports it. */
export function pdfPageCount(path: string): number {
  const info = execFileSync('pdfinfo', [path], { encoding: 'utf8' });
  const match = /^Pages:\s+(\d+)$/m.exec(info);
  if (!match) throw new Error(`pdfinfo reported no page count for ${path}`);
  return Number(match[1]);
}

export interface WindowOptions {
  /** Pages per window. Four keeps a window inside the reader's per-session byte budget. */
  pagesPerWindow?: number;
  /** A window is cut short once its text passes this, so one dense page cannot crowd out a read. */
  maxBytes?: number;
}

export const DEFAULT_PAGES_PER_WINDOW = 4;
export const DEFAULT_WINDOW_BYTES = 12 * 1024;

/**
 * Cut pages into windows. A page never straddles two windows, so a gold page number maps to
 * exactly one window and a hit is unambiguous.
 */
export function windowPages(pages: DocumentPage[], options: WindowOptions = {}): PageWindow[] {
  const perWindow = options.pagesPerWindow ?? DEFAULT_PAGES_PER_WINDOW;
  const maxBytes = options.maxBytes ?? DEFAULT_WINDOW_BYTES;
  const windows: PageWindow[] = [];
  let current: DocumentPage[] = [];
  let bytes = 0;

  const flush = () => {
    if (current.length === 0) return;
    const firstPage = current[0]!.page;
    const lastPage = current[current.length - 1]!.page;
    windows.push({
      id: windowId(firstPage, lastPage),
      firstPage,
      lastPage,
      text: current.map((page) => `[page ${page.page}]\n${page.text.trim()}`).join('\n\n'),
    });
    current = [];
    bytes = 0;
  };

  for (const page of pages) {
    const size = Buffer.byteLength(page.text, 'utf8');
    if (current.length > 0 && (current.length >= perWindow || bytes + size > maxBytes)) flush();
    current.push(page);
    bytes += size;
  }
  flush();
  return windows;
}

/** `pages-0001-0004`: sorts in page order and reads as a citation. */
export function windowId(firstPage: number, lastPage: number): string {
  return `pages-${String(firstPage).padStart(4, '0')}-${String(lastPage).padStart(4, '0')}`;
}

/** The page range in a window id, for scoring a retrieval that only reports ids. */
export function pageRangeFromId(id: string): { firstPage: number; lastPage: number } | undefined {
  const match = /^pages-(\d+)-(\d+)$/.exec(id);
  if (!match) return undefined;
  return { firstPage: Number(match[1]), lastPage: Number(match[2]) };
}

/** Does this window hold the page the label points at? */
export function windowHoldsPage(window: PageWindow | { firstPage: number; lastPage: number }, page: number): boolean {
  return page >= window.firstPage && page <= window.lastPage;
}

/** Windows are dated one day apart from this epoch, in page order. */
export const WINDOW_EPOCH = Date.parse('2020-01-01T09:00:00.000Z');
const DAY_MS = 24 * 60 * 60 * 1000;

/**
 * A window as the retrieval sees it: the page text is one user turn, because a document has no
 * speakers and the reading path treats user turns as the evidence roles for a lookup.
 */
export function windowsAsSessions(windows: PageWindow[]): RetrievableSession[] {
  return windows.map((window, index) => ({
    id: window.id,
    date: new Date(WINDOW_EPOCH + index * DAY_MS).toISOString(),
    turns: [{ role: 'user' as const, text: window.text }],
  }));
}
