import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { describe, expect, it } from 'vitest';

import {
  DOCUMENT_MEMORG_FORMAT,
  DOCUMENT_MEMORG_ROOT_KEY,
  createDocumentMemorgExport,
  serializeDocumentMemorgExport,
  verifyDocumentMemorgExport,
} from '../src/document/memorg.js';

describe('real document Memorg export', () => {
  it('maps the full corpus into parent-first Memorg memory items', () => {
    const exported = createDocumentMemorgExport();
    const verified = verifyDocumentMemorgExport(exported);

    expect(exported).toMatchObject({
      format: DOCUMENT_MEMORG_FORMAT,
      version: 1,
      target: {
        package: 'memorg',
        version: '0.1.2',
        method: 'MemorgSystem.create_memory_item',
      },
    });
    expect(exported.items[0]).toMatchObject({
      key: DOCUMENT_MEMORG_ROOT_KEY,
      parent_key: null,
    });
    expect(verified).toMatchObject({
      valid: true,
      itemCount: 66,
      documentCount: 4,
      acceptedClaimCount: 17,
      proposedClaimCount: 4,
      reviewedRuleCount: 8,
      questionCount: 12,
    });
    const seen = new Set<string>();
    for (const item of exported.items) {
      if (item.parent_key !== null) expect(seen.has(item.parent_key)).toBe(true);
      seen.add(item.key);
    }
  });

  it('preserves source hashes and proposed-only authority in Memorg metadata', () => {
    const exported = createDocumentMemorgExport();
    const w9 = exported.items.find((item) => item.key === 'document-irs-w9-english');
    const proposed = exported.items.find(
      (item) => item.key === 'claim-un-multilingualism-un-claim-future-expansion'
    );

    expect(w9).toMatchObject({
      item_type: 'document',
      parent_key: DOCUMENT_MEMORG_ROOT_KEY,
      metadata: {
        authority: 'source',
        publisher: 'Internal Revenue Service',
        pdf_sha256: '2d420cbb4123dcf1fb82595b2359cfbb5d81f00b9df9d359fcc7af361d093f53',
        page_images: [
          expect.objectContaining({
            image_sha256: '05ebe549fcd6805423e1757b45b9901bd3d14ccd5d80281c06ce249f05948dda',
          }),
        ],
      },
    });
    expect(proposed).toMatchObject({
      metadata: {
        authority: 'proposed_only',
        review_state: 'proposed',
      },
      tags: expect.arrayContaining(['review-required']),
    });
    expect(proposed?.tags).not.toContain('proof-bearing');
  });

  it('serializes deterministically and rejects content or hierarchy tampering', () => {
    const first = createDocumentMemorgExport();
    const second = createDocumentMemorgExport();
    const text = serializeDocumentMemorgExport(first);

    expect(first).toEqual(second);
    expect(verifyDocumentMemorgExport(text).sha256).toBe(first.sha256);

    const tampered = structuredClone(first);
    tampered.items[1]!.content = 'tampered';
    expect(() => verifyDocumentMemorgExport(tampered)).toThrow(/sha-256 validation/i);

    const invalidHierarchy = structuredClone(first);
    invalidHierarchy.items[1]!.parent_key = 'missing-parent';
    expect(() => verifyDocumentMemorgExport(invalidHierarchy)).toThrow(/follow its parent/i);
  });

  it('matches the frozen user-facing Memorg artifact', () => {
    const path = fileURLToPath(
      new URL(
        '../docs/research/results/document-intelligence.memorg.json',
        import.meta.url
      )
    );
    const frozen = readFileSync(path, 'utf8').trim();
    const publicPath = fileURLToPath(
      new URL(
        '../web/public/documents/document-intelligence.memorg.json',
        import.meta.url
      )
    );
    const browserArtifact = readFileSync(publicPath, 'utf8').trim();
    const generated = serializeDocumentMemorgExport(createDocumentMemorgExport());

    expect(JSON.parse(frozen)).toEqual(JSON.parse(generated));
    expect(browserArtifact).toBe(frozen);
    expect(verifyDocumentMemorgExport(frozen)).toMatchObject({
      valid: true,
      itemCount: 66,
      documentCount: 4,
    });
  });
});
