import { createHash } from 'node:crypto';

import { MAX_OUTPUT_BYTES, assertBoundedOutput } from '../safety.js';
import {
  DOCUMENT_SHOWCASE_FIXTURES,
  type DocumentShowcaseFixture,
  type DocumentShowcaseFixtureClaim,
  type DocumentShowcaseFixtureQuestion,
  type DocumentShowcaseFixtureRegion,
  type DocumentShowcaseFixtureRule,
} from './showcase-fixture.js';

export const DOCUMENT_MEMORG_FORMAT = 'remembero-memorg-import';
export const DOCUMENT_MEMORG_VERSION = 1;
export const DOCUMENT_MEMORG_TARGET_VERSION = '0.1.2';
export const DOCUMENT_MEMORG_ROOT_KEY = 'remembero-document-intelligence';
export const DOCUMENT_MEMORG_MAX_ITEMS = 1_000;

export type DocumentMemorgItemType = 'custom' | 'document';
export type DocumentMemorgAuthority =
  | 'collection'
  | 'source'
  | 'evidence'
  | 'accepted'
  | 'proposed_only'
  | 'reviewed_rule'
  | 'evaluation_contract';

export interface DocumentMemorgItem {
  key: string;
  item_type: DocumentMemorgItemType;
  parent_key: string | null;
  content: string;
  metadata: Record<string, unknown>;
  tags: string[];
}

export interface DocumentMemorgExport {
  format: typeof DOCUMENT_MEMORG_FORMAT;
  version: typeof DOCUMENT_MEMORG_VERSION;
  target: {
    package: 'memorg';
    version: typeof DOCUMENT_MEMORG_TARGET_VERSION;
    method: 'MemorgSystem.create_memory_item';
  };
  items: DocumentMemorgItem[];
  sha256: string;
}

export interface DocumentMemorgVerification {
  valid: true;
  sha256: string;
  bytes: number;
  itemCount: number;
  documentCount: number;
  acceptedClaimCount: number;
  proposedClaimCount: number;
  reviewedRuleCount: number;
  questionCount: number;
}

const SHA256_RE = /^[a-f0-9]{64}$/;
const KEY_RE = /^[a-z0-9_-]+$/;
const ITEM_TYPES = new Set<DocumentMemorgItemType>(['custom', 'document']);
const AUTHORITIES = new Set<DocumentMemorgAuthority>([
  'collection',
  'source',
  'evidence',
  'accepted',
  'proposed_only',
  'reviewed_rule',
  'evaluation_contract',
]);

function compareText(left: string, right: string): number {
  return Buffer.compare(Buffer.from(left, 'utf8'), Buffer.from(right, 'utf8'));
}

function digest(value: string): string {
  return createHash('sha256').update(value).digest('hex');
}

function exactKeys(value: Record<string, unknown>, keys: string[], label: string): void {
  const actual = Object.keys(value).sort();
  const expected = [...keys].sort();
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    throw new Error(`${label} has unexpected or missing fields`);
  }
}

function canonicalTags(values: string[]): string[] {
  return Array.from(new Set(values)).sort(compareText);
}

function authorityMetadata(
  authority: DocumentMemorgAuthority,
  extra: Record<string, unknown> = {}
): Record<string, unknown> {
  return { authority, ...extra };
}

function documentKey(fixture: DocumentShowcaseFixture): string {
  return `document-${fixture.id}`;
}

function regionKey(fixture: DocumentShowcaseFixture, region: DocumentShowcaseFixtureRegion) {
  return `region-${fixture.id}-${region.id}`;
}

function claimKey(fixture: DocumentShowcaseFixture, claim: DocumentShowcaseFixtureClaim) {
  return `claim-${fixture.id}-${claim.id}`;
}

function sourceRegion(
  fixture: DocumentShowcaseFixture,
  regionId: string
): DocumentShowcaseFixtureRegion {
  const region = fixture.pages
    .flatMap((page) => page.regions)
    .find((entry) => entry.id === regionId);
  if (region === undefined) throw new Error(`Memorg claim has no source region ${regionId}`);
  return region;
}

function ruleKey(fixture: DocumentShowcaseFixture, rule: DocumentShowcaseFixtureRule) {
  return `rule-${fixture.id}-${rule.id}`;
}

function questionKey(
  fixture: DocumentShowcaseFixture,
  question: DocumentShowcaseFixtureQuestion
) {
  return `question-${fixture.id}-${question.id}`;
}

function rootItem(fixtures: readonly DocumentShowcaseFixture[]): DocumentMemorgItem {
  return {
    key: DOCUMENT_MEMORG_ROOT_KEY,
    item_type: 'custom',
    parent_key: null,
    content:
      `Remembero real-PDF document intelligence memory. ` +
      `${fixtures.length} source-attributed documents preserve parsing evidence, review state, ` +
      'deterministic recall, proof rules, and honest abstention boundaries.',
    metadata: authorityMetadata('collection', {
      workflow: 'document_intelligence',
      document_count: fixtures.length,
      proof_policy:
        'Only accepted claims and reviewed rules are proof-bearing; source regions and proposed claims are evidence only.',
    }),
    tags: canonicalTags(['document-intelligence', 'real-pdf', 'remembero']),
  };
}

function documentItem(fixture: DocumentShowcaseFixture): DocumentMemorgItem {
  const accepted = fixture.claims.filter((claim) => claim.kind === 'accepted').length;
  const proposed = fixture.claims.length - accepted;
  return {
    key: documentKey(fixture),
    item_type: 'document',
    parent_key: DOCUMENT_MEMORG_ROOT_KEY,
    content: [
      fixture.title,
      `${fixture.kindLabel} published by ${fixture.source.publisher}.`,
      `Selected PDF pages: ${fixture.source.selectedPageNumbers.join(', ')} of ${fixture.source.pdfPageCount}.`,
      `${accepted} accepted claims, ${proposed} proposed claims, ${fixture.rules.length} reviewed rules, and ${fixture.questions.length} guided questions.`,
    ].join('\n'),
    metadata: authorityMetadata('source', {
      remembero_document_id: fixture.id,
      remembero_namespace: `documents-${fixture.id}`,
      fixture_version: fixture.fixtureVersion,
      parser_mode: fixture.parserMode,
      parser_label: fixture.parserLabel,
      publisher: fixture.source.publisher,
      source_url: fixture.source.url,
      retrieved_at: fixture.source.retrievedAt,
      pdf_sha256: fixture.source.sha256,
      pdf_page_count: fixture.source.pdfPageCount,
      selected_page_numbers: [...fixture.source.selectedPageNumbers],
      rights_note: fixture.source.rightsNote,
      page_images: fixture.pages.map((page) => ({
        page_number: page.pageNumber,
        image_url: page.imageUrl,
        image_sha256: page.imageSha256,
      })),
    }),
    tags: canonicalTags([
      'document',
      'document-intelligence',
      fixture.id,
      fixture.kindLabel.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, ''),
      'real-pdf',
      'remembero',
    ]),
  };
}

function regionItem(
  fixture: DocumentShowcaseFixture,
  region: DocumentShowcaseFixtureRegion
): DocumentMemorgItem {
  const page = fixture.pages.find((entry) => entry.id === region.pageId);
  if (page === undefined) throw new Error(`Memorg region ${region.id} has no source page`);
  return {
    key: regionKey(fixture, region),
    item_type: 'custom',
    parent_key: documentKey(fixture),
    content: region.text,
    metadata: authorityMetadata('evidence', {
      remembero_document_id: fixture.id,
      region_id: region.id,
      region_label: region.label,
      region_kind: region.kind,
      reading_order: region.order,
      page_number: region.pageNumber,
      bounding_box: { ...region.bbox },
      page_image_sha256: page.imageSha256,
      pdf_sha256: fixture.source.sha256,
    }),
    tags: canonicalTags([
      'document-intelligence',
      'evidence',
      fixture.id,
      `page-${region.pageNumber}`,
      'remembero',
      'source-region',
    ]),
  };
}

function claimItem(
  fixture: DocumentShowcaseFixture,
  claim: DocumentShowcaseFixtureClaim
): DocumentMemorgItem {
  return {
    key: claimKey(fixture, claim),
    item_type: 'custom',
    parent_key: regionKey(fixture, sourceRegion(fixture, claim.regionId)),
    content: `${claim.clause}\n${claim.summary}\nSource: ${claim.sourceText}`,
    metadata: authorityMetadata(claim.kind === 'accepted' ? 'accepted' : 'proposed_only', {
      remembero_document_id: fixture.id,
      claim_id: claim.id,
      clause: claim.clause,
      review_state: claim.kind,
      page_number: claim.pageNumber,
      region_id: claim.regionId,
      pdf_sha256: fixture.source.sha256,
    }),
    tags: canonicalTags([
      claim.kind,
      'claim',
      'document-intelligence',
      fixture.id,
      'remembero',
      claim.kind === 'accepted' ? 'proof-bearing' : 'review-required',
    ]),
  };
}

function ruleItem(
  fixture: DocumentShowcaseFixture,
  rule: DocumentShowcaseFixtureRule
): DocumentMemorgItem {
  return {
    key: ruleKey(fixture, rule),
    item_type: 'custom',
    parent_key: documentKey(fixture),
    content: `${rule.clause}\n${rule.summary}`,
    metadata: authorityMetadata('reviewed_rule', {
      remembero_document_id: fixture.id,
      rule_id: rule.id,
      clause: rule.clause,
      source_text: rule.sourceText,
      proof_bearing: true,
    }),
    tags: canonicalTags([
      'document-intelligence',
      fixture.id,
      'proof-bearing',
      'remembero',
      'reviewed-rule',
    ]),
  };
}

function questionItem(
  fixture: DocumentShowcaseFixture,
  question: DocumentShowcaseFixtureQuestion
): DocumentMemorgItem {
  return {
    key: questionKey(fixture, question),
    item_type: 'custom',
    parent_key: documentKey(fixture),
    content: [
      `Question: ${question.question}`,
      `Canonical query: ${question.query}`,
      `Expected status: ${question.expectedStatus}`,
      question.supportedAnswer === undefined
        ? `Abstention: ${question.unsupportedAnswer}`
        : `Supported answer: ${question.supportedAnswer}`,
    ].join('\n'),
    metadata: authorityMetadata('evaluation_contract', {
      remembero_document_id: fixture.id,
      question_id: question.id,
      query: question.query,
      expected_status: question.expectedStatus,
      related_evidence_ids: [...question.relatedEvidenceIds],
      expected_source_region_ids: [...(question.expectedSourceRegionIds ?? [])],
      expected_accepted_claim_ids: [...(question.expectedAcceptedClaimIds ?? [])],
      expected_rule_ids: [...(question.expectedRuleIds ?? [])],
    }),
    tags: canonicalTags([
      'document-intelligence',
      'evaluation-contract',
      fixture.id,
      question.expectedStatus,
      'question',
      'remembero',
    ]),
  };
}

function exportBody(items: DocumentMemorgItem[]) {
  return {
    format: DOCUMENT_MEMORG_FORMAT,
    version: DOCUMENT_MEMORG_VERSION,
    target: {
      package: 'memorg',
      version: DOCUMENT_MEMORG_TARGET_VERSION,
      method: 'MemorgSystem.create_memory_item',
    },
    items,
  } as const;
}

export function createDocumentMemorgExport(
  fixtures: readonly DocumentShowcaseFixture[] = DOCUMENT_SHOWCASE_FIXTURES
): DocumentMemorgExport {
  if (fixtures.length === 0) throw new Error('Memorg export requires at least one document');
  const items: DocumentMemorgItem[] = [rootItem(fixtures)];
  for (const fixture of fixtures) {
    items.push(documentItem(fixture));
    for (const page of fixture.pages) {
      for (const region of [...page.regions].sort((left, right) => left.order - right.order)) {
        items.push(regionItem(fixture, region));
      }
    }
    items.push(...fixture.claims.map((claim) => claimItem(fixture, claim)));
    items.push(...fixture.rules.map((rule) => ruleItem(fixture, rule)));
    items.push(...fixture.questions.map((question) => questionItem(fixture, question)));
  }
  const body = exportBody(items);
  const artifact: DocumentMemorgExport = {
    ...body,
    sha256: digest(JSON.stringify(body)),
  };
  verifyDocumentMemorgExport(artifact);
  return artifact;
}

function normalizedItem(value: unknown, index: number): DocumentMemorgItem {
  if (typeof value !== 'object' || value === null || Array.isArray(value)) {
    throw new Error(`Memorg item ${index} must be an object`);
  }
  const record = value as Record<string, unknown>;
  exactKeys(record, ['key', 'item_type', 'parent_key', 'content', 'metadata', 'tags'], `Memorg item ${index}`);
  if (typeof record.key !== 'string' || !KEY_RE.test(record.key)) {
    throw new Error(`Memorg item ${index} has an invalid key`);
  }
  if (!ITEM_TYPES.has(record.item_type as DocumentMemorgItemType)) {
    throw new Error(`Memorg item ${record.key} has an invalid item_type`);
  }
  if (record.parent_key !== null && (typeof record.parent_key !== 'string' || !KEY_RE.test(record.parent_key))) {
    throw new Error(`Memorg item ${record.key} has an invalid parent_key`);
  }
  if (typeof record.content !== 'string' || record.content.trim() === '') {
    throw new Error(`Memorg item ${record.key} has empty content`);
  }
  if (typeof record.metadata !== 'object' || record.metadata === null || Array.isArray(record.metadata)) {
    throw new Error(`Memorg item ${record.key} has invalid metadata`);
  }
  const metadata = record.metadata as Record<string, unknown>;
  if (!AUTHORITIES.has(metadata.authority as DocumentMemorgAuthority)) {
    throw new Error(`Memorg item ${record.key} has invalid authority metadata`);
  }
  if (!Array.isArray(record.tags) || record.tags.length === 0 || record.tags.some((tag) => typeof tag !== 'string' || !KEY_RE.test(tag))) {
    throw new Error(`Memorg item ${record.key} has invalid tags`);
  }
  const tags = record.tags as string[];
  if (JSON.stringify(tags) !== JSON.stringify(canonicalTags(tags))) {
    throw new Error(`Memorg item ${record.key} tags must be unique and sorted`);
  }
  return {
    key: record.key,
    item_type: record.item_type as DocumentMemorgItemType,
    parent_key: record.parent_key as string | null,
    content: record.content,
    metadata,
    tags,
  };
}

export function verifyDocumentMemorgExport(
  value: DocumentMemorgExport | string
): DocumentMemorgVerification {
  let parsed: unknown = value;
  if (typeof value === 'string') {
    if (Buffer.byteLength(value, 'utf8') > MAX_OUTPUT_BYTES) {
      throw new Error('Memorg export exceeds the output byte limit');
    }
    try {
      parsed = JSON.parse(value);
    } catch {
      throw new Error('Memorg export is not valid JSON');
    }
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    throw new Error('Memorg export must be an object');
  }
  const record = parsed as Record<string, unknown>;
  exactKeys(record, ['format', 'version', 'target', 'items', 'sha256'], 'Memorg export');
  if (record.format !== DOCUMENT_MEMORG_FORMAT || record.version !== DOCUMENT_MEMORG_VERSION || typeof record.sha256 !== 'string' || !SHA256_RE.test(record.sha256)) {
    throw new Error('Memorg export identity is invalid');
  }
  if (typeof record.target !== 'object' || record.target === null || Array.isArray(record.target)) {
    throw new Error('Memorg export target must be an object');
  }
  const target = record.target as Record<string, unknown>;
  exactKeys(target, ['package', 'version', 'method'], 'Memorg export target');
  if (target.package !== 'memorg' || target.version !== DOCUMENT_MEMORG_TARGET_VERSION || target.method !== 'MemorgSystem.create_memory_item') {
    throw new Error('Memorg export target is unsupported');
  }
  if (!Array.isArray(record.items) || record.items.length === 0 || record.items.length > DOCUMENT_MEMORG_MAX_ITEMS) {
    throw new Error(`Memorg export items must contain 1 to ${DOCUMENT_MEMORG_MAX_ITEMS} entries`);
  }
  const items = record.items.map(normalizedItem);
  const keys = new Set<string>();
  for (const [index, item] of items.entries()) {
    if (keys.has(item.key)) throw new Error(`Memorg export contains duplicate key ${item.key}`);
    if (index === 0) {
      if (item.key !== DOCUMENT_MEMORG_ROOT_KEY || item.parent_key !== null) {
        throw new Error('Memorg export must begin with the root memory item');
      }
    } else if (item.parent_key === null || !keys.has(item.parent_key)) {
      throw new Error(`Memorg item ${item.key} must follow its parent`);
    }
    keys.add(item.key);
  }
  const body = exportBody(items);
  const expected = digest(JSON.stringify(body));
  if (record.sha256 !== expected) throw new Error('Memorg export failed SHA-256 validation');
  const canonical = JSON.stringify({ ...body, sha256: expected });
  assertBoundedOutput(canonical, 'Memorg export', MAX_OUTPUT_BYTES);
  const authorityCount = (authority: DocumentMemorgAuthority) =>
    items.filter((item) => item.metadata.authority === authority).length;
  return {
    valid: true,
    sha256: expected,
    bytes: Buffer.byteLength(canonical, 'utf8'),
    itemCount: items.length,
    documentCount: items.filter((item) => item.item_type === 'document').length,
    acceptedClaimCount: authorityCount('accepted'),
    proposedClaimCount: authorityCount('proposed_only'),
    reviewedRuleCount: authorityCount('reviewed_rule'),
    questionCount: authorityCount('evaluation_contract'),
  };
}

export function serializeDocumentMemorgExport(exported: DocumentMemorgExport): string {
  verifyDocumentMemorgExport(exported);
  const text = JSON.stringify(exported, null, 2);
  assertBoundedOutput(text, 'Memorg export', MAX_OUTPUT_BYTES);
  return text;
}
