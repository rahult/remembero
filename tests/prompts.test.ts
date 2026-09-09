import { describe, expect, it } from 'vitest';
import {
  extractionSystemPrompt,
  transcriptExtractionSystemPrompt,
} from '../src/llm/prompts.js';

describe('extraction prompts state the subject conventions the benchmark relies on', () => {
  for (const [name, prompt] of [
    [
      'text',
      extractionSystemPrompt('% (no memories yet)', 'accepted', 'rahul'),
    ],
    [
      'transcript',
      transcriptExtractionSystemPrompt('% (no memories yet)', 'rahul'),
    ],
  ] as const) {
    it(`${name}: "we"/"our"/"my team" is the group, not the speaker`, () => {
      expect(prompt).toMatch(/deploy_day\(team, friday\)/);
      expect(prompt).toMatch(/"?[Ww]e"?.*not rahul|not the speaker/);
    });
    it(`${name}: an unnamed subject is the generic noun the text describes, never a schema sample`, () => {
      expect(prompt).toMatch(/budget_dollars\(project, 250000\)/);
      expect(prompt).toMatch(/[Nn]ever (borrow|copy) a subject/);
    });
  }
});
