import { describe, expect, it } from 'vitest';
import {
  REDACTED_SOURCE,
  containsSensitiveText,
  maskSensitiveSpans,
  normalizeUnicodeScalarText,
  redactSensitiveText,
} from '../src/safety.js';

describe('sensitive text detection', () => {
  it('does not erase ordinary discussion of credential-adjacent topics', () => {
    for (const text of [
      'Compare credit card rewards for travel.',
      'Explain password protection to a new user.',
      'The secret garden is my favourite novel.',
      'Show the account number field in the settings form.',
      'Rotate access token permissions every quarter.',
    ]) {
      expect(containsSensitiveText(text)).toBe(false);
      expect(redactSensitiveText(text)).toEqual({ text, redacted: false });
    }
  });

  it('still blocks assigned credentials and raw token or card formats', () => {
    for (const text of [
      'My password is correct-horse-battery-staple.',
      'My password hunter2.',
      'API key: ordinary-looking-value',
      '"password": "ordinary-looking-value"',
      "password(user, 'ordinary-looking-value').",
      'refresh_token=ordinary-looking-value',
      'Authorization: Bearer abcdefgh12345678',
      'Use sk-supersecretvalue for this request.',
      'The card number is 4111 1111 1111 1111.',
    ]) {
      expect(containsSensitiveText(text)).toBe(true);
      expect(redactSensitiveText(text)).toEqual({
        text: REDACTED_SOURCE,
        redacted: true,
      });
    }
  });

  it('does not block long digit runs that fail the card checksum', () => {
    for (const text of [
      'The deploy finished at 1756518000000.',
      'Build 1717171717171 is green.',
      'Session stamp 20260830115959 was recorded.',
      'The card-like value 4111 1111 1111 1112 is not a valid number.',
    ]) {
      expect(containsSensitiveText(text)).toBe(false);
    }
  });

  it('keeps the readable half of a turn and hides the secret', () => {
    const result = maskSensitiveSpans(
      'I set my api key = sk-abc123456789 for the deploy'
    );
    expect(result.masked).toBe(1);
    expect(result.text).toBe('I set my [redacted] for the deploy');
    expect(containsSensitiveText(result.text)).toBe(false);
  });

  it('leaves ordinary text alone', () => {
    expect(maskSensitiveSpans('I rode 40 km on the new bike')).toEqual({
      text: 'I rode 40 km on the new bike',
      masked: 0,
      truncated: false,
    });
  });

  it('masks a Luhn-valid card run but not a long digit run', () => {
    expect(
      maskSensitiveSpans('The card number is 4111 1111 1111 1111.')
    ).toEqual({
      text: 'The card number is [redacted].',
      masked: 1,
      truncated: false,
    });
    const stamp = maskSensitiveSpans('The deploy finished at 1756518000000.');
    expect(stamp.masked).toBe(0);
  });

  it('masks the whole call when a credential is passed as an argument', () => {
    const result = maskSensitiveSpans(
      "password(user, 'ordinary-looking-value')."
    );
    expect(result.text).toBe('[redacted].');
    expect(result.masked).toBe(1);
    expect(containsSensitiveText(result.text)).toBe(false);
  });

  it('masks a call that never closes, and one that nests', () => {
    const unclosed = maskSensitiveSpans('password(hunter2');
    expect(unclosed.text).not.toContain('hunter2');
    expect(unclosed.masked).toBe(1);
    expect(containsSensitiveText(unclosed.text)).toBe(false);

    const nested = maskSensitiveSpans("password(secret(x), 'val')");
    expect(nested.text).not.toContain('val');
    expect(nested.masked).toBe(1);
    expect(containsSensitiveText(nested.text)).toBe(false);
  });

  it('masks a call that spans several lines', () => {
    const result = maskSensitiveSpans("password(\n  'hunter2'\n)");
    expect(result.text).toBe('[redacted]');
    expect(result.masked).toBe(1);
    expect(result.text).not.toContain('hunter2');
    expect(containsSensitiveText(result.text)).toBe(false);
  });

  it('stops an unclosed call at the blank line and keeps the prose after it', () => {
    const result = maskSensitiveSpans(
      "password(\n  'hunter2'\n\nI rode 40 km on the new bike"
    );
    expect(result.text).toBe('[redacted]\n\nI rode 40 km on the new bike');
    expect(result.masked).toBe(1);
    expect(result.text).not.toContain('hunter2');
    // The span stopped early, so a caller that cannot inspect the text — the
    // session store — is told the masking was not conclusive.
    expect(result.truncated).toBe(true);
  });

  it('caps how far an unclosed call can swallow, and reports the cut', () => {
    const result = maskSensitiveSpans(`password(${'a'.repeat(600)}`);
    expect(result.masked).toBe(1);
    expect(result.text).toBe(`[redacted]${'a'.repeat(600 - 512)}`);
    // The residue is context-free, so no pattern matches it: the only honest
    // signal that something was left behind is this flag.
    expect(result.truncated).toBe(true);
  });

  it('reports a long balanced call cut off at the cap', () => {
    const args = Array.from(
      { length: 26 },
      (_unused, index) => `  arg${index} = ${'v'.repeat(30)},`
    ).join('\n');
    const block = `password(\n${args}\n  value = 'hunter2'\n)`;
    expect(block.length).toBeGreaterThan(900);
    const result = maskSensitiveSpans(block);
    expect(result.masked).toBe(1);
    expect(result.truncated).toBe(true);
    // The word that made it a credential is inside the masked span, so the tail
    // that survives matches nothing at all.
    expect(containsSensitiveText(result.text)).toBe(false);
    expect(result.text).toContain('hunter2');
  });

  it('reports a blank line between the paren and the arguments', () => {
    const result = maskSensitiveSpans("password(\n\n  'hunter2'\n)");
    expect(result.masked).toBe(1);
    expect(result.truncated).toBe(true);
  });

  it('reports nothing truncated when a call closes on its own', () => {
    for (const text of [
      "password(user, 'ordinary-looking-value').",
      "password(secret(x), 'val')",
      "password(\n  'hunter2'\n)",
    ]) {
      expect(maskSensitiveSpans(text).truncated).toBe(false);
    }
    expect(maskSensitiveSpans('I rode 40 km on the new bike').truncated).toBe(
      false
    );
  });

  it('masks an assigned credential and leaves nothing sensitive behind', () => {
    const result = maskSensitiveSpans('My password is correct-horse.');
    expect(result.masked).toBe(1);
    expect(containsSensitiveText(result.text)).toBe(false);
  });

  it('preserves valid Unicode and replaces only lone surrogate code units', () => {
    expect(normalizeUnicodeScalarText('hello 👋 world')).toBe('hello 👋 world');
    expect(normalizeUnicodeScalarText(`left\uD800right`)).toBe('left�right');
    expect(normalizeUnicodeScalarText(`left\uDC00right`)).toBe('left�right');
  });
});
