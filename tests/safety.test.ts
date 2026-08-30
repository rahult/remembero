import { describe, expect, it } from 'vitest';
import {
  REDACTED_SOURCE,
  containsSensitiveText,
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

  it('preserves valid Unicode and replaces only lone surrogate code units', () => {
    expect(normalizeUnicodeScalarText('hello 👋 world')).toBe('hello 👋 world');
    expect(normalizeUnicodeScalarText(`left\uD800right`)).toBe('left�right');
    expect(normalizeUnicodeScalarText(`left\uDC00right`)).toBe('left�right');
  });
});
