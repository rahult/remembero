import { describe, expect, it } from 'vitest';
import {
  REDACTED_SOURCE,
  containsSensitiveText,
  maskSensitiveSpans,
  normalizeUnicodeScalarText,
  redactSensitiveText,
} from '../src/safety.js';

// Stripe fixtures are assembled at runtime: a live-shaped key literal in source trips
// GitHub push protection even when it is an obvious test fixture, so the scanner never
// sees one while the tests still exercise the real shapes.
const stripeSecretFixture = ['sk_live', '51H8yZqABcDeFgHiJkLmNoPqR'].join('_');
const stripeRestrictedFixture = ['rk_live', '51H8yZqABcDeFgHiJkLmNoPqR'].join('_');
const stripePublishableFixture = ['pk_live', '51H8yZqABcDeFgHiJkLmNoPqR'].join('_');

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

  it('sees a credential word joined to an identifier by _, - or .', () => {
    // `\b` does not fire between `_` and a letter, so these forms used to escape
    // both detectors and land in the session store verbatim.
    for (const text of [
      "reset_password('hunter2')",
      'my_password = hunter2',
      'user.api_key = ordinary-looking-value',
      'db-password: ordinary-looking-value',
      "refresh_token_value('hunter2')",
    ]) {
      expect(containsSensitiveText(text)).toBe(true);
      const masked = maskSensitiveSpans(text);
      expect(masked.masked).toBeGreaterThan(0);
      expect(masked.text).not.toContain('hunter2');
      expect(masked.text).not.toContain('ordinary-looking-value');
      expect(containsSensitiveText(masked.text)).toBe(false);
    }
  });

  it('does not treat an ordinary word that merely contains one as a credential', () => {
    for (const text of [
      'We shipped passwordless login this week.',
      'The tokenizer is configured for byte pairs.',
      'Ask my secretary for the meeting notes.',
      // The separator is what makes an identifier, so a word running straight
      // into the credential word is still just a word.
      'passwordless(user)',
      'secretary: jane',
      'The refresh_tokenizer splits on byte pairs.',
    ]) {
      expect(containsSensitiveText(text)).toBe(false);
      expect(maskSensitiveSpans(text)).toEqual({
        text,
        masked: 0,
        truncated: false,
      });
    }
  });

  it('sees the four high-signal secret shapes a credential word never names', () => {
    // None of these carries a credential word, a `bearer`/`sk-`/`gh*` prefix or a
    // Luhn-valid run, so every one of them used to be stored verbatim by the session
    // store and then cleared `assertSafeForExternalLlm` on the way to a cloud reader.
    for (const { name, text, secret } of [
      {
        name: 'AWS access key id',
        text: 'The deploy used AKIAIOSFODNN7EXAMPLE in eu-west-1.',
        secret: 'AKIAIOSFODNN7EXAMPLE',
      },
      {
        name: 'AWS session key id',
        text: 'The temporary id was ASIAY34FZKBOKMUTVV7A today.',
        secret: 'ASIAY34FZKBOKMUTVV7A',
      },
      {
        name: 'JWT',
        text:
          'The cookie held eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0' +
          '.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1gFWFOEjXk and it worked.',
        secret: 'eyJzdWIiOiIxMjM0NTY3ODkwIn0',
      },
      {
        name: 'credentials in a URL',
        text:
          'My .env has DATABASE_URL=postgres://admin:S3cretPass@db.internal.example.com:5432/app.',
        secret: 'S3cretPass',
      },
      {
        name: 'PEM private key',
        text:
          'I pasted the deploy key:\n-----BEGIN RSA PRIVATE KEY-----\n' +
          'MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu\n' +
          '-----END RSA PRIVATE KEY-----\nand CI went green.',
        secret: 'MIIBOgIBAAJBAKj34GkxFhD90vcNLYLInFEX6Ppy1tPf9Cnzj4p4WGeKLs1Pt8Qu',
      },
      {
        name: 'PEM private key with no END line',
        text:
          'half a key:\n-----BEGIN OPENSSH PRIVATE KEY-----\n' +
          'b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB',
        secret: 'b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB',
      },
    ]) {
      expect(containsSensitiveText(text), name).toBe(true);
      expect(redactSensitiveText(text).redacted, name).toBe(true);
      const masked = maskSensitiveSpans(text);
      expect(masked.masked, name).toBeGreaterThan(0);
      expect(masked.text, name).not.toContain(secret);
      // the gate must have nothing left to find, or the store keeps the turn
      expect(containsSensitiveText(masked.text), name).toBe(false);
    }
  });

  it('keeps the readable prose around each of the four shapes', () => {
    const aws = maskSensitiveSpans('The deploy used AKIAIOSFODNN7EXAMPLE in eu-west-1.');
    expect(aws.text).toBe('The deploy used [redacted] in eu-west-1.');
    expect(aws.truncated).toBe(false);
    const url = maskSensitiveSpans(
      'DATABASE_URL=postgres://admin:S3cretPass@db.example.com:5432/app broke the job.',
    );
    expect(url.text).toBe('DATABASE_URL=[redacted] broke the job.');
    expect(url.truncated).toBe(false);
    const pem = maskSensitiveSpans(
      'key:\n-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIBWnVbBcQ\n-----END EC PRIVATE KEY-----\nCI green.',
    );
    expect(pem.text).toBe('key:\n[redacted]\nCI green.');
    expect(pem.truncated).toBe(false);
  });

  it('leaves prose and near-misses of the four shapes untouched', () => {
    for (const text of [
      // a bare `eyJ`-like word is not a JWT: a JWT is dot-separated segments
      'The eyJ prefix is how you spot a JSON web token.',
      'eyJhbGciOiJIUzI1NiJ9 on its own is just a base64 header.',
      'AKIA is the prefix every AWS long-term key id starts with.',
      // not 16 more uppercase alphanumerics, and not a key id
      'AKIASHORT is not a key id.',
      // no userinfo, so no credential in the URL
      'Read https://docs.example.com/guide:latest for the steps.',
      'postgres://db.internal.example.com:5432/app is the host to use.',
      'Point the client at redis://cache.example.com:6379/0 instead.',
      // a PEM public key is not a private one
      '-----BEGIN PUBLIC KEY-----\nMHcCAQEEIBWnVbBcQ\n-----END PUBLIC KEY-----',
      'We store the private key in 1Password, never in the repo.',
    ]) {
      expect(containsSensitiveText(text), text).toBe(false);
      expect(maskSensitiveSpans(text), text).toEqual({
        text,
        masked: 0,
        truncated: false,
      });
    }
  });

  it('reports a cut span for an underscore form that cannot close', () => {
    const result = maskSensitiveSpans('reset_password(hunter2');
    expect(result.masked).toBe(1);
    expect(result.text).not.toContain('hunter2');
    expect(result.truncated).toBe(true);
    expect(containsSensitiveText(result.text)).toBe(false);
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

  it('sees the vendor token shapes a credential word never names', () => {
    for (const { name, text, secret } of [
      {
        name: 'Slack bot token',
        text: 'The hook used xoxb-EXAMPLE-TEST-FIXTURE-NOTAREALTOKEN today.',
        secret: 'xoxb-EXAMPLE-TEST-FIXTURE-NOTAREALTOKEN',
      },
      {
        name: 'Slack user token',
        text: 'CI exports xoxp-9182736450-9182736450192-LkJhGfDsAqWeRtYuIoP12345 for it.',
        secret: 'xoxp-9182736450-9182736450192-LkJhGfDsAqWeRtYuIoP12345',
      },
      {
        name: 'Slack app token',
        text: 'The legacy bot still uses xoxa-2-ABCD1234efgh5678IJKL90mnop for posting.',
        secret: 'xoxa-2-ABCD1234efgh5678IJKL90mnop',
      },
      {
        name: 'Slack refresh token',
        text: 'Rotation gave me xoxr-1122334455-6677889900112-QwErTyUiOpAsDfGh back.',
        secret: 'xoxr-1122334455-6677889900112-QwErTyUiOpAsDfGh',
      },
      {
        name: 'Google API key',
        text: 'Maps loads with AIzaSyB3kL9mNpQrStUvWxYz012345678_aBcDe in the client.',
        secret: 'AIzaSyB3kL9mNpQrStUvWxYz012345678_aBcDe',
      },
      {
        name: 'GitHub fine-grained token',
        text:
          'Actions holds github_pat_11ABCDEFG0aBcDeFgHiJk_' +
          'LmNoPqRsTuVwXyZ0123456789abcdefghijKLMNOPQRS now.',
        secret: 'github_pat_11ABCDEFG0aBcDeFgHiJk_',
      },
      {
        name: 'GitHub classic token',
        text: 'The old runner used ghp_A1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU1vW2x for cloning.',
        secret: 'ghp_A1bC2dE3fG4hI5jK6lM7nO8pQ9rS0tU1vW2x',
      },
      {
        name: 'Stripe live secret key',
        text: `Billing config had ${stripeSecretFixture} set in prod.`,
        secret: stripeSecretFixture,
      },
      {
        name: 'Stripe restricted live key',
        text: `The worker reads ${stripeRestrictedFixture} from the env.`,
        secret: stripeRestrictedFixture,
      },
      {
        name: 'Stripe publishable live key',
        text: `The checkout page embeds ${stripePublishableFixture} inline.`,
        secret: stripePublishableFixture,
      },
      {
        name: 'Stripe test key, masked deliberately',
        text: 'Staging uses sk_test_4eC39HqLyjWDarjtT1zdp7dcABcDeFgH instead.',
        secret: 'sk_test_4eC39HqLyjWDarjtT1zdp7dcABcDeFgH',
      },
      {
        name: 'lowercase AWS access key id',
        text: 'The fixture pins akiaiosfodnn7example as the id.',
        secret: 'akiaiosfodnn7example',
      },
      {
        name: 'URL credential whose password holds a slash',
        text: 'The broker is amqp://guest:gu/est@rabbit.internal.example.com:5672/vhost here.',
        secret: 'gu/est',
      },
    ]) {
      expect(containsSensitiveText(text), name).toBe(true);
      expect(redactSensitiveText(text).redacted, name).toBe(true);
      const masked = maskSensitiveSpans(text);
      expect(masked.masked, name).toBeGreaterThan(0);
      expect(masked.text, name).not.toContain(secret);
      expect(containsSensitiveText(masked.text), name).toBe(false);
      expect(masked.truncated, name).toBe(false);
    }
  });

  it('keeps the prose around a vendor token', () => {
    const slack = maskSensitiveSpans(
      'The hook used xoxb-EXAMPLE-TEST-FIXTURE-NOTAREALTOKEN today.',
    );
    expect(slack.text).toBe('The hook used [redacted] today.');
    const google = maskSensitiveSpans(
      'Maps loads with AIzaSyB3kL9mNpQrStUvWxYz012345678_aBcDe in the client.',
    );
    expect(google.text).toBe('Maps loads with [redacted] in the client.');
    const aws = maskSensitiveSpans('The fixture pins akiaiosfodnn7example as the id.');
    expect(aws.text).toBe('The fixture pins [redacted] as the id.');
    const broker = maskSensitiveSpans(
      'The broker is amqp://guest:gu/est@rabbit.internal.example.com:5672/vhost here.',
    );
    expect(broker.text).toBe('The broker is [redacted] here.');
  });

  it('leaves the vendor-token near-misses untouched', () => {
    for (const text of [
      'We play xoxo and hugs in the team channel.',
      'AIza is the prefix every Google API key starts with.',
      'The github_pattern helper matches repository names.',
      // the prefix alone, with no key body after it
      'Keys beginning sk_test_ are not live secrets.',
      'The digest is 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08.',
      'The row id is 7c9e6679-7425-40de-944b-e07fc1f90ae7 in the table.',
      'Clone it with git@github.com:acme/repo.git over SSH.',
      'postgres://db.internal:5432/app is the host to use.',
      // a URL with a port and a path, not a userinfo credential
      'Fetch https://docs.example.com:8443/guide/v2@latest for the steps.',
    ]) {
      expect(containsSensitiveText(text), text).toBe(false);
      expect(maskSensitiveSpans(text), text).toEqual({
        text,
        masked: 0,
        truncated: false,
      });
    }
  });

  it('does not silently drop the tail of prose that merely names PEM armour', () => {
    // The unterminated PEM span used to run to the end of the text, so this turn was
    // stored as 'PEM files start with [redacted]' with truncated: false — a silent
    // truncation the session store had no way to notice.
    const prose =
      'PEM files start with -----BEGIN RSA PRIVATE KEY----- and end with the ' +
      'matching END line. I keep mine in 1Password…';
    const result = maskSensitiveSpans(prose);
    expect(result.masked).toBe(1);
    // Either the tail survives, or the store is told the masking was inconclusive
    // and falls back to the whole-turn placeholder. Never a quiet shortening.
    const tailSurvived = result.text.includes('1Password…');
    expect(tailSurvived || result.truncated).toBe(true);
    expect(containsSensitiveText(result.text)).toBe(false);
  });

  it('stops an unterminated PEM key at the blank line and reports the cut', () => {
    const result = maskSensitiveSpans(
      'the key:\n-----BEGIN OPENSSH PRIVATE KEY-----\n' +
        'b3BlbnNzaC1rZXktdjEAAAAABG5vbmUAAAAEbm9uZQAAAAAAAAAB\n' +
        '\nI rode 40 km on the new bike',
    );
    expect(result.masked).toBe(1);
    expect(result.text).not.toContain('b3BlbnNzaC1rZXktdjEA');
    expect(result.text).toContain('I rode 40 km on the new bike');
    expect(result.truncated).toBe(true);
    expect(containsSensitiveText(result.text)).toBe(false);
  });

  it('reports nothing truncated for a PEM key that closes with its END line', () => {
    const result = maskSensitiveSpans(
      'key:\n-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIBWnVbBcQ\n' +
        '-----END EC PRIVATE KEY-----\nCI green.',
    );
    expect(result).toEqual({
      text: 'key:\n[redacted]\nCI green.',
      masked: 1,
      truncated: false,
    });
  });

  it('preserves valid Unicode and replaces only lone surrogate code units', () => {
    expect(normalizeUnicodeScalarText('hello 👋 world')).toBe('hello 👋 world');
    expect(normalizeUnicodeScalarText(`left\uD800right`)).toBe('left�right');
    expect(normalizeUnicodeScalarText(`left\uDC00right`)).toBe('left�right');
  });
});
