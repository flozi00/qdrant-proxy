/* ============================================================================
 * Utility functions
 * ============================================================================ */

/** Human-readable relative time. */
export function timeAgo(date: Date): string {
  const seconds = Math.floor((Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

/**
 * Pure-JS SHA-1 (no crypto.subtle — works over HTTP).
 * Returns raw 20-byte Uint8Array.
 */
function sha1(data: Uint8Array): Uint8Array {
  const pad = new Uint8Array(((data.length + 72) & ~63));
  pad.set(data);
  pad[data.length] = 0x80;
  const bits = data.length * 8;
  const dv = new DataView(pad.buffer);
  dv.setUint32(pad.length - 4, bits >>> 0);
  dv.setUint32(pad.length - 8, (bits / 0x100000000) >>> 0);

  let h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE, h3 = 0x10325476, h4 = 0xC3D2E1F0;
  const w = new Uint32Array(80);
  for (let off = 0; off < pad.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = dv.getUint32(off + i * 4);
    for (let i = 16; i < 80; i++) { const t = w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16]; w[i] = (t << 1) | (t >>> 31); }
    let a = h0, b = h1, c = h2, d = h3, e = h4;
    for (let i = 0; i < 80; i++) {
      const f = i < 20 ? ((b & c) | (~b & d)) + 0x5A827999
                : i < 40 ? (b ^ c ^ d) + 0x6ED9EBA1
                : i < 60 ? ((b & c) | (b & d) | (c & d)) + 0x8F1BBCDC
                : (b ^ c ^ d) + 0xCA62C1D6;
      const t = (((a << 5) | (a >>> 27)) + f + e + w[i]) >>> 0;
      e = d; d = c; c = ((b << 30) | (b >>> 2)) >>> 0; b = a; a = t;
    }
    h0 = (h0 + a) >>> 0; h1 = (h1 + b) >>> 0; h2 = (h2 + c) >>> 0; h3 = (h3 + d) >>> 0; h4 = (h4 + e) >>> 0;
  }
  const out = new Uint8Array(20);
  const odv = new DataView(out.buffer);
  odv.setUint32(0, h0); odv.setUint32(4, h1); odv.setUint32(8, h2); odv.setUint32(12, h3); odv.setUint32(16, h4);
  return out;
}

/** Generate a UUID v5 from a URL (DNS namespace). Works over HTTP (no crypto.subtle). */
export function urlToDocId(url: string): string {
  const DNS_NS = '6ba7b810-9dad-11d1-80b4-00c04fd430c8';

  const nsHex = DNS_NS.replace(/-/g, '');
  const nsBytes = new Uint8Array(16);
  for (let i = 0; i < 16; i++) nsBytes[i] = parseInt(nsHex.substr(i * 2, 2), 16);

  const nameBytes = new TextEncoder().encode(url);
  const data = new Uint8Array(nsBytes.length + nameBytes.length);
  data.set(nsBytes, 0);
  data.set(nameBytes, nsBytes.length);

  const hash = sha1(data).slice(0, 16);

  // Set version (5) and variant (RFC 4122)
  hash[6] = (hash[6] & 0x0f) | 0x50;
  hash[8] = (hash[8] & 0x3f) | 0x80;

  const hex = Array.from(hash)
    .map((b) => b.toString(16).padStart(2, '0'))
    .join('');
  return [hex.substring(0, 8), hex.substring(8, 12), hex.substring(12, 16), hex.substring(16, 20), hex.substring(20, 32)].join('-');
}

/** Truncate text to maxLen characters with ellipsis. */
export function truncate(text: string, maxLen: number): string {
  if (text.length <= maxLen) return text;
  return text.substring(0, maxLen) + '...';
}
