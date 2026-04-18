const FIELD_PATTERN = /([+-]?)(allintext|intext|allinurl|inurl|allintitle|intitle|site|filetype|ext|link|allinanchor|inanchor|allinpostauthor|inpostauthor|related|cache|before|after|numrange):("[^"]*"|\([^)]*\)|[^\s()]+)/gi;

const QUOTE_TRANSLATIONS: Record<string, string> = {
  '“': '"',
  '”': '"',
  '„': '"',
  '‟': '"',
  '’': '\'',
  '‘': '\'',
};

function normalizeQuotes(value: string): string {
  return Array.from(value).map((char) => QUOTE_TRANSLATIONS[char] || char).join('');
}

function dedupe(values: string[]): string[] {
  const seen = new Set<string>();
  return values.filter((value) => {
    const normalized = value.trim().toLowerCase();
    if (!normalized || seen.has(normalized)) return false;
    seen.add(normalized);
    return true;
  });
}

function normalizeClauseValue(raw: string): string {
  let value = raw.trim();
  if (value.startsWith('"') && value.endsWith('"')) value = value.slice(1, -1);
  if (value.startsWith('(') && value.endsWith(')')) value = value.slice(1, -1);
  return value.trim();
}

function splitClauseTerms(raw: string, splitWords = false): string[] {
  const value = normalizeClauseValue(raw);
  if (!value) return [];
  if (value.includes('|')) return dedupe(value.split('|').map((part) => part.trim().replace(/^['"]|['"]$/g, '')));
  if (splitWords) return dedupe(value.split(/\s+/));
  return [value];
}

function parseFreeTerms(query: string): string[] {
  const matches = query.match(/"[^"]*"|\S+/g) || [];
  const terms: string[] = [];

  for (const rawToken of matches) {
    let token = rawToken.trim();
    if (!token || ['&', '|', '&&', '||'].includes(token)) continue;

    while (token.startsWith('+') || token.startsWith('-') || token.startsWith('~')) {
      if (token.startsWith('-')) {
        token = '';
        break;
      }
      token = token.slice(1);
    }

    token = token.trim().replace(/^[()]+|[()]+$/g, '');
    if (!token || ['&', '|'].includes(token)) continue;
    if (token.startsWith('"') && token.endsWith('"')) token = token.slice(1, -1);

    if (token.includes('|')) {
      terms.push(...splitClauseTerms(token));
    } else {
      terms.push(token);
    }
  }

  return dedupe(terms);
}

export interface ParsedSearchSyntax {
  contentTerms: string[];
  urlTerms: string[];
}

export function parseSearchSyntax(query: string): ParsedSearchSyntax {
  const normalized = normalizeQuotes(query || '');
  const contentTerms: string[] = [];
  const urlTerms: string[] = [];
  const consumedRanges: Array<[number, number]> = [];

  for (const match of normalized.matchAll(FIELD_PATTERN)) {
    const prefix = match[1] || '';
    const operator = (match[2] || '').toLowerCase();
    const rawValue = match[3] || '';
    const index = match.index ?? -1;
    if (index >= 0) consumedRanges.push([index, index + match[0].length]);
    if (prefix === '-') continue;

    if (operator === 'intext' || operator === 'allintext' || operator === 'inpostauthor' || operator === 'allinpostauthor') {
      contentTerms.push(...splitClauseTerms(rawValue, operator === 'allintext' || operator === 'allinpostauthor'));
      continue;
    }

    if (operator === 'inurl' || operator === 'allinurl' || operator === 'site' || operator === 'filetype' || operator === 'ext' || operator === 'link' || operator === 'inanchor' || operator === 'allinanchor' || operator === 'related' || operator === 'cache') {
      urlTerms.push(...splitClauseTerms(rawValue, operator === 'allinurl' || operator === 'allinanchor'));
    }
  }

  const remainingChars = normalized.split('');
  for (const [start, end] of consumedRanges) {
    for (let i = start; i < end; i += 1) remainingChars[i] = ' ';
  }

  contentTerms.push(...parseFreeTerms(remainingChars.join('')));

  return {
    contentTerms: dedupe(contentTerms),
    urlTerms: dedupe(urlTerms),
  };
}
