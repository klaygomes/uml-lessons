export default k => {
  const { g, line, path, rect, text, arrow, box, step, range } = k;
  const xmark = (x, y, r = 4.5) => g({ cls: 'k-cross' }, path(`M${x - r} ${y - r}L${x + r} ${y + r}M${x + r} ${y - r}L${x - r} ${y + r}`, { cls: 'k-mark-bad' }));
  const bits = i => i.toString(2).padStart(4, '0');
  const ORDER = { '1111': 1, '0000': 2, '1110': 3, '1011': 4, '0001': 5, '0100': 6, '1101': 7, '0010': 8 };
  const SEEN = Object.keys(ORDER);
  const junk = m => m[0] === '1';

  const mark = (x, y, isJunk, s = 10, o = {}) => rect(x - s / 2, y - s / 2, s, s, { cls: isJunk ? 'l1-junk' : 'l1-keep', ...o });
  const both = slug => `<defs><marker id="${slug}-both" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto-start-reverse"><polygon class="k-head" points="0 0, 8 3, 0 6"/></marker></defs>`;
  const twoWay = (slug, x1, y1, x2, y2) => path(`M${x1} ${y1}L${x2} ${y2}`, { cls: 'k-arrow', 'marker-start': `url(#${slug}-both)`, 'marker-end': `url(#${slug}-both)` });

  const messages = L => {
    const ph = L.phone;
    const cw = ph ? 72 : 116, chh = 72, x0 = ph ? 52 : 164, y0 = ph ? 64 : 72;
    const fs = ph ? 'font-size:13px' : 'font-size:15px';
    const cells = range(0, 15).map(i => {
      const m = bits(i), r = i >> 2, c = i & 3;
      const x = x0 + c * cw, y = y0 + r * chh, cx = x + cw / 2;
      const seen = m in ORDER, j = junk(m);
      return g({},
        rect(x + 3, y + 3, cw - 6, chh - 6, { cls: seen ? 'l1-seen' : 'l1-unseen', rx: 4 }),
        seen ? text(x + 8, y + (ph ? 16 : 18), String(ORDER[m]), { anchor: 'start', cls: 'k-num-input', style: 'font-size:12px' }) : '',
        text(cx, y + (ph ? 29 : 26), m, { cls: 'k-num', style: fs }),
        mark(cx - 20, y + 41, j, 9),
        text(cx - 10, y + 45, j ? 'junk' : 'keep', { anchor: 'start', cls: 'k-label' }),
        !seen && j ? g({}, xmark(cx - 20, y + 58),
          text(cx - 10, y + 62, 'wrong', { anchor: 'start', cls: 'k-num-accent' })) : '');
    });
    const heads = g({},
      text(x0 + 2 * cw, y0 - 30, 'U N, the last two bits', { cls: 'k-tick' }),
      range(0, 3).map(c => text(x0 + c * cw + cw / 2, y0 - 10, bits(c).slice(2).split('').join(' '), { cls: 'k-tick' })),
      text(x0 - 10, y0 - 10, 'F L', { anchor: 'end', cls: 'k-tick' }),
      range(0, 3).map(r => text(x0 - 10, y0 + r * chh + chh / 2 + 4, bits(r).slice(2).split('').join(' '), { anchor: 'end', cls: 'k-tick' })));
    const ky = y0 + 4 * chh + 40;
    const keyItem = (x, y, kind, label) => g({},
      kind === 'seen' ? rect(x, y - 10, 22, 14, { cls: 'l1-seen', rx: 3 })
        : kind === 'unseen' ? rect(x, y - 10, 22, 14, { cls: 'l1-unseen', rx: 3 })
          : kind === 'cross' ? xmark(x + 11, y - 3)
            : mark(x + 11, y - 3, kind === 'junk', 9),
      text(x + 30, y + 1, label, { anchor: 'start', cls: 'k-label' }));
    const items = [
      ['seen', 'an example; the number is its order'],
      ['unseen', 'unseen message'],
      ['junk', 'junk (F = 1)'],
      ['junk-no', 'keep (F = 0)'],
      ['cross', 'wrong: the memorizer says keep'],
    ];
    const key = ph
      ? items.map(([kd, lb], i) => keyItem(16, ky + i * 22, kd === 'junk-no' ? 'keep' : kd, lb))
      : [keyItem(40, ky, 'seen', items[0][1]), keyItem(400, ky, 'unseen', items[1][1]),
        keyItem(40, ky + 26, 'junk', items[2][1]), keyItem(220, ky + 26, 'keep', items[3][1]), keyItem(400, ky + 26, 'cross', items[4][1])];
    return [heads, cells, key];
  };

  const flip = L => {
    const ph = L.phone;
    const P1 = { 3: 0, 5: 0, 6: 0, 7: 0, 9: 1, 10: 1, 12: 1 };
    const P2 = { 3: 0, 5: 0, 6: 0, 7: 0, 9: 0, 10: 0, 12: 0 };
    const P3 = { 3: 1, 5: 0, 6: 1, 7: 0, 9: 0, 10: 1, 12: 0 };
    const label = (p, i, at1000) => (SEEN.includes(bits(i)) ? i >= 8 : i === 8 ? at1000 : Boolean(p[i]));
    const pitch = ph ? 19 : 14, sq = ph ? 12 : 10;
    const hyp = (x0, cy, p, at1000) => g({}, range(0, 15).map(i => {
      const cx = x0 + i * pitch + pitch / 2, seen = SEEN.includes(bits(i));
      return g({},
        mark(cx, cy, label(p, i, at1000), sq, seen ? { style: 'opacity:0.3' } : {}),
        i === 8 ? rect(cx - pitch / 2 + 0.5, cy - pitch / 2 + 0.5, pitch - 1, pitch - 1, { cls: 'l1-ring', rx: 2 }) : '');
    }));
    const ruler = (x0, y) => g({},
      text(x0, y, '0000', { anchor: 'start', cls: 'k-tick' }),
      text(x0 + 8 * pitch + pitch / 2, y, '1000', { cls: 'k-num-accent' }),
      text(x0 + 16 * pitch, y, '1111', { anchor: 'end', cls: 'k-tick' }));
    const pats = [P1, P2, P3];
    if (ph) {
      const x0 = 28, cx8 = x0 + 8 * pitch + pitch / 2;
      const rowsA = [104, 196, 288];
      return [
        both(L.slug),
        text(180, 24, 'φ: flip the label at 1000, nothing else', { cls: 'k-note', style: 'font-size:15px' }),
        text(180, 44, 'upper row junk at 1000, lower row keep', { cls: 'k-tick' }),
        ruler(x0, 78),
        rowsA.map((ya, j) => g({},
          hyp(x0, ya, pats[j], true),
          hyp(x0, ya + 44, pats[j], false),
          twoWay(L.slug, cx8, ya + 9, cx8, ya + 35),
          text(cx8 + 10, ya + 27, 'φ', { anchor: 'start', cls: 'k-note', style: 'font-size:15px' }))),
        text(180, 372, '⋮', { cls: 'k-name', style: 'font-size:20px' }),
        text(180, 408, '128 with h(1000) = 1', { cls: 'k-name' }),
        text(180, 430, '128 with h(1000) = 0', { cls: 'k-name' }),
        text(180, 450, 'φ pairs them off, so the counts match', { cls: 'k-tick' }),
        g({}, mark(24, 484, true, 10), text(38, 488, 'junk', { anchor: 'start', cls: 'k-label' }),
          mark(104, 484, false, 10), text(118, 488, 'keep', { anchor: 'start', cls: 'k-label' })),
        g({}, mark(24, 510, true, 10, { style: 'opacity:0.3' }), mark(36, 510, false, 10, { style: 'opacity:0.3' }),
          text(50, 514, 'faint: a seen message, fixed by S', { anchor: 'start', cls: 'k-label' })),
        g({}, rect(19, 530, 13, 13, { cls: 'l1-ring', rx: 2 }), text(50, 541, 'the label φ changes', { anchor: 'start', cls: 'k-label' })),
      ];
    }
    const xl = 24, xr = 472, rows = [140, 180, 220], half = 8 * pitch;
    return [
      both(L.slug),
      text(xl + half, 44, 'h(1000) = 1', { cls: 'k-math' }),
      text(xl + half, 64, 'says junk at 1000', { cls: 'k-tick' }),
      text(xr + half, 44, 'h(1000) = 0', { cls: 'k-math' }),
      text(xr + half, 64, 'says keep at 1000', { cls: 'k-tick' }),
      text(360, 80, 'φ: flip the label at 1000,', { cls: 'k-note' }),
      text(360, 100, 'nothing else', { cls: 'k-note' }),
      ruler(xl, 120), ruler(xr, 120),
      rows.map((y, j) => g({},
        hyp(xl, y, pats[j], true), hyp(xr, y, pats[j], false),
        twoWay(L.slug, xl + 16 * pitch + 12, y, xr - 12, y))),
      text(xl + half, 256, '⋮', { cls: 'k-name', style: 'font-size:20px' }),
      text(xr + half, 256, '⋮', { cls: 'k-name', style: 'font-size:20px' }),
      text(xl + half, 296, '128', { cls: 'k-name', style: 'font-size:24px' }),
      text(xr + half, 296, '128', { cls: 'k-name', style: 'font-size:24px' }),
      text(xl + half, 316, 'consistent hypotheses', { cls: 'k-tick' }),
      text(xr + half, 316, 'consistent hypotheses', { cls: 'k-tick' }),
      text(360, 300, 'φ pairs them off,', { cls: 'k-tick' }),
      text(360, 316, 'so the counts match', { cls: 'k-tick' }),
      g({}, mark(40, 366, true, 10), text(54, 370, 'junk', { anchor: 'start', cls: 'k-label' }),
        mark(120, 366, false, 10), text(134, 370, 'keep', { anchor: 'start', cls: 'k-label' }),
        mark(214, 366, true, 10, { style: 'opacity:0.3' }), mark(226, 366, false, 10, { style: 'opacity:0.3' }),
        text(240, 370, 'faint: a seen message, fixed by S', { anchor: 'start', cls: 'k-label' }),
        rect(523, 359.5, 13, 13, { cls: 'l1-ring', rx: 2 }), text(544, 370, 'the label φ changes', { anchor: 'start', cls: 'k-label' })),
    ];
  };

  const funnel = L => {
    const ph = L.phone;
    const EX = [['1111', 'junk'], ['0000', 'keep'], ['1110', 'junk'], ['1011', 'junk'], ['0001', 'keep'], ['0100', 'keep'], ['1101', 'junk'], ['0010', 'keep']];
    const ALIVE = ['FLUN', 'FLUN', 'FLUN', 'FLU', 'FU', 'FU', 'FU', 'F', 'F'];
    const OUT = { 3: 'N', 4: 'L', 7: 'U' };
    const big = n => n.toLocaleString('en-US');
    const W = ph
      ? { lc: 64, mc: 166, rc: 262, chip0: 220, cp: 22, cw: 18, unit: 6, top: 132, bh: 20, fs: undefined, fy: 400, fh: 96 }
      : { lc: 152, mc: 360, rc: 568, chip0: 476, cp: 36, cw: 28, unit: 14, top: 148, bh: 22, fs: undefined, fy: 416, fh: 92 };
    const ry = r => W.top + r * 30;
    const leftRow = r => {
      const w = W.unit * (16 - r), y = ry(r);
      return g({}, rect(W.lc - w / 2, y - W.bh / 2, w, W.bh, { cls: 'l1-bar', rx: 3 }),
        text(W.lc, y + 4, big(2 ** (16 - r)), { cls: 'k-num', style: W.fs }));
    };
    const midRow = r => (r === 0
      ? text(W.mc, ry(0) + 4, ph ? 'no example' : 'before any example', { cls: 'k-tick' })
      : text(W.mc, ry(r) + 4, `${r}  ${EX[r - 1][0]}  ${EX[r - 1][1]}`, { cls: 'k-num', style: W.fs }));
    const rightRow = r => {
      const y = ry(r);
      return g({}, 'FLUN'.split('').map((f, j) => {
        const x = W.chip0 + j * W.cp, alive = ALIVE[r].includes(f), dying = OUT[r] === f;
        return g({},
          rect(x, y - W.bh / 2, W.cw, W.bh, { cls: alive ? 'l1-chip' : 'l1-chip-dead', rx: 3 }),
          alive ? text(x + W.cw / 2, y + 5, f, { cls: 'k-name', style: ph ? 'font-size:12px' : 'font-size:13px' }) : '',
          dying ? g({ cls: 'l1-ink-mark' }, xmark(x + W.cw / 2, y, 4)) : '');
      }),
      ph ? '' : text(W.chip0 + 4 * W.cp + 4, y + 4, String(ALIVE[r].length), { anchor: 'start', cls: 'k-num' }),
      OUT[r] ? text(W.chip0 + 4 * W.cp + (ph ? 4 : 24), y + 4, ph ? 'out' : `${OUT[r]} out`, { anchor: 'start', cls: 'k-tick' }) : '');
    };
    const hF = (x, y, rest, cls) => `<text x="${x}" y="${y}" text-anchor="middle" class="${cls}">h<tspan baseline-shift="sub" font-size="12">F</tspan>${rest}</text>`;
    const finals = () => {
      const lx = ph ? 16 : 24, lw = ph ? 328 : 256, rx = ph ? 16 : 440, rw = ph ? 328 : 256, y = W.fy, h = W.fh, c = lx + lw / 2, ry2 = ph ? y + h + 12 : y, rh = ph ? 72 : h;
      const bw = ph ? 64 : 52, by = y + 60;
      return g({},
        box(lx, y, lw, h, {}),
        text(c, y + 22, '256 consistent', { cls: 'k-name', style: ph ? 'font-size:14px' : undefined }),
        text(c, y + 40, 'for each unseen message:', { cls: 'k-tick' }),
        line(c - bw, by, c + bw, by, { cls: 'k-line' }),
        line(c, by, c, by + 22, { cls: 'k-line' }),
        line(c - 12, by + 22, c + 12, by + 22, { cls: 'k-line' }),
        line(c - bw, by, c - bw, by + 6, { cls: 'k-line' }),
        line(c + bw, by, c + bw, by + 6, { cls: 'k-line' }),
        text(c - bw, by + 20, '128 junk', { cls: 'k-num' }),
        text(c + bw, by + 20, '128 keep', { cls: 'k-num' }),
        box(rx, ry2, rw, rh, { cls: 'k-box-ok' }),
        hF(rx + rw / 2, ry2 + (ph ? 30 : 40), ', the truth', 'k-name-ok'),
        text(rx + rw / 2, ry2 + (ph ? 52 : 62), 'true risk 0', { cls: 'k-num-ok' }));
    };
    return [
      arrow(L.slug, [[W.lc, ph ? 60 : 64], [W.lc, ph ? 76 : 84]]),
      arrow(L.slug, [[W.rc, ph ? 60 : 64], [W.rc, ph ? 76 : 84]]),
      box(ph ? 16 : 120, 16, ph ? 328 : 480, ph ? 44 : 48, { name: 'all hypotheses: 65,536' }),
      text(W.lc, ph ? 94 : 102, 'anything goes', { cls: 'k-name', style: ph ? 'font-size:13px' : undefined }),
      text(W.lc, ph ? 110 : 120, ph ? 'log widths' : 'widths on a log scale', { cls: 'k-tick' }),
      text(W.mc, ph ? 94 : 102, 'example', { cls: 'k-name', style: ph ? 'font-size:13px' : undefined }),
      text(W.rc, ph ? 94 : 102, 'bias first', { cls: 'k-name', style: ph ? 'font-size:13px' : undefined }),
      text(W.rc, ph ? 110 : 120, ph ? 'one feature' : 'one-feature hypotheses', { cls: 'k-tick' }),
      leftRow(0), midRow(0), rightRow(0),
      range(1, 8).map(r => step(r, `example ${r}, ${EX[r - 1][0]} ${EX[r - 1][1]}: ${big(2 ** (16 - r))} left of all hypotheses; one-feature survivors ${ALIVE[r].split('').join(', ')}`,
        g({ anim: { k: 'in', d: 0.4 + (r - 1) * 0.9, len: 0.7 } }, leftRow(r), midRow(r), rightRow(r)))),
      step(8, 'the end: 256 consistent hypotheses split 128 and 128 on every unseen message; the biased learner keeps only h F, the truth, true risk 0',
        g({ anim: { k: 'in', d: 7.6, len: 0.8 } }, finals())),
    ];
  };

  const taxonomy = L => {
    const ph = L.phone;
    const ROWS = [
      ['supervised|or not', [['supervised', ['the mail filter:', 'junk and keep', 'clicks'], true], ['reinforcement', ['chess: games', 'labelled with', 'who won'], false], ['unsupervised', ['anomaly detection,', 'clustering', '(lesson 22)'], false]]],
      ['active or|passive', [['passive', ['the mail filter waits', "for users' labels"], true], ['active', ['the learner picks or', 'composes messages'], false]]],
      ['what kind|of teacher', [['helpful', ['a child, a student'], false], ['indifferent nature', ['what a scientist', 'learns from:', 'statistical'], true], ['adversarial', ['spammers,', 'fraudsters; also', 'the worst case'], false]]],
      ['batch or|online', [['batch', ['a data miner with', 'a big archive'], true], ['online', ['a stockbroker, every', 'day (lesson 21)'], false]]],
    ];
    const cell = (x, y, w, h, [name, lines, focus], o) => g({},
      rect(x, y, w, h, { cls: focus ? 'l1-focus' : 'l1-cell', rx: 4 }),
      text(x + (o.one ? 10 : 12), y + (o.one ? 18 : 22), name, { anchor: 'start', cls: 'k-name', style: 'font-size:14px' }),
      o.one ? text(x + 10, y + 35, lines.join(' '), { anchor: 'start', cls: 'k-tick' })
        : lines.map((s, i) => text(x + 12, y + 40 + i * 15, s, { anchor: 'start', cls: 'k-tick' })));
    if (ph) {
      let y = 28;
      const out = [];
      ROWS.forEach(([title, cells]) => {
        out.push(text(16, y, title.replace('|', ' '), { anchor: 'start', cls: 'k-label' }));
        y += 12;
        cells.forEach(c => { out.push(cell(16, y, 328, 44, c, { one: true })); y += 50; });
        y += 22;
      });
      out.push(rect(16, y - 4, 22, 14, { cls: 'l1-focus', rx: 3 }), text(46, y + 7, "this course's main focus", { anchor: 'start', cls: 'k-label' }));
      return out;
    }
    const x0 = 152, span = 544, gap = 12, h = 84;
    return [
      ROWS.map(([title, cells], r) => {
        const y = 24 + r * 96, w = (span - gap * (cells.length - 1)) / cells.length;
        return g({},
          title.split('|').map((t, i) => text(24, y + h / 2 - 3 + i * 18, t, { anchor: 'start', cls: 'k-name', style: 'font-size:14px' })),
          cells.map((c, i) => cell(x0 + i * (w + gap), y, w, h, c, {})));
      }),
      rect(152, 412, 22, 14, { cls: 'l1-focus', rx: 3 }),
      text(182, 423, "this course's focus: supervised, passive, statistical, batch", { anchor: 'start', cls: 'k-label' }),
    ];
  };

  const css = [
    '.k-fig .l1-seen { fill: var(--paper); stroke: var(--ink); stroke-width: 1.5; }',
    '.k-fig .l1-unseen { fill: var(--paper); stroke: var(--ink-soft); stroke-width: 1.2; stroke-dasharray: 4 3; }',
    '.k-fig .l1-junk { fill: var(--ink); }',
    '.k-fig .l1-keep { fill: var(--paper); stroke: var(--ink); stroke-width: 1.2; }',
    '.k-fig .l1-ring { fill: none; stroke: var(--accent); stroke-width: 2; }',
    '.k-fig .l1-bar { fill: var(--paper-2); stroke: var(--ink-soft); stroke-width: 1; }',
    '.k-fig .l1-chip { fill: var(--paper-2); stroke: var(--ink); stroke-width: 1.2; }',
    '.k-fig .l1-chip-dead { fill: none; stroke: var(--ink-soft); stroke-width: 1; stroke-dasharray: 3 3; }',
    '.k-fig .l1-ink-mark .k-mark-bad { stroke: var(--ink); }',
    '.k-fig .l1-cell { fill: var(--paper); stroke: var(--rule); stroke-width: 1; }',
    '.k-fig .l1-focus { fill: var(--input-soft); stroke: var(--ink); stroke-width: 1.5; }',
  ].join(' ');
  const fig = (name, size, title, desc, caption, body, steps = 0) => ({
    name, w: 720, h: size[0], phone: { w: 360, h: size[1] }, title, desc, caption, steps,
    eyebrow: 'Lesson 1 · learning needs a bias', css, body: (kk, L) => body(L),
  });

  return [
    fig('messages', [440, 500], 'All 16 messages, eight of them seen',
      'The 16 messages of the tiny world as a 4 by 4 grid; the eight training messages are outlined solid with their order, the eight unseen ones dashed; junk messages are the ones starting with 1, and the memorizer is wrong on the four unseen junk messages 1000, 1001, 1010 and 1100.',
      'The tiny world: every message is four bits, FLUN, and a message is junk exactly when it says "FREE". Solid cells are the eight examples, numbered in order. The memorizer says keep for every unseen message, so it is wrong on the four unseen junk ones.',
      messages),
    fig('funnel', [520, 600], 'Two ways to narrow 65,536 hypotheses',
      'Two columns start from all 65,536 hypotheses; with anything allowed, each of the eight examples halves the count, ending at 256 consistent hypotheses split 128 to 128 on every unseen message; with one-feature hypotheses chosen first, only 4 remain before any example, the examples knock out N, L and U, and h F, the truth, is left with true risk 0.',
      'Left, anything goes: each example halves the count, and 256 consistent hypotheses remain, split evenly on every unseen message. Right, a bias first cuts the field to 4; examples 3, 4 and 7 knock out N, L and U, and only the truth is left.',
      funnel, 8),
    fig('flip', [380, 560], 'Flipping one unseen label',
      'Pairs of consistent hypotheses drawn as rows of 16 labels; the eight seen labels are the same in every row, and the map φ flips only the label at the unseen message 1000, pairing the 128 hypotheses that say junk there with the 128 that say keep.',
      'Each row is one consistent hypothesis, its 16 labels in order from 0000 to 1111. φ changes only the label at 1000, so it pairs every hypothesis that says junk there with one that says keep: 128 each.',
      flip),
    fig('taxonomy', [448, 700], 'Four ways to sort learning problems',
      "Four dimensions of learning, each with its settings and a classic example: supervised, reinforcement or unsupervised; passive or active; a helpful, indifferent or adversarial teacher; batch or online; this course's main focus is tinted in each row.",
      "Four questions that sort any learning problem, with a classic example under each setting. Tinted cells are this course's main focus: supervised, passive, statistical, batch. Online learning comes in lesson 21, clustering in lesson 22.",
      taxonomy),
  ];
};
