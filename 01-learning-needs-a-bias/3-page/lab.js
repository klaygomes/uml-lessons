/* Sieve drawing adapted from the iwrzwr visual archive, Rank Sieve. MIT License, Copyright (c) 2026 Kagan Yaldizkaya. */
(() => {
  const K = window.Kit, D = 10, N = 1 << D, M_MAX = 200, BUDGET = 12, SEED = 20261120;
  const SUB = '₀₁₂₃₄₅₆₇₈₉';
  const sub = n => String(n).split('').map(d => SUB[Number(d)]).join('');
  const hName = hy => hy.map(j => `h${sub(j + 1)}`).join(' ∧ ');
  const bit = (x, j) => (x >> j) & 1;
  const says = (hy, x) => (hy.every(j => bit(x, j)) ? 1 : 0);
  const num = n => n.toLocaleString('en-US');
  const riskText = r => (r === 0 ? '0%' : r < 0.001 ? 'under 0.1%' : `${(r * 100).toFixed(1)}%`);
  const chanceText = p => (p >= 0.01 ? `${(p * 100).toFixed(1)}%` : p >= 0.001 ? `${(p * 100).toFixed(2)}%` : 'under 0.1%');
  const impostorChance = m => 1 - (1 - 2 ** -m) ** (D - 1);
  const labelsText = n => `${n} label${n === 1 ? '' : 's'}`;

  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const draw = (r, n) => Array.from({ length: n }, () => Math.floor(r() * N));

  const SINGLES = Array.from({ length: D }, (_, j) => [j]);
  const PAIRS = SINGLES.flatMap(([j]) => Array.from({ length: D - j - 1 }, (_, i) => [j, j + i + 1]));
  const CLASSES = {
    any: { name: 'anything goes', H: null },
    ten: { name: 'one feature of ten', H: SINGLES },
    three: { name: 'one feature of the first three', H: SINGLES.slice(0, 3) },
    pairs: { name: 'one or two features', H: [...SINGLES, ...PAIRS] },
  };
  const CASES = [5, 1, 8, 4, 7, 0, 9, 3, 6, 2];

  const riskOf = (hy, js) => { let e = 0; for (let x = 0; x < N; x++) if (says(hy, x) !== bit(x, js)) e++; return e / N; };

  function memorize(xs, js) {
    const seen = new Set(xs);
    let wrong = 0;
    for (let x = 0; x < N; x++) if (!seen.has(x) && bit(x, js)) wrong++;
    return { seen, k: seen.size, risk: wrong / N, wrong };
  }

  function sieveRun(H, xs, js, m, pickSeed) {
    const kill = H.map(hy => { const t = xs.slice(0, m).findIndex(x => says(hy, x) !== bit(x, js)); return t < 0 ? Infinity : t + 1; });
    const alive = H.map((_, i) => i).filter(i => kill[i] === Infinity);
    const pick = alive.length ? alive[Math.floor(rng(pickSeed)() * alive.length)] : -1;
    const hy = pick >= 0 ? H[pick] : null;
    return { kill, alive, pick, hy, risk: hy ? riskOf(hy, js) : null, truth: hy ? hy.length === 1 && hy[0] === js : false };
  }

  function guarantee(H, js, m) {
    const risks = H.map(hy => riskOf(hy, js)).filter(r => r > 0);
    return { count: risks.length, chance: 1 - risks.reduce((t, r) => t * (1 - (1 - r) ** m), 1) };
  }

  function supText(s, base, power, x, y, color) {
    const { pen, ctx } = s;
    pen.text(base, x, y, { align: 'left', color, size: 12 });
    const bw = ctx.measureText(base).width;
    pen.text(power, x + bw + 1, y - 5, { align: 'left', color, size: 9 });
    return x + bw + 2 + ctx.measureText(power).width;
  }

  const SUBS = '₀₁₂₃₄₅₆₇₈₉ⱼₖ', PLAIN = '0123456789jk', SUB_RE = /h([₀₁₂₃₄₅₆₇₈₉ⱼₖ]+)/;
  function subify(root) {
    const walk = document.createTreeWalker(root, NodeFilter.SHOW_TEXT), hits = [];
    while (walk.nextNode()) if (SUB_RE.test(walk.currentNode.nodeValue) && !walk.currentNode.parentElement.closest('sub, canvas')) hits.push(walk.currentNode);
    hits.forEach(t => {
      const f = document.createDocumentFragment();
      t.nodeValue.split(/(h[₀₁₂₃₄₅₆₇₈₉ⱼₖ]+)/).forEach(part => {
        const m = part.match(/^h([₀₁₂₃₄₅₆₇₈₉ⱼₖ]+)$/);
        if (!m) { if (part) f.append(part); return; }
        const sb = document.createElement('sub');
        sb.textContent = [...m[1]].map(ch => PLAIN[SUBS.indexOf(ch)]).join('');
        f.append('h', sb);
      });
      t.replaceWith(f);
    });
  }

  function hLabel(s, j, x, y, color) {
    const { pen, ctx } = s, d = String(j);
    pen.text('h', 0, -100, { color, size: 12 });
    const hw = ctx.measureText('h').width;
    pen.text(d, 0, -100, { color, size: 10 });
    const dw = ctx.measureText(d).width, x0 = x - (hw + dw) / 2;
    pen.text('h', x0, y, { align: 'left', color, size: 12 });
    pen.text(d, x0 + hw, y + 3, { align: 'left', color, size: 10 });
  }

  const riskPart = (r, solved) => (solved ? { ok: riskText(r) } : r >= 0.01 ? { changed: riskText(r) } : riskText(r));

  function sup(base, power) {
    const f = document.createDocumentFragment(), s = document.createElement('sup');
    s.textContent = power; f.append(base, s);
    return f;
  }
  function line(...parts) {
    const p = document.createElement('div');
    parts.forEach(x => {
      if (x && (x.changed || x.ok)) { const s = document.createElement('span'); s.className = x.ok ? 'ok' : 'changed'; s.textContent = x.ok || x.changed; p.append(s); } else p.append(x);
    });
    return p;
  }

  function sieveRow(s, o) {
    const { pen, c, w, ctx } = s, n = D, pitch = (w - 24) / n, bw = Math.min(pitch - 6, 46), bh = 28, y0 = o.y;
    const X = i => 12 + pitch * (i + 0.5), small = pitch < 40;
    if (o.outside) {
      const a = X(3) - bw / 2, b = X(n - 1) + bw / 2;
      pen.text(small ? 'outside this class' : 'features 4 to 10: outside this class', (a + b) / 2, y0 - 8, { color: c.soft, size: 12 });
    }
    for (let i = 0; i < n; i++) {
      const x = X(i), inside = o.inside(i), ix = o.index(i);
      if (!inside) {
        pen.line([[x - bw / 2, y0], [x + bw / 2, y0], [x + bw / 2, y0 + bh], [x - bw / 2, y0 + bh], [x - bw / 2, y0]], c.soft, 1, [3, 3], 0.6);
        hLabel(s, i + 1, x, y0 + 17, c.soft);
        continue;
      }
      const k = o.kill[ix], dead = k <= o.p, fresh = !o.done && dead && o.p - k < o.span;
      if (fresh) pen.rect(x - bw / 2 - 2, y0 - 2, bw + 4, bh + 4, c.accent, 0.88);
      const a = dead ? 0.16 : 0.92, col = fresh ? c.paper : c.ink;
      ctx.globalAlpha = a; ctx.strokeStyle = col; ctx.lineWidth = 1;
      ctx.strokeRect(Math.round(x - bw / 2) + 0.5, y0 + 0.5, Math.round(bw), bh);
      ctx.globalAlpha = 1;
      hLabel(s, i + 1, x, y0 + 17, col);
      ctx.globalAlpha = 1;
      if (o.pinned && o.pinned.has(ix)) pen.line([[x - bw / 2 - 4, y0 - 4], [x + bw / 2 + 4, y0 - 4], [x + bw / 2 + 4, y0 + bh + 4], [x - bw / 2 - 4, y0 + bh + 4], [x - bw / 2 - 4, y0 - 4]], c.input, 1.5, [4, 3]);
      if (dead && Number.isFinite(k)) pen.text(`#${k}`, x, y0 + bh + 15, { color: c.soft, size: 12 });
      else if (o.truth === i) pen.text('truth', x, y0 + bh + 15, { color: c.soft, size: small ? 9 : 10 });
      if (o.done && o.pick === ix) {
        pen.line([[x - 5, y0 + bh + 30], [x, y0 + bh + 24], [x + 5, y0 + bh + 30]], o.solved ? c.ok : c.accent, 2);
        pen.text('returned', x, y0 + bh + 43, { color: o.solved ? c.ok : c.accentInk, size: 12 });
      }
    }
  }

  function sieveTriangle(s, o) {
    const { pen, c, w } = s, cs = Math.min(24, Math.floor((w - 60) / D)), x0 = Math.round((w - cs * D) / 2) + 10, y0 = o.y + 14;
    for (let j = 0; j < D; j++) {
      pen.text(String(j + 1), x0 + j * cs + cs / 2, y0 - 4, { color: c.soft, size: 12 });
      pen.text(String(j + 1), x0 - 8, y0 + j * cs + cs / 2 + 4, { align: 'right', color: c.soft, size: 12 });
    }
    o.H.forEach((hy, ix) => {
      const [a, b] = hy.length === 1 ? [hy[0], hy[0]] : hy, x = x0 + b * cs, y = y0 + a * cs;
      const k = o.kill[ix], dead = k <= o.p, fresh = !o.done && dead && o.p - k < o.span;
      if (fresh) pen.rect(x, y, cs - 2, cs - 2, c.accent, 0.88);
      else pen.rect(x, y, cs - 2, cs - 2, c.ink, dead ? 0.12 : 0.85);
      if (o.done && o.pick === ix) {
        s.ctx.strokeStyle = o.solved ? c.ok : c.accent; s.ctx.lineWidth = 2.5; s.ctx.strokeRect(x - 2, y - 2, cs + 2, cs + 2);
      }
    });
    pen.text(w < 480 ? 'diagonal: one feature; row j, column k: hj ∧ hk' : 'diagonal: one feature, hj; row j, column k: two features, hj ∧ hk', w / 2, y0 + D * cs + 18, { color: c.soft, size: 12 });
    return y0 + D * cs + 26;
  }

  function grid(s, o) {
    const { pen, c } = s, cs = o.cs, gap = cs >= 5 ? 1 : 0.6;
    pen.rect(o.x, o.y, 32 * cs, 32 * cs, c.paper2);
    for (let x = 0; x < N; x++) {
      const gx = o.x + (x & 31) * cs, gy = o.y + (x >> 5) * cs;
      if (o.wrong && o.wrong(x)) pen.rect(gx, gy, cs - gap, cs - gap, c.accent);
      else if (o.seenAt && o.seenAt.has(x) && o.seenAt.get(x) < o.p) pen.rect(gx + cs / 4, gy + cs / 4, cs / 2, cs / 2, c.ink);
    }
  }

  function riskScale(s, o) {
    const { pen, c, w } = s, phone = w < 560, x0 = phone ? 16 : 96, x1 = w - 24, y = o.y, X = r => x0 + (x1 - x0) * Math.min(r, 0.5) / 0.5;
    if (!phone) pen.text('true risk', x0 - 14, y + 4, { align: 'right', color: c.soft, size: 12 });
    pen.line([[x0, y], [x1, y]], c.soft, 1);
    [0, 0.25, 0.5].forEach(r => { pen.line([[X(r), y], [X(r), y + 4]], c.soft, 1); pen.text(`${r * 100}%`, X(r), y + 30, { color: c.soft, size: 12 }); });
    if (o.pinned) {
      [o.pinned.mem, o.pinned.one].forEach(r => { pen.line([[X(r), y - 18], [X(r), y + 18]], c.input, 1.5, [4, 3]); });
      s.ctx.strokeStyle = c.input; s.ctx.lineWidth = 1.5;
      s.ctx.strokeRect(X(o.pinned.mem) - 5, y - 14, 10, 10);
      pen.ring(X(o.pinned.one), y + 9, 5, c.input);
    }
    if (o.mem === null) { pen.text('?', X(0.25), y - 8, { color: c.soft, size: 12 }); return; }
    pen.rect(X(o.mem) - 5, y - 14, 10, 10, c.accent);
    pen.text('memorizer', X(o.mem), y - 20, { align: X(o.mem) > x1 - 40 ? 'right' : 'center', color: c.ink, size: 12 });
    if (o.one !== null) {
      pen.dot(X(o.one), y + 9, 5, c.accent);
      pen.text('one-feature', X(o.one), y + 44, { align: X(o.one) < x0 + 40 ? 'left' : X(o.one) > x1 - 40 ? 'right' : 'center', color: c.ink, size: 12 });
    }
  }

  function mount(host, mode) {
    host.replaceChildren(document.getElementById('bias-lab').content.cloneNode(true));
    const uid = `${host.id}-`, $ = k => host.querySelector(`[data-k="${k}"]`);
    host.querySelectorAll('[data-k]').forEach(el => { el.id = uid + el.dataset.k; });
    host.querySelectorAll('[data-for]').forEach(el => { el.htmlFor = uid + el.dataset.for; });
    const explore = mode === 'explore';
    let resets = 0;
    const fresh1 = () => { const seed = SEED + 7919 * resets, r = rng(seed); return { js: Math.floor(r() * D), seed, xs: draw(r, M_MAX) }; };
    const st = explore
      ? { ...fresh1(), m: 100, p: 100, predicted: false, revealed: false, pinned: null, resamples: 0 }
      : { predicted: false, cls: 'any', labels: 5, caseIx: 0, run: null, solved: false, r: rng(SEED + 7), p: 0 };

    const loop = K.loop(host, dt => {
      const target = explore ? st.m : st.run ? st.run.m : 0;
      st.p = Math.min(target + 0.01, st.p + dt * st.rate);
      stage.redraw();
      if (st.p >= target) loop.stop();
    });
    function animate(m) {
      st.rate = Math.max(6, m / 1.4); st.span = Math.max(0.8, st.rate * 0.3);
      if (K.reduced.matches) { st.p = m + 0.01; return; }
      st.p = 0; loop.start();
    }

    function act1() {
      const xs = st.xs.slice(0, st.m), mem = memorize(xs, st.js);
      const one = sieveRun(SINGLES, st.xs, st.js, st.m, st.seed * 7 + st.m);
      const seenAt = new Map(); xs.forEach((x, i) => { if (!seenAt.has(x)) seenAt.set(x, i); });
      return { mem, one, seenAt };
    }

    const stage = K.stage($('stage'), s => {
      const { pen, c, w } = s, phone = w < 560;
      if (explore) {
        const r = act1(), done = st.p >= st.m;
        pen.text(phone ? 'one-feature class, 10 hypotheses' : 'one-feature class: which of the 10 hypotheses survive', 12, 16, { align: 'left', color: c.ink, size: 13, font: 'body', weight: 600 });
        sieveRow(s, { y: 32, inside: () => true, index: i => i, kill: r.one.kill, p: st.p, span: st.span || 1, done, pick: st.predicted ? r.one.pick : -1, truth: st.revealed ? st.js : -1, pinned: st.pinned && st.pinned.alive });
        const cs = Math.min(7, (w - 48) / 64), gw = 32 * cs, gy = 150, gap = Math.min(w - 24 - 2 * gw, 72), left = Math.round((w - 2 * gw - gap) / 2);
        const gx = [left, left + gw + gap];
        pen.text('memorizer', gx[0], gy - 8, { align: 'left', color: c.ink, size: 12 });
        pen.text(phone ? 'one-feature' : 'one-feature learner', gx[1], gy - 8, { align: 'left', color: c.ink, size: 12 });
        const show = st.predicted && done;
        grid(s, { x: gx[0], y: gy, cs, seenAt: r.seenAt, p: st.p, wrong: show ? x => !r.mem.seen.has(x) && bit(x, st.js) : null });
        grid(s, { x: gx[1], y: gy, cs, seenAt: r.seenAt, p: st.p, wrong: show && r.one.hy ? x => says(r.one.hy, x) !== bit(x, st.js) : null });
        const under = gy + gw + 16;
        pen.text(show ? `wrong on ${num(r.mem.wrong)}` : `${num(r.mem.k)} seen`, gx[0], under, { align: 'left', color: c.soft, size: 12 });
        pen.text(show ? `wrong on ${num(Math.round(r.one.risk * N))}` : `${num(r.mem.k)} seen`, gx[1], under, { align: 'left', color: c.soft, size: 12 });
        riskScale(s, { y: under + 44, mem: show ? r.mem.risk : null, one: show ? r.one.risk : null, pinned: st.predicted ? st.pinned : null });
        return;
      }
      const cls = CLASSES[st.cls], run = st.run, done = run && st.p >= run.m;
      pen.text(phone ? cls.name : `hypothesis class: ${cls.name}`, 12, 16, { align: 'left', color: c.ink, size: 13, font: 'body', weight: 600 });
      let y = 40;
      if (!cls.H) {
        const k = run ? run.mem.k : 0;
        const after = supText(s, '2', num(N - k), 12, y + 6, c.soft);
        pen.text(' consistent hypotheses: too many to draw.', after, y + 6, { align: 'left', color: c.soft, size: 12 });
        pen.text('The learner returns the memorizer: its table, and keep everywhere else.', 12, y + 26, { align: 'left', color: c.soft, size: 12 });
        y += 50;
      } else if (st.cls === 'pairs') {
        y = sieveTriangle(s, { y, H: cls.H, kill: run ? run.sieve.kill : cls.H.map(() => Infinity), p: run ? st.p : 0, span: st.span || 1, done, pick: run ? run.sieve.pick : -1, solved: st.solved });
      } else {
        y += 6;
        sieveRow(s, { y, outside: st.cls === 'three', inside: i => i < cls.H.length, index: i => i, kill: run ? run.sieve.kill : cls.H.map(() => Infinity), p: run ? st.p : 0, span: st.span || 1, done, pick: run ? run.sieve.pick : -1, truth: st.solved ? CASES[st.caseIx] : -1, solved: st.solved });
        y += 84;
      }
      const cs = Math.min(7, (w - 24) / 32), gx = Math.round((w - 32 * cs) / 2);
      const hyp = run && (run.sieve ? run.sieve.hy : 'mem');
      const title = !run ? (phone ? 'no examples yet' : 'all 1,024 messages: no examples yet') : !hyp ? (phone ? 'nothing returned' : 'no consistent hypothesis: nothing to run') : phone ? 'returned hypothesis, 1,024 messages' : 'the returned hypothesis on all 1,024 messages';
      pen.text(title, phone ? 12 : gx, y + 12, { align: 'left', color: c.ink, size: 12 });
      const js = CASES[st.caseIx];
      const seenAt = new Map(); if (run) run.xs.forEach((x, i) => { if (!seenAt.has(x)) seenAt.set(x, i); });
      const wrong = !done || !hyp ? null : hyp === 'mem' ? x => !run.mem.seen.has(x) && bit(x, js) : x => says(hyp, x) !== bit(x, js);
      grid(s, { x: gx, y: y + 20, cs, seenAt, p: run ? st.p : 0, wrong });
    }, { height: w => {
      const phone = w < 560;
      if (explore) return 150 + 32 * Math.min(7, (w - 48) / 64) + 16 + 96 + (phone ? 0 : 0);
      const cs = Math.min(7, (w - 24) / 32), grid = 32 * cs + 36;
      if (st.cls === 'pairs') return 40 + 14 + D * Math.min(24, Math.floor((w - 60) / D)) + 26 + grid;
      return (st.cls === 'any' ? 90 : 130) + grid;
    } });

    function lastOut(r) {
      const ks = r.one.kill.filter(k => Number.isFinite(k) && k <= st.m);
      return ks.length ? Math.max(...ks) : 0;
    }

    function exploreWhy(r) {
      const m = st.m, k = r.one.alive.length;
      const impostors = k - 1, p = impostorChance(m);
      if (m <= 3) return `With ${m} example${m === 1 ? '' : 's'}, each impostor survives with chance 1 in ${2 ** m}, but there are nine of them, so the chance that at least one survives is about ${chanceText(impostorChance(m))}. ${k > 1 ? `This run, ${k} of the 10 survive, and the one-feature learner picks one of them blindly.` : 'This run, only the truth survives: a lucky draw. Press Resample to see a typical one.'}`;
      if (m <= 10) return `An impostor sometimes survives: the chance that some wrong feature is still consistent after ${m} examples is ${chanceText(p)}. This run, ${impostors ? `${impostors} impostor${impostors === 1 ? '' : 's'} survive${impostors === 1 ? 's' : ''} beside the truth` : 'only the truth survives'}. Press Resample a few times and watch impostors appear and vanish.`;
      const memo = `The memorizer has seen ${num(r.mem.k)} different messages and says keep for the other ${num(N - r.mem.k)}, about half of them junk, so it is wrong on ${riskText(r.mem.risk)}.`;
      if (m === M_MAX) return `The far end, 200 examples. ${memo} ${lastOut(r) ? `In this run the last impostor was ruled out by example ${lastOut(r)}, and from then on the one-feature learner returned the truth.` : ''}`;
      return `Impostors are all but gone: the chance one survives ${m} examples is ${chanceText(p)}. The one-feature learner sits at 0%. ${memo}`;
    }

    function act1Render() {
      const r = act1(), shown = st.predicted, open = st.revealed;
      $('m').value = st.m; $('m-out').textContent = st.m;
      ['m', 'resample', 'pin'].forEach(k => { $(k).disabled = !shown; });
      const memLine = line('memorizer · consistent: ', shown ? sup('2', num(N - r.mem.k)) : '?', ' · true risk ', shown ? riskPart(r.mem.risk) : '?');
      const role = r.one.truth ? ', the truth' : ', an impostor';
      const oneLine = line(`one-feature · consistent: ${shown ? r.one.alive.length : '?'} of 10 · returns ${shown ? hName(r.one.hy) : '?'}${open ? role : ''} · true risk `, shown ? riskPart(r.one.risk) : '?');
      $('result').replaceChildren(memLine, oneLine);
      $('why').textContent = !shown ? 'Answer the question above to unlock the slider. The lab has already drawn 100 examples.'
        : exploreWhy(r);
      $('key-pin').hidden = !st.pinned;
      $('compare').hidden = !st.pinned;
      if (st.pinned) {
        const P = st.pinned, same = P.m === st.m && P.seed === st.seed;
        const now = `m = ${st.m}, memorizer ${riskText(r.mem.risk)}, one-feature ${riskText(r.one.risk)} (${hName(r.one.hy)}${open ? role : ''})`;
        $('compare').textContent = same ? `Pinned this run: ${now}. Move m or press Resample to compare.`
          : `Pinned: m = ${P.m}, memorizer ${riskText(P.mem)}, one-feature ${riskText(P.one)} (${P.name}). Now: ${now}.`;
      }
      $('stage').setAttribute('aria-label', `${r.one.alive.length} of 10 one-feature hypotheses survive ${st.m} examples; the memorizer has seen ${num(r.mem.k)} of 1,024 messages${shown ? `; true risks ${riskText(r.mem.risk)} and ${riskText(r.one.risk)}` : ''}.`);
      return r;
    }

    function trainOnce() {
      const js = CASES[st.caseIx], m = st.labels, xs = draw(st.r, m), H = CLASSES[st.cls].H;
      const mem = memorize(xs, js);
      const sieve = H ? sieveRun(H, xs, js, m, Math.floor(st.r() * 1e9)) : null;
      st.runId = (st.runId || 0) + 1;
      st.run = { m, xs, mem, sieve, cls: st.cls, risk: sieve ? sieve.risk : mem.risk };
      st.run.sure = H ? guarantee(H, js, m) : null;
      st.solved = Boolean(sieve && sieve.hy && sieve.risk < 0.01 && st.run.sure.chance <= 0.01 && m <= BUDGET && st.cls !== 'three');
      animate(m);
    }

    function bigNote(run) {
      if (run.cls !== 'pairs' || run.m > BUDGET) return '';
      return `This class cannot reach the guarantee within this week’s budget of ${BUDGET} labels: its impostors include 9 pairs, each containing the junk feature, that survive each example with probability 3/4.`;
    }

    function fixWhy() {
      if (!st.predicted) return 'Answer the question above to unlock the controls.';
      const run = st.run;
      if (!run) return 'Choose a hypothesis class and a number of labels, then press “Label and train”.';
      const over = run.m > BUDGET, tail = over ? ` And ${labelsText(run.m)} is over this week’s budget of ${BUDGET}: your users will not click that many.` : '';
      if (!run.sieve) return `Consistent with all your examples, wrong on ${riskText(run.mem.risk)} of fresh messages: half the consistent hypotheses say junk for every message it has not seen.${tail}`;
      const sv = run.sieve, k = sv.alive.length;
      if (!sv.hy) return `No hypothesis in this class is consistent with the examples: the real junk feature is not one this class can see.${tail}`;
      if (sv.risk >= 0.01) return `${k} hypothes${k === 1 ? 'is is' : 'es are'} consistent, and the one returned is wrong on ${riskText(sv.risk)} of fresh messages.${bigNote(run) ? ` ${bigNote(run)}` : ''}${tail}`;
      if (run.cls === 'three') return `The learner returned ${hName(sv.hy)}, true risk ${riskText(sv.risk)}: this class happened to contain this week’s junk feature. That is the rat’s gamble. Next week the spammers can move to any of the ten features, and this class cannot be guaranteed to contain it. The guarantee needs a class that contains every possible junk feature.${tail}`;
      const g = run.sure, many = g.count === 1 ? 'the one wrong hypothesis' : `one of the ${g.count} wrong hypotheses`;
      if (g.chance > 0.01) return `The learner returned the truth, true risk ${riskText(sv.risk)}, but that was luck${k > 1 ? `: ${k} hypotheses were consistent and it picked one blindly` : ''}. With ${labelsText(run.m)}, the chance that ${many} in this class was still consistent is about ${chanceText(g.chance)}. ${bigNote(run) || 'How many labels does this class need before that chance falls under 1%? Work it out with the lesson’s formula, then train again.'}${tail}`;
      if (over) return `Only ${hName(sv.hy)} is consistent and its true risk is 0%, but ${labelsText(run.m)} is over this week’s budget of ${BUDGET}: your users will not click that many.`;
      const name = hName(sv.hy), f = CASES[st.caseIx] + 1;
      return st.caseIx === 0
        ? `Fixed. The learner returns ${name}, true risk 0% on all 1,024 messages, and with ${labelsText(run.m)} the chance that a wrong hypothesis in this class is still consistent is under 1% (${chanceText(g.chance)}). This week’s junk feature was feature ${f}. The inbox empties of junk.`
        : `Fixed for this week too. ${name} survives your ${labelsText(run.m)}: the junk feature was feature ${f}, the true risk is 0%, and the chance of a surviving impostor is ${chanceText(g.chance)}. The inbox empties of junk.`;
    }

    function act2Render() {
      const cls = CLASSES[st.cls], run = st.run;
      host.querySelectorAll('[data-cls]').forEach(b => { b.setAttribute('aria-pressed', String(b.dataset.cls === st.cls)); b.disabled = !st.predicted; });
      $('labels').value = st.labels; $('labels').disabled = !st.predicted;
      $('labels-out').textContent = st.labels > BUDGET ? `${st.labels}, over the budget of ${BUDGET}` : `${st.labels} of a budget of ${BUDGET}`;
      $('train').disabled = !st.predicted;
      $('task-text').textContent = st.caseIx === 0
        ? `The spammers changed feature. This week a different feature marks junk, and nobody knows which. Pick a hypothesis class and a number of labels within the budget of ${BUDGET}, then train. Goal: true risk under 1%, and a chance under 1% that a wrong hypothesis is still consistent.`
        : 'The spammers moved again: another feature marks junk this week. Same budget, same goal.';
      const known = Boolean(run && (run.sieve ? run.sieve.hy : true));
      K.meter($('meter'), known ? run.risk : 0, { max: 0.5, known: Boolean(run), text: !known ? 'none returned' : riskText(run.risk) });
      const parts = [];
      if (run) {
        const n = run.sieve ? `${run.sieve.alive.length} of ${cls.H.length} consistent` : null;
        parts.push(line(`${cls.name} · ${labelsText(run.m)} · `, ...(n ? [n] : ['consistent: ', sup('2', num(N - run.mem.k))]),
          ` · returns ${run.sieve ? (run.sieve.hy ? hName(run.sieve.hy) : 'nothing') : 'the memorizer'} · true risk `,
          run.sieve && !run.sieve.hy ? { changed: 'none' } : riskPart(run.risk, st.solved)));
      } else parts.push(line(`${cls.name} · ${labelsText(st.labels)} · not trained yet`));
      $('result').replaceChildren(...parts);
      $('why').textContent = fixWhy();
      host.querySelectorAll('[data-note]').forEach(el => { el.hidden = el.dataset.note !== st.cls; });
      $('key-out').hidden = !cls.H;
      $('another').hidden = !st.solved;
      $('meter').classList.toggle('k-ok', st.solved);
      $('win').hidden = !st.solved;
      if (st.solved) {
        const js = CASES[st.caseIx], missed = x => bit(x, js) && !says(run.sieve.hy, x);
        let after = 0; for (let x = 0; x < N; x++) if (missed(x)) after++;
        $('win-risk').textContent = riskText(run.risk);
        $('win-before').textContent = num(run.mem.wrong);
        countDown($('win-after'), run.mem.wrong, after);
        $('inbox').setAttribute('aria-label', `This week’s 1,024 fresh messages: ${num(run.mem.wrong)} junk messages reached the inbox with version one, ${num(after)} now.`);
        document.dispatchEvent(new CustomEvent('k:solved', { detail: 'act2' }));
      }
      $('stage').setAttribute('aria-label', run ? `${cls.name}, ${labelsText(run.m)}: ${run.sieve ? `${run.sieve.alive.length} consistent hypotheses` : 'the memorizer'}, true risk ${run.sieve && !run.sieve.hy ? 'none' : riskText(run.risk)}.` : `${cls.name}: no examples yet.`);
    }

    let counting = 0;
    const inbox = K.stage($('inbox'), s => {
      const { pen, c, w } = s, run = st.run, cell = w / 64;
      if (!K.reduced.matches) pen.rect(0, 0, w, s.h, c.paper2);
      if (!run || !run.sieve || !run.sieve.hy) return;
      const js = CASES[st.caseIx], junk = [];
      for (let x = 0; x < N; x++) if (!run.mem.seen.has(x) && bit(x, js)) junk.push(x);
      junk.sort((a, b) => ((a * 2654435761) >>> 0) % 997 - ((b * 2654435761) >>> 0) % 997);
      if (K.reduced.matches) {
        const gapX = Math.max(16, w * 0.06), gw = (w - gapX) / 2, cs = gw / 32, missed = new Set(junk), top = 18;
        pen.text(gw < 220 ? `version one: ${num(junk.length)}` : `version one: ${num(junk.length)} junk`, 0, 12, { align: 'left', color: c.ink, size: 12 });
        pen.text(gw < 220 ? 'now: 0' : 'now: 0 junk', gw + gapX, 12, { align: 'left', color: c.ink, size: 12 });
        [[0, missed], [gw + gapX, new Set()]].forEach(([ox, set]) => {
          for (let x = 0; x < N; x++) {
            const gx = ox + (x & 31) * cs, gy = top + (x >> 5) * cs;
            if (set.has(x)) pen.rect(gx, gy, cs - 0.5, cs - 0.5, c.accent);
            else pen.rect(gx + cs / 2 - 0.75, gy + cs / 2 - 0.75, 1.5, 1.5, c.soft, 0.5);
          }
        });
        return;
      }
      const shown = new Set(junk.slice(0, st.inboxShown || 0)), gap = cell > 5 ? 1 : 0.5;
      for (let x = 0; x < N; x++) {
        const gx = (x & 63) * cell, gy = (x >> 6) * cell;
        if (shown.has(x)) pen.rect(gx, gy, cell - gap, cell - gap, c.accent);
        else pen.rect(gx + cell / 2 - 1, gy + cell / 2 - 1, 2, 2, c.soft, 0.5);
      }
    }, { height: w => (K.reduced.matches ? Math.round(18 + (w - Math.max(16, w * 0.06)) / 2) : Math.round(w / 4)) });
    function countDown(el, from, to) {
      if (el.dataset.run === String(st.runId)) return;
      el.dataset.run = String(st.runId);
      cancelAnimationFrame(counting);
      const show = v => { el.textContent = num(v); st.inboxShown = v; inbox.redraw(); };
      if (K.reduced.matches) { show(to); return; }
      const t0 = performance.now(), span = 900;
      const step = now => {
        const u = Math.min(1, (now - t0) / span), e = 1 - (1 - u) ** 3;
        show(Math.round(from + (to - from) * e));
        if (u < 1) counting = requestAnimationFrame(step);
      };
      counting = requestAnimationFrame(step);
    }

    function render() {
      if (explore) act1Render(); else act2Render();
      stage.resize();
    }

    if (explore) {
      ['task', 'fix-controls', 'meter', 'another', 'win'].forEach(k => { $(k).hidden = true; });
      const q = document.createElement('div');
      $('ask').replaceChildren(q); $('ask').className = 'k-predict';
      K.ask(q, {
        question: 'Predict first. The memorizer has seen 100 examples and is consistent with all of them. About what fraction of fresh messages will it get wrong?',
        options: [['0', 'About 0%'], ['10', 'About 10%'], ['45', 'About 45%'], ['90', 'About 90%']], right: '45',
        hints: {
          0: 'It is right on the messages it saw. How many of the 1,024 can 100 examples cover, and what does it say for the rest?',
          10: 'Count the unseen messages first: at most 100 of the 1,024 are seen. It says keep for every unseen one. How many of those are junk?',
          90: 'It says keep for every message it has not seen, and junk is half of all messages. Can it be wrong on more than half?',
        },
        explain: 'On average, 100 random examples cover about 95 different messages (the readout below gives this run’s exact count), so about 929 are unseen. The memorizer says keep for all of them, and half are junk: 0.5 × 929 / 1,024, about 45%. Now move the slider and watch how little it moves.',
        onSolved: () => { st.predicted = true; animate(st.m); render(); },
        focusAfter: () => $('m'),
      });
      $('m').addEventListener('input', () => { st.m = Number($('m').value); st.p = st.m + 0.01; st.revealed = true; render(); });
      $('resample').addEventListener('click', () => {
        st.resamples++; st.seed = SEED + 7919 * resets + 1000 * st.resamples;
        const r = rng(st.seed); st.xs = draw(r, M_MAX);
        animate(st.m); render();
      });
      $('pin').addEventListener('click', () => {
        const r = act1();
        st.pinned = { m: st.m, seed: st.seed, mem: r.mem.risk, one: r.one.risk, alive: new Set(r.one.alive), name: `${hName(r.one.hy)}${st.revealed ? (r.one.truth ? ', the truth' : ', an impostor') : ''}` };
        render();
      });
      $('reset').addEventListener('click', () => {
        resets++;
        Object.assign(st, fresh1(), { m: 100, pinned: null, resamples: 0, revealed: false });
        st.p = st.m + 0.01; render(); $('reset').focus();
      });
    } else {
      ['explore-controls', 'resample', 'pin', 'jump', 'key-pin'].forEach(k => { $(k).hidden = true; });
      ['task', 'fix-controls', 'meter'].forEach(k => { $(k).hidden = false; });
      $('reset').classList.add('quiet');
      const q = document.createElement('div');
      $('ask').replaceChildren(q); $('ask').className = 'k-predict';
      K.ask(q, {
        question: 'One prediction before you start. Suppose you pick “one of the first three”, and suppose this week’s junk feature were feature 7. Your users label plenty of messages. What would the learner return?',
        options: [['truth', 'The truth, h₇'], ['impostor', 'An impostor'], ['none', 'No consistent hypothesis'], ['mem', 'The memorizer’s answer']], right: 'none',
        hints: {
          truth: 'The learner only ever returns a hypothesis from its class. Is h₇ one of h₁, h₂, h₃?',
          impostor: 'With one or two examples, maybe. But each example rules out each wrong feature with chance one half. What is left after a dozen?',
          mem: 'The learner never strays outside its class, and the memorizer’s hypothesis is not one of h₁, h₂, h₃.',
        },
        explain: 'In that week, features 1 to 3 would have nothing to do with junk, so each one would disagree with about half the examples. After a handful of examples all three would be ruled out and the learner would have nothing to return. That is the rat’s failure: more labels cannot help.',
        onSolved: () => { st.predicted = true; render(); },
        focusAfter: () => host.querySelector('[data-cls="ten"]'),
      });
      host.querySelectorAll('[data-cls]').forEach(b => b.addEventListener('click', () => { st.cls = b.dataset.cls; st.run = null; st.solved = false; render(); }));
      $('labels').addEventListener('input', () => { st.labels = Number($('labels').value); st.run = null; st.solved = false; render(); });
      $('train').addEventListener('click', () => { trainOnce(); render(); });
      $('another').addEventListener('click', () => {
        st.caseIx = (st.caseIx + 1) % CASES.length;
        Object.assign(st, { cls: 'any', labels: 5, run: null, solved: false });
        render(); $('train').focus();
      });
      $('reset').addEventListener('click', () => {
        Object.assign(st, { cls: 'any', labels: 5, run: null, solved: false });
        render(); $('reset').focus();
      });
    }
    st.span = 1;
    render();
    subify(host);
    new MutationObserver(() => subify(host)).observe(host, { childList: true, subtree: true, characterData: true });
    return { host };
  }

  mount(document.getElementById('lab'), 'explore');
  document.getElementById('start-act2').addEventListener('click', () => {
    const m = document.getElementById('act2-mount'), host = document.createElement('div');
    host.className = 'lab'; host.id = 'lab2';
    m.replaceWith(host);
    mount(host, 'fix');
    host.querySelector('.k-ask button').focus();
  });
})();
