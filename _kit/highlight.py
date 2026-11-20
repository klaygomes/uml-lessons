import builtins
import html
import io
import keyword
import re
import token
import tokenize

BUILTINS = {name for name in dir(builtins) if not name.startswith("_")}
STRINGS = {token.STRING} | {getattr(token, n) for n in ("FSTRING_START", "FSTRING_MIDDLE", "FSTRING_END") if hasattr(token, n)}
PSEUDO_KEYWORDS = {"for", "if", "else", "elif", "return", "while", "do", "in", "to", "and", "or", "not", "then", "end", "each", "repeat", "until", "is", "empty"}
PSEUDO = re.compile(r"(?P<arrow>←|→|≠|≤|≥)|(?P<num>\b\d+(?:\.\d+)?\b)|(?P<word>[A-Za-z_][\w]*)|(?P<other>.)", re.S)


def span(kind, text):
    return f'<span class="c-{kind}">{html.escape(text)}</span>'


def python(source):
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    out, pos, prev = [], 0, None
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return html.escape(source)
    for i, tok in enumerate(tokens):
        if tok.type in (token.ENDMARKER, token.NEWLINE, token.NL, token.INDENT, token.DEDENT):
            continue
        start = offsets[tok.start[0] - 1] + tok.start[1]
        end = offsets[tok.end[0] - 1] + tok.end[1]
        out.append(html.escape(source[pos:start]))
        text = source[start:end]
        nxt = next((t for t in tokens[i + 1:] if t.type not in (token.NL, token.NEWLINE)), None)
        if tok.type == token.NAME and keyword.iskeyword(text):
            kind = "keyword"
        elif tok.type == token.NAME and prev is not None and prev.string in ("def", "class"):
            kind = "function"
        elif tok.type == token.NAME and nxt is not None and nxt.string == "(":
            kind = "builtin" if text in BUILTINS else "call"
        elif tok.type == token.NAME and text in BUILTINS:
            kind = "builtin"
        elif tok.type in STRINGS:
            kind = "string"
        elif tok.type == token.NUMBER:
            kind = "number"
        elif tok.type == token.COMMENT:
            kind = "comment"
        elif tok.type == token.OP and text == "...":
            kind = "comment"
        else:
            kind = None
        out.append(span(kind, text) if kind else html.escape(text))
        pos, prev = end, tok
    out.append(html.escape(source[pos:]))
    return "".join(out)


def pseudo(source):
    out, after_assign = [], False
    for m in PSEUDO.finditer(source):
        text = m.group(0)
        if m.group("arrow"):
            out.append(span("keyword", text))
        elif m.group("num"):
            out.append(span("number", text))
        elif m.group("word"):
            rest = source[m.end():m.end() + 1]
            if text in PSEUDO_KEYWORDS or text.isupper() and len(text) > 1:
                out.append(span("keyword", text))
            elif rest == "(" and text[:1].isupper():
                out.append(span("function", text))
            else:
                out.append(html.escape(text))
        else:
            out.append(html.escape(text))
    return "".join(out)


BLOCK = re.compile(r'<pre class="code"( data-lang="([\w-]*)")?><code>(.*?)</code></pre>', re.S)


def highlight(body):
    def one(m):
        lang, source = (m.group(2) or "").lower(), html.unescape(m.group(3))
        coloured = python(source) if lang in ("python", "py") else pseudo(source)
        attr = f' data-lang="{lang}"' if lang else ""
        return f'<pre class="code"{attr}><code>{coloured}</code></pre>'
    return BLOCK.sub(one, body)
