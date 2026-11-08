# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
checker.py - independent type checker for nanoGentzen proof certificates.

No imports from the rest of nanoGentzen: it has its own formula parser and its own typing rules, so a
certificate can be checked without trusting the engine that produced it.

Certificate (JSON):
  {"context": {"h1": "kyc_passed", "h2": "((kyc_passed & is_adult) => approved)", ...},
   "type":    "approved",
   "term":    <term AST>}

Simply typed lambda calculus with products, sums and falsum (intuitionistic propositional logic):
  ["var", x]                         x : A                      if x : A in context
  ["lam", x, A, M]                   A => B                     if M : B with x : A
  ["app", M, N]                      B                          if M : A => B, N : A   (~A is A => ⊥)
  ["pair", M, N]                     A & B
  ["fst", M] / ["snd", M]            A / B                      if M : A & B
  ["inl", M, B] / ["inr", M, A]      A | B
  ["case", M, x, N, y, P]            C                          if M : A | B, N : C with x : A, P : C with y : B
  ["abort", M, C]                    C                          if M : ⊥
  ["let", x, A, M, N]                C                          if M : A and N : C with x : A

Usage:  python checker.py certificate.json      -> prints OK or the first type error, exit code 0 / 1
        check_certificate(cert_dict)              -> (True, None) or (False, "reason")
"""
import json
import re
import sys

BOT = ("bot",)


# ---------------------------------------------------------------- formulas
def parse_formula(s: str):
    toks = re.findall(r"=>|<->|[()&|~]|⊥|[A-Za-z_][A-Za-z0-9_]*|0", s)
    if "".join(toks) != re.sub(r"\s+", "", s):
        raise ValueError(f"cannot read formula {s!r}")
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take(t=None):
        tok = peek()
        if tok is None or (t is not None and tok != t):
            raise ValueError(f"expected {t!r} in {s!r}")
        pos[0] += 1
        return tok

    def imp():
        left = disj()
        if peek() == "=>":
            take()
            return ("imp", left, imp())
        return left

    def disj():
        node = conj()
        while peek() == "|":
            take(); node = ("or", node, conj())
        return node

    def conj():
        node = neg()
        while peek() == "&":
            take(); node = ("and", node, neg())
        return node

    def neg():
        if peek() == "~":
            take()
            return ("imp", neg(), BOT)
        return atom()

    def atom():
        tok = take()
        if tok == "(":
            f = imp(); take(")")
            return f
        if tok in ("⊥", "0"):
            return BOT
        if re.match(r"[A-Za-z_]", tok):
            return ("var", tok)
        raise ValueError(f"unexpected {tok!r} in {s!r}")

    f = imp()
    if peek() is not None:
        raise ValueError(f"trailing input in {s!r}")
    return f


OPS = {"and": "&", "or": "|", "imp": "=>"}


def show(f) -> str:
    if f == BOT:
        return "⊥"
    if f[0] == "var":
        return f[1]
    if f[0] == "imp" and f[2] == BOT:
        return f"~{show(f[1])}"
    return f"({show(f[1])} {OPS[f[0]]} {show(f[2])})"


# ---------------------------------------------------------------- typing
class TypeError_(Exception):
    pass


def infer(t, env):
    k = t[0]
    if k == "var":
        if t[1] not in env:
            raise TypeError_(f"unbound variable {t[1]}")
        return env[t[1]]
    if k == "lam":
        a = parse_formula(t[2])
        return ("imp", a, infer(t[3], {**env, t[1]: a}))
    if k == "app":
        f = infer(t[1], env)
        a = infer(t[2], env)
        if f[0] != "imp":
            raise TypeError_(f"applying a non-function of type {show(f)}")
        if f[1] != a:
            raise TypeError_(f"argument has type {show(a)}, expected {show(f[1])}")
        return f[2]
    if k == "pair":
        return ("and", infer(t[1], env), infer(t[2], env))
    if k in ("fst", "snd"):
        p = infer(t[1], env)
        if p[0] != "and":
            raise TypeError_(f"{k} of non-pair {show(p)}")
        return p[1] if k == "fst" else p[2]
    if k == "inl":
        return ("or", infer(t[1], env), parse_formula(t[2]))
    if k == "inr":
        return ("or", parse_formula(t[2]), infer(t[1], env))
    if k == "case":
        s = infer(t[1], env)
        if s[0] != "or":
            raise TypeError_(f"case on non-sum {show(s)}")
        c1 = infer(t[3], {**env, t[2]: s[1]})
        c2 = infer(t[5], {**env, t[4]: s[2]})
        if c1 != c2:
            raise TypeError_(f"case branches differ: {show(c1)} vs {show(c2)}")
        return c1
    if k == "abort":
        if infer(t[1], env) != BOT:
            raise TypeError_("abort of a term that is not of type ⊥")
        return parse_formula(t[2])
    if k == "let":
        a = parse_formula(t[2])
        b = infer(t[3], env)
        if a != b:
            raise TypeError_(f"let {t[1]}: declared {show(a)}, bound term has {show(b)}")
        return infer(t[4], {**env, t[1]: a})
    raise TypeError_(f"unknown term constructor {k!r}")


def free_vars(t, bound=frozenset()):
    k = t[0]
    if k == "var":
        return set() if t[1] in bound else {t[1]}
    if k == "lam":
        return free_vars(t[3], bound | {t[1]})
    if k == "case":
        return free_vars(t[1], bound) | free_vars(t[3], bound | {t[2]}) | free_vars(t[5], bound | {t[4]})
    if k == "let":
        return free_vars(t[3], bound) | free_vars(t[4], bound | {t[1]})
    out = set()
    for x in t[1:]:
        if isinstance(x, list):
            out |= free_vars(x, bound)
    return out


def check_certificate(cert: dict):
    try:
        env = {h: parse_formula(f) for h, f in cert["context"].items()}
        goal = parse_formula(cert["type"]) if cert.get("type") not in (None, "", "0") else BOT
        got = infer(cert["term"], env)
        if got != goal:
            return False, f"term has type {show(got)}, certificate claims {show(goal)}"
        return True, None
    except (TypeError_, ValueError, KeyError, IndexError) as e:
        return False, str(e)


if __name__ == "__main__":
    cert = json.load(open(sys.argv[1], encoding="utf-8"))
    ok, err = check_certificate(cert)
    print("OK" if ok else f"REJECTED: {err}")
    sys.exit(0 if ok else 1)
