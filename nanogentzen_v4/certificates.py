# Vendored from the nanoGentzen v4 model folder by tools/vendor_v4_engine.py. Do not edit by hand.
"""
certificates.py (Phase 3) - proof terms from G4ip proof trees.

certify(tree) turns a verified proof tree into a typed lambda term (Curry-Howard) with an explicit
typing context: every root premise is a named hypothesis h1..hn, every formula the proof derives is a
`let`-bound variable, and unused bindings are removed. The result is checked with checker.py, which shares
no code with the kernel. relevant premises = the hypotheses that occur free in the term.

Term constructors are listed in checker.py. Translation of the G4ip rules (pivot term t):
  AXIOM        identity: the hypothesis | ⊥ in context: abort | A, ~A: abort (~A A)
  R_IMP/R_NOT  lam x:A. M            R_AND  pair       R_OR_1/2  inl / inr
  L_AND        let a = fst t; let b = snd t
  L_OR         case t of inl x => M | inr y => N
  L_IMP_ATOM   let b = t p
  L_IMP_AND    let f = lam c. lam d. t (c, d)            L_NOT_AND same, type C => ~D
  L_IMP_OR     let f = lam c. t (inl c); let g = lam d. t (inr d)      L_NOT_OR same
  L_IMP_IMP    let b = t (lam c. let e = (lam d. t (lam _. d)) in M)   then N with b
  L_IMP_NOT    let b = t (lam c. M)
  L_NOT_IMP    t (lam c. let e = (lam d. t (lam _. d)) in M)  : ⊥
  L_NOT_NOT    t (lam c. M)  : ⊥
"""
import itertools
from typing import Any, Dict, List, Optional

from .kernel import BOT, And, Bot, Imp, Not, Or, Sequent, Var, g4ip_prove, normalize_formula
from . import checker as _checker


def _parse(s):
    from .parser import parse_symbolic_sequent
    return parse_symbolic_sequent(s)


def _k(f) -> str:
    return normalize_formula(f).to_str()


class CertificateError(Exception):
    pass


def certify(tree: dict) -> Dict[str, Any]:
    """Certificate for a proof tree: context, type, term (AST), term_str, free_hypotheses, checked."""
    root = _parse(tree["sequent"])
    if root is None:
        raise CertificateError("root sequent does not parse")
    counter = itertools.count(1)
    fresh = lambda p="x": f"{p}{next(counter)}"  # noqa: E731
    context, ctx = {}, {}
    for i, f in enumerate(root.gamma, 1):
        context[f"h{i}"] = f.to_str()
        ctx[_k(f)] = ["var", f"h{i}"]

    def let(F, expr, ctx, cont):
        x = fresh("d")
        return ["let", x, _k(F), expr, cont({**ctx, _k(F): ["var", x]})]

    def to_goal(s, bot_term):
        return ["abort", bot_term, s.delta[0].to_str()] if s.delta else bot_term

    def T(node, ctx):
        s = _parse(node["sequent"])
        rule, br = node["rule"], node.get("branches", [])
        if s is None:
            raise CertificateError(f"node does not parse: {node['sequent']}")
        G = s.delta[0] if s.delta else None
        if rule == "AXIOM":
            if G is not None and G in set(s.gamma):
                return ctx[_k(G)]
            if BOT in set(s.gamma):
                return to_goal(s, ctx[_k(BOT)])
            for f in s.gamma:
                if isinstance(f, Not) and f.inner in set(s.gamma):
                    return to_goal(s, ["app", ctx[_k(f)], ctx[_k(f.inner)]])
            raise CertificateError(f"AXIOM node is not an axiom: {node['sequent']}")
        if rule in ("R_IMP", "R_NOT"):
            a = G.left if rule == "R_IMP" else G.inner
            x = fresh("x")
            return ["lam", x, _k(a), T(br[0], {**ctx, _k(a): ["var", x]})]
        if rule == "R_AND":
            return ["pair", T(br[0], ctx), T(br[1], ctx)]
        if rule == "R_OR_1":
            return ["inl", T(br[0], ctx), _k(G.right)]
        if rule == "R_OR_2":
            return ["inr", T(br[0], ctx), _k(G.left)]
        name, idx = rule.rsplit("_", 1)
        p = s.gamma[int(idx)]
        t = ctx[_k(p)]
        if name == "L_AND":
            return let(p.left, ["fst", t], ctx, lambda c: let(p.right, ["snd", t], c, lambda c2: T(br[0], c2)))
        if name == "L_OR":
            x, y = fresh("x"), fresh("y")
            return ["case", t, x, T(br[0], {**ctx, _k(p.left): ["var", x]}), y, T(br[1], {**ctx, _k(p.right): ["var", y]})]
        if name in ("L_IMP_BOT", "L_NOT_BOT", "L_CONTR"):
            return T(br[0], ctx)
        if name == "L_IMP_ATOM":
            return let(p.right, ["app", t, ctx[_k(p.left)]], ctx, lambda c: T(br[0], c))
        if name in ("L_IMP_AND", "L_NOT_AND"):
            inner = p.left if name == "L_IMP_AND" else p.inner
            C, D = inner.left, inner.right
            B = p.right if name == "L_IMP_AND" else BOT
            c, d = fresh("c"), fresh("d")
            expr = ["lam", c, _k(C), ["lam", d, _k(D), ["app", t, ["pair", ["var", c], ["var", d]]]]]
            F = Imp(C, Imp(D, B)) if name == "L_IMP_AND" else Imp(C, Not(D))
            return let(F, expr, ctx, lambda cc: T(br[0], cc))
        if name in ("L_IMP_OR", "L_NOT_OR"):
            inner = p.left if name == "L_IMP_OR" else p.inner
            C, D = inner.left, inner.right
            B = p.right if name == "L_IMP_OR" else BOT
            c, d = fresh("c"), fresh("d")
            f1 = ["lam", c, _k(C), ["app", t, ["inl", ["var", c], _k(D)]]]
            f2 = ["lam", d, _k(D), ["app", t, ["inr", ["var", d], _k(C)]]]
            return let(Imp(C, B), f1, ctx, lambda c1: let(Imp(D, B), f2, c1, lambda c2: T(br[0], c2)))
        if name in ("L_IMP_IMP", "L_NOT_IMP"):
            inner = p.left if name == "L_IMP_IMP" else p.inner
            C, D = inner.left, inner.right
            B = p.right if name == "L_IMP_IMP" else BOT
            c, d, u = fresh("c"), fresh("d"), fresh("u")
            d_to_b = ["lam", d, _k(D), ["app", t, ["lam", u, _k(C), ["var", d]]]]
            body = let(Imp(D, B), d_to_b, {**ctx, _k(C): ["var", c]}, lambda cc: T(br[0], cc))
            applied = ["app", t, ["lam", c, _k(C), body]]
            if name == "L_NOT_IMP":
                return to_goal(s, applied)
            return let(B, applied, ctx, lambda cc: T(br[1], cc))
        if name in ("L_IMP_NOT", "L_NOT_NOT"):
            C = p.left.inner if name == "L_IMP_NOT" else p.inner.inner
            c = fresh("c")
            applied = ["app", t, ["lam", c, _k(C), T(br[0], {**ctx, _k(C): ["var", c]})]]
            if name == "L_NOT_NOT":
                return to_goal(s, applied)
            return let(p.right, applied, ctx, lambda cc: T(br[1], cc))
        if name == "L_IMP":                       # legacy LJ rule
            return let(p.right, ["app", t, T(br[0], ctx)], ctx, lambda cc: T(br[1], cc))
        if name == "L_NOT":                       # legacy LJ rule
            return to_goal(s, ["app", t, T(br[0], ctx)])
        raise CertificateError(f"no translation for rule {rule}")

    term = _dce(T(tree, ctx))
    cert = {"context": context, "type": root.delta[0].to_str() if root.delta else "⊥", "term": term}
    ok, err = _checker.check_certificate(cert)
    fv = _checker.free_vars(term)
    cert.update(term_str=pretty(term), checked=ok, check_error=err,
                free_hypotheses=[h for h in context if h in fv])
    return cert


def _dce(t):
    """Removes let-bindings whose variable is not used."""
    if not isinstance(t, list):
        return t
    if t[0] == "let":
        body = _dce(t[4])
        if t[1] not in _checker.free_vars(body):
            return body
        return ["let", t[1], t[2], _dce(t[3]), body]
    return [t[0]] + [(_dce(x) if isinstance(x, list) else x) for x in t[1:]]


def pretty(t) -> str:
    k = t[0]
    if k == "var":
        return t[1]
    if k == "lam":
        return f"(λ{t[1]}:{t[2]}. {pretty(t[3])})"
    if k == "app":
        return f"({pretty(t[1])} {pretty(t[2])})"
    if k == "pair":
        return f"⟨{pretty(t[1])}, {pretty(t[2])}⟩"
    if k in ("fst", "snd"):
        return f"{k}({pretty(t[1])})"
    if k in ("inl", "inr"):
        return f"{k}[{t[2]}]({pretty(t[1])})"
    if k == "case":
        return f"(case {pretty(t[1])} of inl {t[2]} ⇒ {pretty(t[3])} | inr {t[4]} ⇒ {pretty(t[5])})"
    if k == "abort":
        return f"abort[{t[2]}]({pretty(t[1])})"
    if k == "let":
        return f"let {t[1]}:{t[2]} = {pretty(t[3])} in {pretty(t[4])}"
    return str(t)


def relevance(root: Sequent, cert: Dict[str, Any], minimal: bool = True, deadline_s: float = 0.2) -> Dict[str, Any]:
    """Premises used by the certified proof (free hypotheses), and an irredundant subset that still proves the goal."""
    prem = [f.to_str() for f in root.gamma]
    used = [cert["context"][h] for h in cert["free_hypotheses"]]
    out = {"relevant_witnesses": used, "unused_premises": [p for p in prem if p not in used],
           "is_strictly_relevant": len(used) == len(prem),
           "relevance_ratio": round(len(used) / len(prem), 4) if prem else 1.0}
    if minimal:
        keep = [f for f in root.gamma if f.to_str() in used]
        for f in list(keep):
            trial = [g for g in keep if g != f]
            tree, to = g4ip_prove(Sequent(tuple(trial), root.delta), deadline_s=deadline_s)
            if tree is not None:
                keep = trial
        out["minimal_witnesses"] = [f.to_str() for f in keep]
    return out
