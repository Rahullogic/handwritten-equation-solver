"""Solve / evaluate a LaTeX equation string (the model's output) with SymPy.

    python -m v3.solve "\\frac{1}{2}+\\frac{1}{3}"      -> evaluates to 5/6
    python -m v3.solve "2x+3=11"                         -> x = 4
    python -m v3.solve "x^2-5x+6=0"                      -> x in {2, 3}

Handles: arithmetic, equations in one or more variables, simple checks like "2+2=4",
derivatives (\\frac{d}{dx}f, d(f)/dx) and indefinite integrals (\\int f dx).
Needs:  pip install sympy antlr4-python3-runtime==4.11
"""
import argparse
import re

import sympy as sp
from sympy.parsing.latex import parse_latex

_STRIP = [r"\left", r"\right", r"\displaystyle", r"\!", r"\,", r"\;", r"\:", r"\ ", r"\quad", r"\qquad"]
_SWAP = {r"\dfrac": r"\frac", r"\tfrac": r"\frac", r"\times": r"\cdot", r"\ast": r"\cdot",
         r"\div": "/", r"\lbrace": "{", r"\rbrace": "}", r"\lbrack": "[", r"\rbrack": "]"}


def clean_latex(s):
    """Normalise handwriting-model LaTeX into something SymPy's parser accepts."""
    for t in _STRIP:
        s = s.replace(t, "")
    for a, b in _SWAP.items():
        s = s.replace(a, b)
    s = s.replace("[", "(").replace("]", ")")
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s.rstrip("?").strip()


def _parse(expr):
    return parse_latex(expr)


def _fmt(x):
    return sp.sstr(sp.simplify(x)) if not isinstance(x, (list, tuple, set)) else str(x)


def _strip_outer_parens(s):
    """'(abc)' -> 'abc' when the first '(' closes at the very end."""
    while s.startswith("(") and s.endswith(")"):
        depth = 0
        for i, ch in enumerate(s):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(s) - 1:
                return s
        s = s[1:-1].strip()
    return s


_DERIV = [re.compile(r"^\\frac\{d\}\{d([a-zA-Z])\}(.+)$"),               # \frac{d}{dx} f
          re.compile(r"^\\frac\{d\s*(.+)\}\{d([a-zA-Z])\}$"),            # \frac{d f}{dx}
          re.compile(r"^d\s*(\(.+\)|\S+)\s*/\s*d([a-zA-Z])$")]            # d(f)/dx
_INTEG = re.compile(r"^\\int\s*(.+?)\s*d([a-zA-Z])$")                       # \int f dx (indefinite)


def _calculus(s):
    """Return (kind, detail, answer) for derivatives / indefinite integrals, else None."""
    s = _strip_outer_parens(s)
    for k, rx in enumerate(_DERIV):
        m = rx.match(s)
        if m:
            f, v = (m.group(2), m.group(1)) if k == 0 else (m.group(1), m.group(2))
            f = _strip_outer_parens(f.strip())
            var = sp.Symbol(v)
            res = sp.simplify(sp.diff(_parse(f), var))
            return "derivative", f"d/d{v} [{sp.sstr(_parse(f))}] = {sp.sstr(res)}", res
    m = _INTEG.match(s)
    if m:
        var = sp.Symbol(m.group(2))
        f = _parse(m.group(1))
        res = sp.integrate(f, var)
        return "integral", f"integral of {sp.sstr(f)} d{m.group(2)} = {sp.sstr(res)} + C", res
    return None


def solve_latex(latex):
    """Return a dict: {'latex', 'cleaned', 'kind', 'answer', 'detail'}; never raises on bad input."""
    out = {"latex": latex, "cleaned": clean_latex(latex), "kind": "error", "answer": None, "detail": ""}
    s = out["cleaned"]
    try:
        calc = _calculus(s)
        if calc:
            out.update(kind=calc[0], answer=calc[2], detail=calc[1])
            return out
        # Systems like "x+y=3, x-y=1" (also ';' or '\\')
        parts = [p for p in re.split(r"\\\\|;|,(?![^{]*\})", s) if p.strip()]
        if len(parts) > 1 and all("=" in p for p in parts):
            eqs = []
            for p in parts:
                l, r = p.split("=", 1)
                eqs.append(sp.Eq(_parse(l), _parse(r)))
            syms = sorted(set().union(*[e.free_symbols for e in eqs]), key=str)
            sol = sp.solve(eqs, syms, dict=True)
            out.update(kind="system", answer=sol, detail=str(sol))
            return out

        if s.count("=") > 1:                                  # e.g. "2+3=5=5" or "2+3 = 5 = ..." -> chain check
            vals = [_parse(p) for p in s.split("=") if p.strip()]
            out.update(kind="chain", answer=all(sp.simplify(v - vals[0]) == 0 for v in vals))
            out["detail"] = "all sides equal" if out["answer"] else "sides are not equal"
            return out

        if "=" in s:
            lhs, rhs = s.split("=", 1)
            if not rhs.strip():                               # "3+4="  -> evaluate
                expr = _parse(lhs)
                val = sp.nsimplify(sp.simplify(expr))
                out.update(kind="evaluate", answer=val, detail=f"{sp.sstr(val)}" + _approx(val))
                return out
            l, r = _parse(lhs), _parse(rhs)
            syms = sorted((l - r).free_symbols, key=str)
            if not syms:                                      # "2+2=4"  -> true / false
                ok = sp.simplify(l - r) == 0
                out.update(kind="check", answer=bool(ok),
                           detail=f"{sp.sstr(sp.simplify(l))} = {sp.sstr(sp.simplify(r))}  ->  {'TRUE' if ok else 'FALSE'}")
                return out
            if len(syms) == 1:
                x = syms[0]
                sol = sp.solveset(sp.Eq(l, r), x, domain=sp.S.Complexes)
                real = sp.solveset(sp.Eq(l, r), x, domain=sp.S.Reals)
                out.update(kind="solve", answer=real if real != sp.S.EmptySet else sol,
                           detail=f"{x} = {_sol_str(real if real != sp.S.EmptySet else sol)}")
                return out
            sol = sp.solve(sp.Eq(l, r), syms[-1], dict=False)  # several unknowns: solve for the last one
            out.update(kind="solve", answer=sol, detail=f"{syms[-1]} = {sol}")
            return out

        expr = _parse(s)                                      # no '=': evaluate / simplify
        if expr.free_symbols:
            val = sp.simplify(expr)
            out.update(kind="simplify", answer=val, detail=sp.sstr(val))
        else:
            val = sp.nsimplify(sp.simplify(expr))
            out.update(kind="evaluate", answer=val, detail=f"{sp.sstr(val)}" + _approx(val))
        return out
    except Exception as e:                                    # parse failure / unsupported syntax
        out["detail"] = f"could not solve: {type(e).__name__}: {e}"
        return out


def _approx(val):
    try:
        if val.is_Rational and not val.is_Integer:
            return f"  (~ {float(val):.6g})"
        if not val.is_Number:
            return f"  (~ {float(val):.6g})"
    except Exception:
        pass
    return ""


def _sol_str(sol):
    if isinstance(sol, sp.FiniteSet):
        return ", ".join(sp.sstr(v) for v in sorted(sol, key=lambda v: (not v.is_real, str(v))))
    return sp.sstr(sol)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("latex")
    args = ap.parse_args()
    r = solve_latex(args.latex)
    print(f"{r['kind']}: {r['detail']}")


if __name__ == "__main__":
    main()
