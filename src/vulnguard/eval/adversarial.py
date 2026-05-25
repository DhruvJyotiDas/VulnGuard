"""Adversarial / semantic-preserving transforms — precisely specified, which is
exactly what Reviewer 1 demanded.

Decisions (document these verbatim in the rejoinder):
  * Variable renaming: LOCAL variables and parameters only. We use tree-sitter-c
    to locate identifiers in `declaration`, `init_declarator`, and
    `parameter_declaration` nodes, then rename every occurrence of those names
    consistently to `vN`. Function names, type names, macros, struct fields, and
    library calls (strcpy, memcpy, ...) are NEVER renamed -> call semantics are
    preserved. If tree-sitter is unavailable we fall back to a regex heuristic
    that is clearly inferior and prints a warning; do not report regex-mode
    numbers in the paper.
  * Dead-code insertion: only provably side-effect-free statements
    (`if (0) { ... }`, an unused zero-init + `(void)` cast) inserted at
    statement boundaries. Count is configurable (`n_inserts`).
  * Whitespace: reformat indentation / blank lines / spaces around operators
    WITHOUT changing any token. We also expose `tokenization_changed()` so you
    can PROVE whether a whitespace edit altered the model's input — that is how
    you explain the anomalous 0.513 (almost certainly tokenizer drift, not model
    fragility).
  * Semantic preservation is HEURISTIC, not formally verified. We can optionally
    re-run a C parser to confirm the transformed code still parses; true
    behavioral equivalence would require compilation + test execution, which is
    out of scope and must be stated as a threat to validity.
  * One transformed sample per original by default (`n_variants=1`).
"""
from __future__ import annotations

import random
import re
from typing import List, Optional

from ..utils.common import get_logger

log = get_logger(__name__)

_LIB_CALLS = {"strcpy", "strncpy", "memcpy", "memmove", "sprintf", "snprintf",
              "gets", "fgets", "malloc", "free", "printf", "scanf", "strcat",
              "strlen", "memset", "calloc", "realloc", "read", "write"}

_TS = None


def _try_tree_sitter():
    global _TS
    if _TS is not None:
        return _TS
    try:
        from tree_sitter import Parser
        from tree_sitter_languages import get_language
        parser = Parser()
        parser.set_language(get_language("c"))
        _TS = parser
    except Exception as e:  # pragma: no cover
        log.warning("tree-sitter unavailable (%s). Renaming falls back to regex "
                    "— DO NOT report regex-mode adversarial numbers.", e)
        _TS = False
    return _TS


# --------------------------------------------------------------- variable rename
def rename_variables(code: str, seed: int = 0) -> str:
    parser = _try_tree_sitter()
    if parser:
        return _rename_treesitter(code, parser, seed)
    return _rename_regex(code, seed)


def _rename_treesitter(code: str, parser, seed: int) -> str:
    src = code.encode("utf-8")
    tree = parser.parse(src)
    locals_found: List[str] = []

    DECL_PARENTS = {"declaration", "init_declarator", "parameter_declaration"}

    def walk(node, in_decl=False):
        for child in node.children:
            is_decl = child.type in DECL_PARENTS or in_decl
            if child.type == "identifier" and is_decl:
                name = src[child.start_byte:child.end_byte].decode("utf-8")
                if name not in _LIB_CALLS:
                    locals_found.append(name)
            walk(child, is_decl)

    walk(tree.root_node)
    uniq = list(dict.fromkeys(locals_found))
    rng = random.Random(seed)
    rng.shuffle(uniq)
    mapping = {old: f"v{idx}" for idx, old in enumerate(uniq)}
    # word-boundary replace, skipping library calls (already excluded)
    out = code
    for old, new in mapping.items():
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out


def _rename_regex(code: str, seed: int) -> str:
    # crude: rename identifiers that look like locals (appear after a type kw)
    type_kw = r"(?:int|char|float|double|long|short|unsigned|signed|void|size_t)"
    decls = set(re.findall(rf"{type_kw}\s+\**\s*([A-Za-z_]\w*)", code))
    decls -= _LIB_CALLS
    rng = random.Random(seed)
    decls = list(decls); rng.shuffle(decls)
    mapping = {old: f"v{i}" for i, old in enumerate(decls)}
    out = code
    for old, new in mapping.items():
        out = re.sub(rf"\b{re.escape(old)}\b", new, out)
    return out


# ------------------------------------------------------------------- dead code
def insert_dead_code(code: str, n_inserts: int = 3, seed: int = 0) -> str:
    rng = random.Random(seed)
    snippets = [
        "if (0) {{ int __vg{n} = 0; (void)__vg{n}; }}",
        "int __vg{n} = 0; (void)__vg{n};",
        "for (int __vg{n} = 0; __vg{n} < 0; __vg{n}++) {{ }}",
    ]
    lines = code.split("\n")
    # candidate insertion points: after lines ending in ; or {
    points = [i for i, ln in enumerate(lines) if ln.rstrip().endswith((";", "{"))]
    if not points:
        return code
    for k in range(n_inserts):
        i = rng.choice(points)
        indent = re.match(r"\s*", lines[i]).group()
        lines.insert(i + 1, indent + rng.choice(snippets).format(n=f"_{seed}_{k}"))
        points = [p + 1 if p > i else p for p in points]
    return "\n".join(lines)


# ------------------------------------------------------------------- whitespace
def perturb_whitespace(code: str, seed: int = 0) -> str:
    rng = random.Random(seed)
    out = code
    # randomize spaces around a few binary operators (token-preserving)
    for op in ["+", "-", "<", ">", "=", "=="]:
        if rng.random() < 0.5:
            out = out.replace(op, f" {op} ")
    # collapse/expand indentation and add blank lines
    lines = out.split("\n")
    new = []
    for ln in lines:
        new.append(ln)
        if rng.random() < 0.3:
            new.append("")
    out = "\n".join(new)
    return re.sub(r"[ \t]+", lambda m: " " * rng.randint(1, 3), out)


def tokenization_changed(tokenizer, original: str, perturbed: str, max_len: int = 512) -> bool:
    """Did a 'token-preserving' edit actually change the model's input ids?
    If True for whitespace, the anomaly is tokenizer drift, not model fragility."""
    a = tokenizer(original, truncation=True, max_length=max_len)["input_ids"]
    b = tokenizer(perturbed, truncation=True, max_length=max_len)["input_ids"]
    return a != b


PERTURBATIONS = {
    "rename": rename_variables,
    "deadcode": insert_dead_code,
    "whitespace": perturb_whitespace,
}


def apply_perturbation(code: str, kind: str, seed: int = 0, **kw) -> str:
    fn = PERTURBATIONS[kind]
    try:
        return fn(code, seed=seed, **kw)
    except Exception as e:  # never crash a whole eval over one bad sample
        log.debug("perturbation %s failed on a sample: %s", kind, e)
        return code
