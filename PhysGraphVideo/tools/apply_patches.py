# 文件：tools/apply_patches.py
# 用法：python -m tools.apply_patches
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    (ROOT / "tools" / "train.py",      "train"),
    (ROOT / "tools" / "evaluate.py",   "val"),
    (ROOT / "tools" / "export_viz.py", "val"),
]

def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8")

def _write(p: Path, s: str) -> None:
    p.write_text(s, encoding="utf-8")

def _line_start(s: str, idx: int) -> int:
    j = s.rfind("\n", 0, idx)
    return 0 if j < 0 else j + 1

def _guess_var_and_indent(s: str, call_pos: int) -> tuple[str, str]:
    ls = _line_start(s, call_pos)
    line = s[ls:call_pos]
    # 形如 "    train_loader = "
    var = ""
    i = 0
    while i < len(line) and line[i].isspace():
        i += 1
    indent = line[:i]
    eq = line.find("=")
    if eq != -1:
        var = line[i:eq].strip()
    return var, indent

def _find_call_span(s: str, start: int) -> tuple[int, int]:
    """
    输入：s, 'm' 的位置（即 'make_dataloader(' 的 m）
    返回：整段调用的 [start, end) 区间，end 指向**已包含**右括号后一位。
    若后面紧跟一个多余的')'，自动吞掉它。
    """
    i = start
    level = 0
    in_str = None
    while i < len(s):
        ch = s[i]
        if in_str:
            if ch == in_str and s[i-1] != "\\":
                in_str = None
        else:
            if ch in ("'", '"'):
                in_str = ch
            elif ch == "(":
                level += 1
            elif ch == ")":
                level -= 1
                if level == 0:
                    end = i + 1
                    # 吞掉紧跟的一个多余 ')'
                    j = end
                    while j < len(s) and s[j] in " \t\r":
                        j += 1
                    if j < len(s) and s[j] == ")":
                        end = j + 1
                    return start, end
        i += 1
    raise RuntimeError("Unbalanced parentheses while scanning make_dataloader call.")

def _decide(split_default: str, var: str) -> tuple[str, str, str]:
    v = (var or "").lower()
    if "train" in v: sp, tr = "train", "True"
    elif "test" in v: sp, tr = "test", "False"
    elif "val" in v or "valid" in v: sp, tr = "val", "False"
    else: sp, tr = split_default, ("True" if split_default == "train" else "False")
    bs = "8" if sp == "train" else "1"
    return sp, tr, bs

def _build_block(indent: str, var: str, split: str, is_train: str, bs: str) -> str:
    lhs = (var + " = ") if var else ""
    return (
f"""{indent}{lhs}make_dataloader(
{indent}    name=cfg.data.name,
{indent}    root=cfg.data.root,
{indent}    split={split!r},
{indent}    T_in=cfg.data.t_in,
{indent}    T_out=cfg.data.t_out,
{indent}    size=cfg.data.image_size,
{indent}    channels=getattr(cfg.data, 'channels', 3),
{indent}    stride=getattr(cfg.data, 'stride', 1),
{indent}    batch_size=getattr(cfg.data, 'batch_size', {bs}),
{indent}    num_workers=getattr(cfg.data, 'num_workers', 0),
{indent}    pin_memory=getattr(cfg.data, 'pin_memory', False),
{indent}    is_train={is_train},
{indent})"""
    )

def patch_one(path: Path, split_default: str) -> int:
    src = _read(path)
    out = []
    i = 0
    changes = 0
    key = "make_dataloader("
    while True:
        j = src.find(key, i)
        if j == -1:
            out.append(src[i:])
            break
        # 追加前半段
        out.append(src[i:j])
        var, indent = _guess_var_and_indent(src, j)
        beg, end = _find_call_span(src, j)
        split, is_train, bs = _decide(split_default, var)
        block = _build_block(indent, var, split, is_train, bs)
        out.append(block)
        i = end
        changes += 1
    new_code = "".join(out)
    if changes:
        _write(path, new_code)
        # 语法检查
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            # 打印附近代码帮助定位
            ctx = new_code.splitlines()
            ln = max(e.lineno - 3, 0)
            rn = min(e.lineno + 2, len(ctx))
            snippet = "\n".join(f"{k+1:>5}: {ctx[k]}" for k in range(ln, rn))
            raise SystemExit(
                f"[ERROR] {path.name} syntax error: {e}\n----- context -----\n{snippet}\n--------------------"
            )
    return changes

def main():
    total = 0
    for p, default_split in TARGETS:
        if not p.exists():
            print(f"[MISS ] {p}")
            continue
        c = patch_one(p, default_split)
        print(f"[PATCH] {p} -> {c} call(s) patched.")
        total += c
    print(f"[OK] patches applied. total={total}")

if __name__ == "__main__":
    main()
