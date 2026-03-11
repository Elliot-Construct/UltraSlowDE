from __future__ import annotations

import argparse
import re
import subprocess
from datetime import date
from pathlib import Path


def _extract_braced(text: str, start_idx: int) -> tuple[str, int]:
    if start_idx >= len(text) or text[start_idx] != "{":
        raise ValueError("Expected '{' at start_idx")
    depth = 0
    i = start_idx
    out: list[str] = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out), i + 1
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    raise ValueError("Unbalanced braces")


def _extract_command_arg(text: str, command: str) -> str:
    m = re.search(r"\\" + re.escape(command) + r"\s*\{", text)
    if not m:
        return ""
    brace_start = m.end() - 1
    val, _ = _extract_braced(text, brace_start)
    return val.strip()


def _extract_all_command_args(text: str, command: str) -> list[str]:
    vals: list[str] = []
    start = 0
    pat = r"\\" + re.escape(command) + r"\s*\{"
    while True:
        m = re.search(pat, text[start:])
        if not m:
            break
        abs_start = start + m.start()
        brace_start = start + m.end() - 1
        val, end_idx = _extract_braced(text, brace_start)
        vals.append(val.strip())
        start = end_idx
        if start <= abs_start:
            break
    return vals


def _remove_all_command_with_braced_arg(text: str, command: str) -> str:
    out = text
    pat = r"\\" + re.escape(command) + r"\s*\{"
    while True:
        m = re.search(pat, out)
        if not m:
            break
        brace_start = m.end() - 1
        _, end_idx = _extract_braced(out, brace_start)
        out = out[: m.start()] + out[end_idx:]
    return out


def _surname_from_name(name: str) -> str:
    tokens = [tok for tok in re.split(r"\s+", name.strip()) if tok]
    if not tokens:
        return "Author"
    surname = re.sub(r"[^\w\-]", "", tokens[-1])
    return surname or "Author"


def _convert_deluxetable_block(block: str) -> str:
    m_align = re.search(r"\\begin\{deluxetable\*?\}\{([^}]*)\}", block)
    align = m_align.group(1) if m_align else "l"

    caption = ""
    label = ""
    cap_raw = _extract_command_arg(block, "tablecaption")
    if cap_raw:
        m_lbl = re.search(r"\\label\{([^}]*)\}", cap_raw)
        if m_lbl:
            label = m_lbl.group(1)
            cap_raw = re.sub(r"\\label\{[^}]*\}", "", cap_raw).strip()
        caption = cap_raw

    headers = []
    head_raw = _extract_command_arg(block, "tablehead")
    if head_raw:
        headers = _extract_all_command_args(head_raw, "colhead")

    data_rows = ""
    m_data = re.search(r"\\startdata(.*?)\\enddata", block, flags=re.S)
    if m_data:
        data_rows = m_data.group(1).strip()

    comment = _extract_command_arg(block, "tablecomments")

    lines: list[str] = ["\\begin{table}[t]", "\\centering", "\\small"]
    if caption:
        if label:
            lines.append(f"\\caption{{{caption}}}\\label{{{label}}}")
        else:
            lines.append(f"\\caption{{{caption}}}")

    lines.append(r"\resizebox{\columnwidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{align}}}")
    lines.append("\\hline")
    if headers:
        lines.append(" & ".join(headers) + r" \\")
        lines.append("\\hline")
    if data_rows:
        lines.append(data_rows)
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("}")
    if comment:
        lines.append(r"\begin{flushleft}\footnotesize\textit{Note.} " + comment + r"\end{flushleft}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def convert_aas_to_revtex(aas_text: str) -> str:
    out = aas_text

    out = re.sub(
        r"\\documentclass\[[^\]]*\]\{aastex701\}",
        r"\\documentclass[aps,prd,twocolumn,superscriptaddress,nofootinbib,floatfix]{revtex4-2}",
        out,
    )

    # Remove AASTeX-only metadata commands
    out = re.sub(r"^\\shorttitle\{.*?\}\s*$", "", out, flags=re.M)
    out = re.sub(r"^\\shortauthors\{.*?\}\s*$", "", out, flags=re.M)
    out = re.sub(r"^\\NewPageAfterKeywords\s*$", "", out, flags=re.M)

    # Convert deluxetable blocks
    while True:
        m = re.search(r"\\begin\{deluxetable\*?\}\{[^}]*\}", out)
        if not m:
            break
        start = m.start()
        m_end = re.search(r"\\end\{deluxetable\*?\}", out[m.end() :])
        if not m_end:
            break
        end = m.end() + m_end.end()
        block = out[start:end]
        converted = _convert_deluxetable_block(block)
        out = out[:start] + converted + out[end:]

    # Bibliography style swap for REVTeX
    out = re.sub(r"\\bibliographystyle\{[^}]*\}", r"\\bibliographystyle{apsrev4-2}", out)

    # Keep section flow natural in REVTeX to avoid float-placement deadlocks.

    # Force single-column float placement and fit graphics to a column.
    out = out.replace("\\begin{figure*}", "\\begin{figure}")
    out = out.replace("\\end{figure*}", "\\end{figure}")
    out = out.replace("\\begin{table*}", "\\begin{table}")
    out = out.replace("\\end{table*}", "\\end{table}")
    out = out.replace("\\textwidth", "\\columnwidth")
    out = out.replace("\\clearpage\n\n\\appendix", "\\newpage\n\n\\appendix", 1)
    out = re.sub(r"\\begin\{figure\}\[[^\]]*\]", r"\\begin{figure}[!htbp]", out)
    out = re.sub(r"\\begin\{table\}\[[^\]]*\]", r"\\begin{table}[!htbp]", out)

    # REVTeX requires \maketitle to render title/author/abstract block.
    # Ordering: abstract -> keywords (optional) -> maketitle.
    if "\\maketitle" not in out:
        if re.search(r"\\keywords\s*\{", out):
            out = re.sub(r"(\\keywords\s*\{[^}]*\}\s*)", r"\1\\maketitle\n\n", out, count=1)
        else:
            out = re.sub(r"(\\end\{abstract\}\s*)", r"\1\\maketitle\n\n", out, count=1)

    # Collapse excessive blank lines
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip() + "\n"


def convert_aas_to_mnras(aas_text: str, pubyear: int) -> str:
    title = _extract_command_arg(aas_text, "title") or "Untitled"
    authors = _extract_all_command_args(aas_text, "author")
    affiliations = _extract_all_command_args(aas_text, "affiliation")
    emails = _extract_all_command_args(aas_text, "email")

    out = aas_text
    out = re.sub(
        r"\\documentclass\[[^\]]*\]\{aastex701\}",
        r"\\documentclass[fleqn,usenatbib]{mnras}",
        out,
    )
    out = re.sub(r"\\documentclass\{aastex701\}", r"\\documentclass[fleqn,usenatbib]{mnras}", out)
    if "\\usepackage{placeins}" not in out:
        out = re.sub(
            r"(\\documentclass(?:\[[^\]]*\])?\{mnras\}\s*)",
            r"\1\\usepackage{placeins}\n",
            out,
            count=1,
        )

    # Remove AASTeX-only metadata commands.
    out = re.sub(r"^\\shorttitle\{.*?\}\s*$", "", out, flags=re.M)
    out = re.sub(r"^\\shortauthors\{.*?\}\s*$", "", out, flags=re.M)
    out = re.sub(r"^\\NewPageAfterKeywords\s*$", "", out, flags=re.M)

    # Convert deluxetable blocks to plain LaTeX tables.
    while True:
        m = re.search(r"\\begin\{deluxetable\*?\}\{[^}]*\}", out)
        if not m:
            break
        start = m.start()
        m_end = re.search(r"\\end\{deluxetable\*?\}", out[m.end() :])
        if not m_end:
            break
        end = m.end() + m_end.end()
        block = out[start:end]
        converted = _convert_deluxetable_block(block)
        out = out[:start] + converted + out[end:]

    # Convert AASTeX-specific sections.
    out = out.replace("\\begin{acknowledgments}", "\\section*{Acknowledgements}")
    out = out.replace("\\end{acknowledgments}", "")
    out = re.sub(
        r"\\keywords\s*\{([^}]*)\}",
        r"\\begin{keywords}\n\1\n\\end{keywords}",
        out,
        count=1,
        flags=re.S,
    )

    # Use MNRAS bibliography style.
    out = re.sub(r"\\bibliographystyle\{[^}]*\}", r"\\bibliographystyle{mnras}", out)
    out = re.sub(
        r"(?m)^\\bibliographystyle\{mnras\}",
        r"\\raggedbottom\n\\bibliographystyle{mnras}",
        out,
        count=1,
    )

    # Let the MNRAS class handle figure placement without carrying over AASTeX
    # placement specifiers, which can surface as visible text in some builds.
    out = re.sub(r"\\begin\{figure\*\}\[[^\]]*\]", r"\\begin{figure*}", out)
    out = re.sub(r"\\begin\{figure\}\[[^\]]*\]", r"\\begin{figure}", out)
    out = re.sub(
        r"\\clearpage\s*(\\section\{Discussion and Conclusions\})",
        r"\\FloatBarrier\n\n\1",
        out,
        count=1,
    )

    m_begin = re.search(r"\\begin\{document\}", out)
    if not m_begin:
        return out.strip() + "\n"

    preamble = out[: m_begin.start()].rstrip()
    body = out[m_begin.end() :]
    body = re.sub(r"\\end\{document\}\s*$", "", body, flags=re.S).strip()

    # Remove existing AASTeX front matter from body; we reinsert MNRAS front matter.
    for cmd in ("title", "author", "affiliation", "email"):
        body = _remove_all_command_with_braced_arg(body, cmd)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    if not authors:
        authors = ["Anonymous"]
    if not affiliations:
        affiliations = ["Affiliation not provided"]

    if len(affiliations) == 1:
        aff_ids = [1] * len(authors)
    else:
        aff_ids = [min(i + 1, len(affiliations)) for i in range(len(authors))]

    author_entries: list[str] = []
    for i, author in enumerate(authors):
        entry = f"{author}$^{{{aff_ids[i]}}}$"
        if i == 0 and emails:
            entry += f"\\thanks{{E-mail: {emails[0]}}}"
        author_entries.append(entry)

    if len(author_entries) == 1:
        author_line = author_entries[0]
    elif len(author_entries) == 2:
        author_line = f"{author_entries[0]} and {author_entries[1]}"
    else:
        author_line = ", ".join(author_entries[:-1]) + ", and " + author_entries[-1]

    aff_lines = [f"$^{{{i}}}${aff}" for i, aff in enumerate(affiliations, start=1)]

    short_title = title if len(title) <= 70 else title[:67].rstrip() + "..."
    short_authors = _surname_from_name(authors[0])
    if len(authors) > 1:
        short_authors += " et al."

    front_matter = (
        f"\\title[{short_title}]{{{title}}}\n\n"
        f"\\author[{short_authors}]{{\n"
        f"{author_line}\\\\\n"
        + "\n".join(aff_lines)
        + "\n}\n\n"
        "\\date{Accepted XXX. Received YYY; in original form ZZZ}\n"
        f"\\pubyear{{{pubyear}}}\n\n"
        "\\begin{document}\n"
        "\\label{firstpage}\n"
        "\\pagerange{\\pageref{firstpage}--\\pageref{lastpage}}\n"
        "\\maketitle\n\n"
    )

    if r"\label{lastpage}" not in body:
        body = body.rstrip() + "\n\n\\label{lastpage}"

    out = f"{preamble}\n\n{front_matter}{body}\n\n\\end{{document}}\n"
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out


def normalize_aas_float_placement(aas_text: str) -> str:
    """Normalize AASTeX float placement directives to reduce stuck-float warnings."""
    out = aas_text
    # Use permissive float placement to reduce deferred/stuck-float warnings.
    out = re.sub(r"\\begin\{figure\*\}\[[^\]]*\]", r"\\begin{figure*}[!htbp]", out)
    out = re.sub(r"\\begin\{figure\}\[[^\]]*\]", r"\\begin{figure}[!htbp]", out)
    out = re.sub(r"\\begin\{table\*\}\[[^\]]*\]", r"\\begin{table*}[!htbp]", out)
    out = re.sub(r"\\begin\{table\}\[[^\]]*\]", r"\\begin{table}[!htbp]", out)
    return out


def run_latexmk(tex_path: Path, outdir: Path) -> int:
    cmd = [
        "latexmk",
        "-pdf",
        "-interaction=nonstopmode",
        "-halt-on-error",
        "-outdir=" + str(outdir),
        str(tex_path),
    ]
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert AASTeX manuscript to REVTeX and compile both.")
    parser.add_argument("--aas", default="output/paper_aas.tex", help="Path to AASTeX source")
    parser.add_argument("--revtex", default="output/paper_prd.tex", help="Path to output REVTeX source")
    parser.add_argument("--mnras", default="output/paper_mnras.tex", help="Path to output MNRAS source")
    parser.add_argument("--outdir", default="output", help="latexmk output directory")
    parser.add_argument("--skip-build", action="store_true", help="Only convert, do not run latexmk")
    parser.add_argument("--build-mnras", action="store_true", help="Also build generated MNRAS manuscript")
    parser.add_argument("--pubyear", type=int, default=date.today().year, help="Publication year for MNRAS front matter")
    parser.add_argument(
        "--normalize-aas-floats",
        action="store_true",
        help="Rewrite AASTeX float placement directives before conversion/build",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    aas_path = (repo_root / args.aas).resolve()
    revtex_path = (repo_root / args.revtex).resolve()
    mnras_path = (repo_root / args.mnras).resolve()
    outdir = (repo_root / args.outdir).resolve()

    aas_text = aas_path.read_text(encoding="utf-8")
    if args.normalize_aas_floats:
        normalized = normalize_aas_float_placement(aas_text)
        if normalized != aas_text:
            aas_path.write_text(normalized, encoding="utf-8")
            aas_text = normalized
            print(f"Normalized AASTeX float placement: {aas_path}")
    revtex_text = convert_aas_to_revtex(aas_text)
    revtex_path.write_text(revtex_text, encoding="utf-8")
    print(f"Wrote REVTeX file: {revtex_path}")

    mnras_text = convert_aas_to_mnras(aas_text, pubyear=args.pubyear)
    mnras_path.write_text(mnras_text, encoding="utf-8")
    print(f"Wrote MNRAS file: {mnras_path}")

    if args.skip_build:
        return 0

    print("Building AASTeX manuscript...")
    code_aas = run_latexmk(aas_path, outdir)
    print(f"AASTeX build exit code: {code_aas}")

    print("Building REVTeX manuscript...")
    code_rev = run_latexmk(revtex_path, outdir)
    print(f"REVTeX build exit code: {code_rev}")

    code_mnras = 0
    if args.build_mnras:
        print("Building MNRAS manuscript...")
        code_mnras = run_latexmk(mnras_path, outdir)
        print(f"MNRAS build exit code: {code_mnras}")

    codes = [code_aas, code_rev]
    if args.build_mnras:
        codes.append(code_mnras)
    return 0 if all(code == 0 for code in codes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
