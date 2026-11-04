"""
make-plots.py  --  make the AMOC paper figures and tables by number.

    python make-plots.py              # everything
    python make-plots.py 1            # Fig 1 (food web + growth vs temperature)
    python make-plots.py 4 5 s2       # several at once
    python make-plots.py s1           # Fig S1
    python make-plots.py ts1          # Table S1
    python make-plots.py MF_PRIM      # a script name also works
    python make-plots.py --list       # show the numbering and what each runs
    python make-plots.py 7 --dry-run  # print the command, run nothing

Names:  1..9 = main figures,  s1..s6 = supplementary figures,
        ts1, ts2 = supplementary tables (so they don't clash with Fig S1, S2).
Several numbers can come from one script (8 and 9 are both MF_FLUX; s3 and s4
are both SF_ASS_SPECIES; 1 runs two scripts). Each script runs at most once
per call.

Each script is run as its own process with this folder as the working
directory, using the same Python that runs this file (start it from the
swamp2 environment). One failure does not stop the rest; a summary is printed
at the end and the exit code is 1 if anything failed.
Outputs go to AMOC-TOM/plts (set in each script); logs to AMOC-TOM/logs/<script>.log.

EDIT THE TABLES BELOW to change the numbering or the flags a script gets.
"""

import os
import sys
import time
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(HERE), "logs")

# script name -> (file, extra arguments)
# Arguments are the ones in the figure list sheet / script docstrings.
SCRIPTS = {
    "food_web_diagram": ("food_web_diagram.py", []),
    "SF_GROWTH_TEMP":   ("SF_GROWTH_TEMP.py",   []),
    "MF_AMOC":          ("MF_AMOC.py",          []),
    "MF_PHYS":          ("MF_PHYS.py",          []),   # add "--with-ice" for the ice column
    "MF_PRIM":          ("MF_PRIM.py",          []),
    "MF_LIMNUT":        ("MF_LIMNUT.py",        []),
    "MF_ZOO":           ("MF_ZOO.py",           []),
    "MF_ASS":           ("MF_ASS.py",           ["--with-dom", "--nobac"]),
    "MF_FLUX":          ("MF_FLUX.py",          []),
    "SF_LIMCOMP":       ("SF_LIMCOMP.py",       []),
    "SF_ASS_SPECIES":   ("SF_ASS_SPECIES.py",   []),
    "MF_EXPD":          ("MF_EXPD.py",          []),
    "SF_CFLX_SIGN":     ("SF_CFLX_SIGN.py",     []),
    "ST_ECO":           ("ST_ECO.py",           []),
    "ST_ASS":           ("ST_ASS.py",           []),
}

# figure/table label -> (description, [script names]); order here = default run order.
# An empty list means no script is assigned yet.
FIGURES = [
    ("1",   "Fig 1: food web + growth vs temperature", ["food_web_diagram", "SF_GROWTH_TEMP"]),
    ("2",   "Fig 2: AMOC streamfunction and regions",  ["MF_AMOC"]),
    ("3",   "Fig 3: SST, SSS, MLD",                    ["MF_PHYS"]),
    ("4",   "Fig 4: biomass, PP, limitation",          ["MF_PRIM"]),
    ("5",   "Fig 5: limiting nutrients",               ["MF_LIMNUT"]),
    ("6",   "Fig 6: zooplankton",                      ["MF_ZOO"]),
    ("7",   "Fig 7: assemblage",                       ["MF_ASS"]),
    ("8",   "Fig 8: export, ef, Cflx",                 ["MF_FLUX"]),
    ("9",   "Fig 9: made by MF_FLUX together with Fig 8", ["MF_FLUX"]),
    ("s1",  "Fig S1: nutrient concentrations (script not assigned yet)", []),
    ("s2",  "Fig S2: light vs nutrient limitation",    ["SF_LIMCOMP"]),
    ("s3",  "Fig S3: per-species maps",                ["SF_ASS_SPECIES"]),
    ("s4",  "Fig S4: made by SF_ASS_SPECIES with S3",  ["SF_ASS_SPECIES"]),
    ("s5",  "Fig S5: export decomposition",            ["MF_EXPD"]),
    ("s6",  "Fig S6: air-sea flux sign",               ["SF_CFLX_SIGN"]),
    ("ts1", "Table S1: ecosystem metrics by region",   ["ST_ECO"]),
    ("ts2", "Table S2: assemblage by region",          ["ST_ASS"]),
]
BY_LABEL = {lab: (desc, scr) for lab, desc, scr in FIGURES}


def run_one(name, dry_run):
    script, args = SCRIPTS[name]
    path = os.path.join(HERE, script)
    cmd = [sys.executable, path] + list(args)
    print("\n" + "=" * 78)
    print("[%s]  %s" % (name, " ".join(cmd)))
    print("=" * 78)
    if dry_run:
        return "dry-run", 0.0
    if not os.path.isfile(path):
        print("MISSING: %s" % path)
        return "MISSING", 0.0

    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")      # batch run: no display (one script calls plt.show())
    t0 = time.time()
    with open(os.path.join(LOG_DIR, name + ".log"), "w") as log:
        log.write("$ %s\n\n" % " ".join(cmd))
        proc = subprocess.Popen(cmd, cwd=HERE, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, universal_newlines=True)
        for line in proc.stdout:             # show live and keep a copy
            sys.stdout.write(line)
            log.write(line)
        proc.wait()
    status = "ok" if proc.returncode == 0 else "FAILED (exit %d)" % proc.returncode
    return status, time.time() - t0


def resolve(tokens):
    """Turn command-line tokens into (ordered unique script names, notes)."""
    names, notes = [], []
    for tok in tokens:
        key = tok.lower().replace("fig", "").replace("table", "t")
        low = tok.lower()
        if key in BY_LABEL:
            desc, scr = BY_LABEL[key]
            if not scr:
                notes.append("%s: %s" % (tok, desc))
            for s in scr:
                if s not in names:
                    names.append(s)
        else:
            match = [s for s in SCRIPTS if s.lower() == low]
            if not match:
                valid = ", ".join(lab for lab, _, _ in FIGURES)
                sys.exit("unknown figure %r.\nValid: %s\n(or a script name; see --list)" % (tok, valid))
            if match[0] not in names:
                names.append(match[0])
    return names, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("figures", nargs="*", metavar="FIG",
                    help="1-9, s1-s6, ts1, ts2, or a script name; default: all")
    ap.add_argument("--list", action="store_true", help="show numbering and exit")
    ap.add_argument("--dry-run", action="store_true", help="print commands, run nothing")
    ap.add_argument("--stop-on-error", action="store_true", help="stop at the first failure")
    a = ap.parse_args()

    if a.list:
        for lab, desc, scr in FIGURES:
            flags = " ".join(" ".join(SCRIPTS[s][1]) for s in scr).strip()
            print("%-4s %-52s %s%s" % (lab, desc, ", ".join(scr) or "-",
                                       ("  [" + flags + "]") if flags else ""))
        return 0

    if a.figures:
        todo, notes = resolve(a.figures)
    else:  # everything with a script, in table order, each script once
        todo, notes = [], []
        for lab, desc, scr in FIGURES:
            for s in scr:
                if s not in todo:
                    todo.append(s)
    for n in notes:
        print("NOTE  " + n)

    results = []
    for name in todo:
        status, dt = run_one(name, a.dry_run)
        results.append((name, status, dt))
        if a.stop_on_error and status not in ("ok", "dry-run"):
            print("\nStopping after %s (--stop-on-error)." % name)
            break

    print("\n" + "=" * 78 + "\nSUMMARY\n" + "=" * 78)
    for name, status, dt in results:
        print("%-18s %-18s %7.1f s" % (name, status, dt))
    bad = [r for r in results if r[1] not in ("ok", "dry-run")]
    print("\n%d run, %d failed.%s" % (len(results), len(bad),
          "" if a.dry_run else "  Logs: %s" % LOG_DIR))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
