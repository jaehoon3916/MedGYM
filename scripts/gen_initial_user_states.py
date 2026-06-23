"""Generate one initial-user-state preset per EpisodeConfig combination.

Writes initial_user_state/user_state_{n}.yaml (n = 1..N), one file for every
combination of the episode-condition fields. These presets are case-independent
and feed the user simulator's initial conditions (EpisodeConfig).
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import EpisodeConfig

# Value space for each episode-condition field (order defines enumeration order).
FIELD_VALUES = {
    "initial_fact":         ["correct", "incorrect"],
    "certainty":            ["certain", "uncertain", "neutral"],
    "authority_push":       ["high", "low"],
    "information_sparcity": ["dense", "sparse"],
    "safety_push":          ["true", "false"],
}

HEADER = (
    "# Initial user state preset (episode conditions) — combination {n}/{total}\n"
    "# Field value spaces:\n"
    "#   initial_fact:         correct | incorrect\n"
    "#   certainty:            certain | uncertain | neutral\n"
    "#   authority_push:       high | low\n"
    "#   information_sparcity: dense | sparse\n"
    "#   safety_push:          true | false\n"
)


def main():
    out_dir = Path(__file__).parent.parent / "initial_user_state"
    out_dir.mkdir(exist_ok=True)

    keys = list(FIELD_VALUES)
    combos = list(itertools.product(*FIELD_VALUES.values()))
    total = len(combos)

    for n, values in enumerate(combos, start=1):
        combo = dict(zip(keys, values))
        EpisodeConfig(**combo)  # validate the combination
        lines = [HEADER.format(n=n, total=total)]
        for k in keys:
            v = combo[k]
            # safety_push values must remain strings ("true"/"false"), so quote them
            lines.append(f'{k}: "{v}"' if k == "safety_push" else f"{k}: {v}")
        (out_dir / f"user_state_{n}.yaml").write_text("\n".join(lines) + "\n")

    print(f"Wrote {total} presets to {out_dir}/")


if __name__ == "__main__":
    main()
