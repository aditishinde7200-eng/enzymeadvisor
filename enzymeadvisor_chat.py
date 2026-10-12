"""Rule-based chat layer for EnzymeAdvisor.

No language model is used: every reply is assembled from the engine's numbers, so it cannot invent content.
"""
import re
import enzymeadvisor_engine as ea

HELP = (
    "Commands:\n"
    "  sequence <amino-acid sequence or FASTA>   set the protein (clears earlier measurements)\n"
    "  budget <n>                                how many variants you can test in one round\n"
    "  recommend                                 get the panel for the next round\n"
    "  measured A148D 1.3, G156S 0.4             enter lab results (higher = better; negate if lower is better)\n"
    "  why not A148R                             why a variant was not picked\n"
    "  forget A148D                              remove one measurement\n"
    "  status | check | reset | help"
)
MUT = r'[A-Za-z]\d+[A-Za-z]'


def _sequence_from_text(text):
    """If the whole message is a pasted protein sequence (optionally with a FASTA header), return it cleaned."""
    lines = [l.strip() for l in str(text).splitlines() if l.strip()]
    body = [l for l in lines if not (l.startswith('>') or '|' in l or ' OS=' in l)]
    seq = re.sub(r'[\s\d]+', '', ''.join(body)).upper()
    if len(seq) >= 20 and set(seq) <= set(ea.AAS):
        return seq
    return None



class EnzymeAdvisorChat:
    def __init__(self):
        self.seq, self.budget, self.measured, self.res = None, 5, {}, None

    def reply(self, text):
        t = str(text).strip()
        low = t.lower()
        if not t or low in ('help', '?', 'commands'):
            return HELP
        guess = _sequence_from_text(t)
        if guess and not re.match(r'^(sequence|seq)\b', t, re.I):
            t = 'sequence ' + guess
            low = t.lower()

        m = re.match(r'^(sequence|seq)\b\s*[:=]?\s*(.*)$', t, re.I | re.S)
        if m:
            try:
                self.seq = ea.clean_sequence(m.group(2))
            except ValueError as e:
                return f"Could not use that sequence: {e}"
            self.measured, self.res = {}, None
            return (f"Sequence set: {len(self.seq)} residues, {19 * len(self.seq)} possible single mutations. "
                    f"Budget is {self.budget}. Type 'recommend' for a first panel.")

        m = re.match(r'^budget\b\s*[:=]?\s*(\d+)\s*$', low)
        if m:
            n = int(m.group(1))
            if not 1 <= n <= 50:
                return "The budget must be between 1 and 50."
            self.budget, self.res = n, None
            return f"Budget set to {n}."

        m = re.match(r'^(i\s+)?(measured|measure|results?)\b(.*)$', t, re.I | re.S)
        if m:
            if not self.seq:
                return "Please set the protein first: sequence <amino-acid sequence>."
            pairs = re.findall(rf'({MUT})\s*[:=]?\s*(-?\d+(?:\.\d+)?)', m.group(3))
            if not pairs:
                return "I could not find any 'mutation value' pairs. Example: measured A148D 1.3, G156S 0.4"
            ok, problems = ea.parse_measured(self.seq, pairs)
            self.measured.update(ok)
            self.res = None
            out = (f"Recorded {len(ok)} measurement(s); {len(self.measured)} in total "
                   f"(the learning method needs at least {ea.MIN_MEASURED}).")
            if problems:
                out += " Ignored: " + "; ".join(problems[:5]) + ("..." if len(problems) > 5 else "")
            return out.rstrip(".") + ". Remember: higher values must mean better."

        m = re.match(rf'^forget\s+({MUT})\s*$', t, re.I)
        if m:
            k = m.group(1).upper()
            if self.measured.pop(k, None) is None:
                return f"{k} is not among your measurements."
            self.res = None
            return f"Removed {k}. {len(self.measured)} measurements remain."

        if re.match(r'^(recommend|suggest|panel|go|run|next)\b', low):
            if not self.seq:
                return "Please set the protein first: sequence <amino-acid sequence>."
            try:
                self.res = ea.recommend(self.seq, self.budget, list(self.measured.items()))
            except ValueError as e:
                return f"Could not make a recommendation: {e}"
            return (ea.format_recommendation(self.res) + self._context(self.res) +
                    "\n\nAfter testing these, tell me the results, e.g. 'measured A148D 1.3, G156S 0.4', "
                    "then type 'recommend' again.")

        m = re.match(rf'^why(\s+not)?\s+({MUT})\s*\??$', t, re.I)
        if m:
            if self.res is None:
                return "Type 'recommend' first, then ask why a variant was or was not picked."
            return ea.why_not(self.res, m.group(2))

        if low in ('status', 'state'):
            if not self.seq:
                return "No protein set yet."
            n = len(self.measured)
            mode = ("the learning method" if n >= ea.MIN_MEASURED else
                    f"ESM-2 ranking ({ea.MIN_MEASURED - n} more needed for the learning method)")
            return f"Protein: {len(self.seq)} residues | budget {self.budget} | measured {n} | next panel uses {mode}"

        if low == 'check':
            if self.res is None:
                return "Type 'recommend' first."
            return "Self-check (recomputes every number in the explanation independently): " + str(ea.verify(self.res))

        if low == 'reset':
            self.__init__()
            return "Cleared everything. Start with: sequence <amino-acid sequence>"

        return "I did not understand that.\n" + HELP

    def _context(self, res):
        if res['mode'] != 'gp_diverse':
            return ""
        esm = res['X'][:, -1]
        lookup = dict(zip(res['names'], esm))
        lines = ["", "", "ESM-2 CONTEXT (where each pick's ESM-2 score sits among all candidates)"]
        for p in res['panel']:
            v = lookup[p['mutation']]
            lines.append(f"- {p['mutation']}: ESM-2 score {v:.1f}, higher than {100.0 * (esm < v).mean():.0f}% "
                         f"of all candidate substitutions")
        lines.append("How to read this: in this model, substitutions at the same position share one position "
                     "description and differ only through their ESM-2 score. So the learning method mainly finds "
                     "promising positions; within a position it follows ESM-2's ranking. A low percentile means "
                     "ESM-2 finds that substitution unlikely, so consider the substitution as well as the position.")
        return "\n".join(lines)
