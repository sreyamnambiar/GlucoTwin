"""
Free-text food entry: "2 idli, sambar, coconut chutney" -> carbs, glycemic index, glycemic load.

The nutrient values are rounded, literature-style APPROXIMATIONS for typical Indian servings.
They are not a clinical food table, and GI varies by variety, cooking and what it is eaten with.
Items the parser cannot find are reported back, never guessed.
"""
import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.generate import MEALS  # noqa: E402  (the 27 composite meals used to generate the data)

# (name, aliases, serving, carbs_g per serving, GI, lower-GI swap)
_ROWS = [
    # tiffin / breakfast
    ("idli", ["idly"], "1 piece", 8, 70, "ragi idli"),
    ("ragi idli", [], "1 piece", 9, 55, None),
    ("plain dosa", ["dosa", "dosai", "set dosa"], "1 medium", 25, 75, "ragi dosa"),
    ("masala dosa", [], "1 large", 45, 70, "ragi dosa"),
    ("ragi dosa", ["ragi dosai"], "1 medium", 20, 55, None),
    ("rava dosa", [], "1 medium", 28, 70, "ragi dosa"),
    ("uttapam", ["uthappam"], "1 medium", 30, 70, "ragi dosa"),
    ("pesarattu", ["green gram dosa", "moong dosa"], "1 medium", 20, 45, None),
    ("upma", ["rava upma"], "1 cup", 40, 65, "oats upma"),
    ("oats upma", ["vegetable oats upma"], "1 cup", 30, 55, None),
    ("poha", ["aval", "aval upma", "flattened rice"], "1 cup", 40, 68, "oats upma"),
    ("pongal", ["ven pongal"], "1 cup", 38, 68, "millet pongal"),
    ("millet pongal", ["samai pongal"], "1 cup", 32, 52, None),
    ("idiyappam", ["string hoppers", "idiappam"], "1 piece", 12, 70, "ragi idiyappam"),
    ("ragi idiyappam", [], "1 piece", 11, 55, None),
    ("appam", ["palappam"], "1 piece", 15, 70, None),
    ("puttu", [], "1 cup", 40, 75, "ragi puttu"),
    ("ragi puttu", [], "1 cup", 35, 55, None),
    ("paratha", ["plain paratha"], "1 piece", 30, 65, "multigrain roti"),
    ("aloo paratha", [], "1 piece", 38, 65, "multigrain roti"),
    ("white bread", ["bread", "bread slice", "toast"], "1 slice", 13, 75, "multigrain bread"),
    ("multigrain bread", ["brown bread"], "1 slice", 12, 53, None),
    ("oats porridge", ["oats", "oatmeal"], "1 cup", 27, 55, None),
    ("cornflakes", ["corn flakes", "cereal"], "1 cup", 24, 80, "oats porridge"),
    ("boiled egg", ["egg", "eggs"], "1 egg", 1, 30, None),
    ("omelette", ["omelet", "egg omelette"], "1 egg", 1, 30, None),
    # rice, rotis, rice dishes
    ("white rice", ["rice", "steamed rice", "chawal", "sadam", "plain rice"], "1 cup cooked", 45, 73, "millet rice"),
    ("brown rice", [], "1 cup cooked", 42, 55, None),
    ("millet rice", ["foxtail millet rice", "samai rice", "millet"], "1 cup cooked", 38, 52, None),
    ("ragi mudde", ["ragi ball", "ragi kali", "kali"], "1 ball", 28, 55, None),
    ("roti", ["chapati", "phulka", "chapathi", "fulka"], "1 piece", 18, 62, "multigrain roti"),
    ("multigrain roti", ["atta roti"], "1 piece", 16, 50, None),
    ("jowar roti", ["jolada rotti", "sorghum roti"], "1 piece", 17, 55, None),
    ("bajra roti", ["pearl millet roti"], "1 piece", 17, 55, None),
    ("naan", [], "1 piece", 45, 71, "multigrain roti"),
    ("puri", ["poori"], "1 piece", 15, 70, "multigrain roti"),
    ("curd rice", ["thayir sadam", "mosaru anna"], "1 cup", 35, 65, "millet rice"),
    ("lemon rice", [], "1 cup", 45, 70, "millet rice"),
    ("tamarind rice", ["puliyodarai", "puliyogare"], "1 cup", 48, 70, "millet rice"),
    ("pulao", ["veg pulao", "vegetable pulao"], "1 cup", 45, 68, "millet rice"),
    ("biryani", ["chicken biryani", "veg biryani", "mutton biryani"], "1 cup", 50, 70, "brown rice"),
    ("khichdi", ["khichadi"], "1 cup", 35, 55, None),
    ("bisibele bath", ["bisi bele bath"], "1 cup", 40, 60, None),
    # dal, curries, sides
    ("sambar", [], "1 cup", 12, 35, None),
    ("dal", ["toor dal", "dal tadka", "moong dal", "lentil", "dal fry"], "1 cup", 20, 30, None),
    ("rasam", [], "1 cup", 6, 35, None),
    ("rajma", ["rajma curry"], "1 cup", 30, 35, None),
    ("chole", ["chana masala", "chickpea curry"], "1 cup", 33, 40, None),
    ("kootu", [], "1 cup", 15, 35, None),
    ("aloo sabzi", ["potato curry", "aloo curry"], "1 cup", 25, 70, "mixed vegetable curry"),
    ("mixed vegetable curry", ["sabzi", "vegetable curry", "veg curry", "poriyal"], "1 cup", 10, 40, None),
    ("palak paneer", [], "1 cup", 10, 30, None),
    ("paneer curry", ["paneer butter masala", "paneer"], "1 cup", 12, 30, None),
    ("chicken curry", ["chicken"], "1 cup", 6, 30, None),
    ("fish curry", ["fish", "fish fry"], "1 cup", 5, 30, None),
    ("egg curry", [], "1 cup", 6, 30, None),
    ("sundal", ["chickpea sundal"], "1 cup", 25, 35, None),
    ("coconut chutney", ["chutney", "tomato chutney", "mint chutney"], "2 tbsp", 2, 30, None),
    # dairy and drinks
    ("curd", ["yogurt", "dahi", "yoghurt", "thayir"], "1 cup", 8, 35, None),
    ("buttermilk", ["chaas", "moru", "majjige"], "1 glass", 4, 30, None),
    ("milk", [], "1 glass", 12, 40, None),
    ("sweet lassi", ["lassi"], "1 glass", 30, 50, "buttermilk"),
    ("tea with sugar", ["tea", "chai", "sweet tea"], "1 cup", 9, 55, "tea without sugar"),
    ("tea without sugar", ["black tea", "green tea", "unsweetened tea"], "1 cup", 1, 10, None),
    ("coffee with sugar", ["coffee", "filter coffee"], "1 cup", 9, 55, None),
    ("coconut water", ["tender coconut", "elaneer"], "1 glass", 9, 55, None),
    ("fruit juice", ["juice", "sugarcane juice"], "1 glass", 28, 60, None),
    ("soft drink", ["cola", "soda"], "1 glass", 35, 65, None),
    # snacks
    ("samosa", [], "1 piece", 26, 68, None),
    ("medu vada", ["vada", "ulundu vada"], "1 piece", 12, 50, None),
    ("bonda", ["potato bonda"], "1 piece", 15, 60, None),
    ("bajji", ["pakora", "pakoda"], "3 pieces", 18, 55, None),
    ("bhel puri", ["bhel"], "1 plate", 40, 60, None),
    ("pani puri", ["golgappa", "puchka"], "6 pieces", 30, 65, None),
    ("vada pav", [], "1 piece", 40, 70, None),
    ("veg sandwich", ["sandwich"], "1 piece", 30, 65, None),
    ("tea biscuits", ["biscuit", "biscuits", "marie biscuit"], "3 pieces", 20, 65, None),
    ("almonds", ["nuts", "cashews", "walnuts"], "1 handful", 6, 15, None),
    ("peanuts", ["groundnuts", "roasted peanuts"], "1 handful", 6, 14, None),
    ("maggi noodles", ["noodles", "instant noodles", "maggi"], "1 pack", 45, 70, None),
    ("pizza", ["pizza slice"], "1 slice", 30, 60, None),
    ("burger", [], "1 piece", 40, 65, None),
    ("french fries", ["fries", "chips"], "1 serving", 40, 75, None),
    # fruit
    ("banana", [], "1 medium", 27, 55, None),
    ("apple", [], "1 medium", 25, 38, None),
    ("orange", ["sweet lime", "mosambi"], "1 medium", 15, 43, None),
    ("mango", [], "1 cup", 25, 56, "guava"),
    ("papaya", [], "1 cup", 14, 59, None),
    ("guava", [], "1 medium", 14, 20, None),
    ("watermelon", [], "1 cup", 11, 72, "guava"),
    ("grapes", [], "1 cup", 27, 59, None),
    ("pomegranate", [], "1 cup", 32, 35, None),
    # sweets
    ("gulab jamun", [], "1 piece", 25, 70, "almonds"),
    ("jalebi", [], "1 piece", 18, 75, "almonds"),
    ("rasgulla", [], "1 piece", 20, 65, "almonds"),
    ("laddu", ["ladoo"], "1 piece", 20, 70, "almonds"),
    ("payasam", ["kheer"], "1 cup", 35, 65, None),
    ("halwa", ["sooji halwa", "kesari"], "1/2 cup", 30, 70, None),
    ("ice cream", [], "1 scoop", 15, 60, None),
    ("chocolate", [], "1 piece", 8, 40, None),
]

FOODS = [dict(name=n, aliases=a, serving=s, carbs=c, gi=g, swap=w, source="food list") for n, a, s, c, g, w in _ROWS]
_names = {f["name"] for f in FOODS}
for slot_items in MEALS.values():                      # composite plates used by the generator, so typed library names also work
    for n, carbs, gi, _ in slot_items:
        nm = n.replace("_", " ")
        if nm not in _names:
            FOODS.append(dict(name=nm, aliases=[], serving="1 plate", carbs=carbs, gi=gi, swap=None, source="meal library"))
            _names.add(nm)
BY_NAME = {f["name"]: f for f in FOODS}

_STOP = {"of", "the", "some", "and", "with", "little"}


def _norm(s: str) -> str:
    toks = re.sub(r"[^a-z0-9 ]", " ", s.lower()).split()
    toks = [t for t in toks if t not in _STOP]
    toks = [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]
    return " ".join(toks)


_KEYS = []                                             # (normalised key, food)
for _f in FOODS:
    for _k in [_f["name"], *_f["aliases"]]:
        _KEYS.append((_norm(_k), _f))


def match(text: str):
    """Best food for a typed name -> (food or None, score 0..1)."""
    q = _norm(text)
    if not q:
        return None, 0.0
    qt = set(q.split()); best, score = None, 0.0
    for key, f in _KEYS:
        kt = set(key.split())
        if key == q:
            s = 1.0
        elif qt <= kt:
            s = .9 - .04 * (len(kt) - len(qt))
        elif kt <= qt:
            s = .85 - .04 * (len(qt) - len(kt))
        else:
            r = difflib.SequenceMatcher(None, q, key).ratio()
            s = r * .95 if r >= .8 else 0
        if s > score:
            best, score = f, s
    return (best, score) if score >= .7 else (None, score)


def search(q: str, limit: int = 6):
    """Autocomplete: prefix matches first, then word-prefix, then fuzzy."""
    nq = _norm(q)
    if not nq:
        return []
    scored = {}
    for key, f in _KEYS:
        if key.startswith(nq):
            s = 3 - len(key) / 100
        elif all(any(k.startswith(t) for k in key.split()) for t in nq.split()):
            s = 2 - len(key) / 100
        else:
            r = difflib.SequenceMatcher(None, nq, key).ratio()
            s = r if r >= .74 else 0
        if s and s > scored.get(f["name"], (0,))[0]:
            scored[f["name"]] = (s, f)
    out = sorted(scored.values(), key=lambda x: -x[0])[:limit]
    return [dict(name=f["name"], serving=f["serving"], carbs=f["carbs"], gi=f["gi"]) for _, f in out]


_NUM = {"half": .5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "a": 1, "an": 1}
_QTY = re.compile(r"^(\d+(?:\.\d+)?|\d+/\d+|half|one|two|three|four|five|six|a|an)(?:\s*[x×])?\s+(?:(?:pieces?|pcs?|nos?|cups?|plates?|bowls?|glass(?:es)?|slices?|scoops?)\s+)?(?:of\s+)?(.+)$", re.I)
_QTY_ATTACHED = re.compile(r"^(\d+(?:\.\d+)?)\s*[x×]?\s*([a-z].*)$", re.I)
_GRAMS = re.compile(r"(\d+(?:\.\d+)?)\s*g(?:rams?)?\b(?:\s*carbs?)?", re.I)


def _qty(part: str):
    for rx in (_QTY, _QTY_ATTACHED):
        m = rx.match(part.strip())
        if m:
            raw = m.group(1).lower()
            q = _NUM.get(raw) or (float(raw.split("/")[0]) / float(raw.split("/")[1]) if "/" in raw else float(raw))
            return min(max(q, .25), 12.0), m.group(2).strip()
    return 1.0, part.strip()


def parse_meal(text: str) -> dict:
    """Split a typed meal into items and total its carbs and glycemic load."""
    items, unmatched = [], []
    for part in re.split(r",|\+|;|\n|\band\b|\bwith\b|&", text or "", flags=re.I):
        part = part.strip()
        if not part:
            continue
        custom = None
        g = _GRAMS.search(part)
        if g:
            custom = float(g.group(1)); part = (part[:g.start()] + part[g.end():]).strip(" :-")
        qty, name = _qty(part)
        if custom is not None and not name:
            name = "custom item"
        f, score = match(name)
        if f is None and custom is None:
            unmatched.append(name or part); items.append(dict(query=part, matched=False)); continue
        gi = f["gi"] if f else 60
        carbs = custom if custom is not None else qty * f["carbs"]
        items.append(dict(query=part, matched=True, name=f["name"] if f else name, qty=qty, serving=(f["serving"] if f else "custom"),
                          carbs_g=round(carbs, 1), gi=gi, gl=round(carbs * gi / 100, 1), assumed_gi=f is None))
    ok = [i for i in items if i["matched"]]
    carbs = sum(i["carbs_g"] for i in ok); gl = sum(i["gl"] for i in ok)
    return dict(items=items, unmatched=unmatched, carbs_g=round(carbs, 1), glycemic_load=round(gl, 1),
                gi=round(gl / carbs * 100) if carbs > 0 else 0, ok=bool(ok))


def suggest_swap(text: str) -> dict:
    """Replace items that have a lower-GI alternative, keeping quantities."""
    changes, parts = [], []
    for part in re.split(r"(,|\+|;|\n|\band\b|\bwith\b|&)", text or "", flags=re.I):
        if re.fullmatch(r",|\+|;|\n|and|with|&", part.strip(), flags=re.I) or not part.strip():
            parts.append(part); continue
        qty_txt = ""
        m = _QTY.match(part.strip()) or _QTY_ATTACHED.match(part.strip())
        name = m.group(2) if m else part.strip()
        if m:
            qty_txt = part.strip()[:part.strip().rfind(name)].strip() + " "
        f, _ = match(name)
        if f and f["swap"] and f["swap"] in BY_NAME and BY_NAME[f["swap"]]["gi"] < f["gi"]:
            parts.append(f" {qty_txt}{f['swap']}"); changes.append(dict(was=f["name"], now=f["swap"], gi_was=f["gi"], gi_now=BY_NAME[f["swap"]]["gi"]))
        else:
            parts.append(part)
    return dict(text="".join(parts).strip(), changes=changes)
