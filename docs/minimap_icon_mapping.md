# Minimap Icon Mapping

This table covers the actor types that currently appear in
`openra_bench/scenarios/packs/*.yaml` actor lists. The scan intentionally
counts only authored or spawned actors under `actors:` lists, so event
names and predicate `type:` fields are not included.

Scan command used:

```bash
python3 - <<'PY'
import yaml, pathlib, collections
root = pathlib.Path('openra_bench/scenarios/packs')
counts = collections.Counter()
files = collections.defaultdict(set)

def collect(seq, path):
    if not isinstance(seq, list):
        return
    for item in seq:
        if isinstance(item, dict) and 'type' in item and (
            'owner' in item or 'position' in item or
            'count' in item or 'spawn_point' in item
        ):
            t = str(item['type']).lower()
            counts[t] += 1
            files[t].add(path.name)

def walk(node, path):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == 'actors':
                collect(v, path)
            walk(v, path)
    elif isinstance(node, list):
        for v in node:
            walk(v, path)

for p in sorted(root.glob('*.yaml')):
    if p.name == 'TEMPLATE.yaml':
        continue
    walk(yaml.safe_load(p.read_text()), p)

for t, n in sorted(counts.items()):
    print(t, n, len(files[t]))
PY
```

| Actor type | Count | Files | Class | Shape | Accent |
| --- | ---: | ---: | --- | --- | --- |
| `1tnk` | 196 | 25 | unit | square | none |
| `2tnk` | 1080 | 88 | unit | diamond | none |
| `3tnk` | 117 | 18 | unit | hexagon | none |
| `4tnk` | 77 | 8 | unit | triangle | none |
| `agun` | 6 | 3 | building | def_square | x |
| `apc` | 1 | 1 | unit | hexagon | dot |
| `arty` | 11 | 1 | unit | star | crosshair |
| `atek` | 16 | 4 | building | hex_tall | none |
| `barr` | 36 | 13 | building | triangle | dot |
| `dd` | 5 | 1 | unit | diamond | wave |
| `dog` | 4 | 3 | unit | diamond | dot |
| `dome` | 35 | 10 | building | circle | ring |
| `e1` | 1542 | 157 | unit | circle | none |
| `e2` | 32 | 4 | unit | diamond | cross |
| `e3` | 609 | 81 | unit | triangle | dot |
| `e6` | 8 | 2 | unit | pentagon | cross |
| `fact` | 1283 | 200 | building | pentagon | halo |
| `fix` | 123 | 28 | building | square | cross |
| `gun` | 47 | 11 | building | def_square | ring |
| `harv` | 310 | 55 | unit | tridown | none |
| `hbox` | 1 | 1 | building | def_square | dot |
| `heli` | 11 | 3 | unit | chevron | none |
| `hpad` | 16 | 4 | building | square | x |
| `jeep` | 371 | 57 | unit | pentagon | none |
| `mcv` | 76 | 14 | unit | trapezoid | halo |
| `medi` | 5 | 3 | unit | plus | dot |
| `mine` | 376 | 52 | building | circle | ring |
| `mslo` | 4 | 1 | building | star | none |
| `pbox` | 169 | 15 | building | def_square | crosshair |
| `powr` | 533 | 108 | building | diamond | dot |
| `proc` | 376 | 92 | building | trapezoid | none |
| `sam` | 1 | 1 | building | def_square | antenna |
| `silo` | 111 | 5 | building | bar | none |
| `spy` | 4 | 1 | unit | kite | ring |
| `syrd` | 8 | 2 | building | square | wave |
| `tanya` | 6 | 2 | unit | star | halo |
| `tent` | 262 | 67 | building | triangle | none |
| `thf` | 4 | 1 | unit | kite | dot |
| `tsla` | 9 | 4 | building | star | ring |
| `weap` | 165 | 38 | building | hexagon | none |

