# Implementation Notes — Static Mission Player Site

## Architecture

The mission player is a **static single-page application** in `site/`:

```
site/
  index.html        — Full SPA (HTML + CSS + JS, no framework)
  generate.py       — Maintainer preprocessing script
  public/
    scenarios.json   — Generated scenario metadata (196 scenarios)
    maps/            — Map thumbnails (requires engine wheel)
```

## For Human Players / Annotators

1. Open the hosted website (GitHub Pages or local file server).
2. Browse the mission selection screen — cards with title, capability badge, status.
3. Search/filter by name or capability.
4. Click a card to open the mission page.
5. Read the objective in English or Chinese (toggle tabs).
6. Switch difficulty levels (easy/medium/hard) to see different objectives.
7. Use the map toolbar: Select, + Point, + Region.
8. Click the map to add point annotations; drag to add region annotations.
9. Tag annotations with predefined or custom tags; add notes.
10. Edit or delete annotations from the list.
11. Mark the scenario as "annotated" or "complete".
12. Click "Next Mission" or "Next Unannotated" to continue.
13. Save progress (persisted in localStorage).
14. Export annotations as JSON; import previously exported JSON.

## For Maintainers

### Generate / refresh static data

```bash
python site/generate.py            # generate scenarios.json + map thumbnails
python site/generate.py --dry-run  # print counts without writing files
```

### Regenerate after new scenarios are added

The pipeline reads live pack data from `openra_bench/scenarios/packs/`.
After adding/modifying packs, re-run `python site/generate.py`.

### Map thumbnails

Map rendering requires the Rust engine wheel (`openra_train`).
Install it with:
```bash
cd OpenRA-Rust && PATH=$HOME/.cargo/bin:$PATH maturin develop --release
```
Without the wheel, the site works with placeholder map areas.

### Run tests

```bash
python -m pytest tests/test_site.py tests/test_app.py -v
```

### Build for static hosting

The site is already static. Copy `site/index.html` and `site/public/`
to any static host (GitHub Pages, Netlify, etc.).

```bash
# Example: deploy to gh-pages branch
cp -r site/index.html site/public/ docs/  # to your deploy directory
```

### Bilingual descriptions

Descriptions are generated deterministically by `site/generate.py`
(no AI API required). The pipeline translates common objective patterns
to Chinese using pattern-matching substitution. To improve translations,
edit the `_translate_objective_zh()` function in `site/generate.py`.

To use AI-assisted translation instead, run a preprocessing step with
your API key BEFORE deployment — the hosted site never calls AI APIs.

## Design Decisions

- **Single HTML file**: No build step, no npm, no framework. Deployable anywhere.
- **localStorage persistence**: No server needed for progress/annotations.
- **Normalized coordinates**: Annotations use [0,1] coordinates relative to the map container, making them resolution-independent.
- **Deterministic ZH generation**: Pattern-matching translation avoids AI API dependency. Quality is "functional" not "native" — suitable for annotation workflow.
- **No raw JSON in primary UI**: Scenario data is rendered as mission cards and objective panels, never as raw JSON viewers.

## Known Limitations

- **Map thumbnails**: Require the Rust engine wheel to render. Without it, the map area shows a placeholder. Annotations still work on the placeholder area.
- **Chinese translations**: Deterministic pattern-matching produces functional but imperfect translations. For production quality, run an AI translation preprocessing step.
- **Path annotations**: Not implemented (marked as nice-to-have in the spec). Point and region annotations are fully functional.
- **Step-by-step guides**: The scenario packs don't have discrete "steps" — they have level descriptions and win/fail conditions which are displayed as the objective panel.
- **Leaderboard integration**: The static site reads from `scenarios.json`; leaderboard data is available in the main Gradio app.
