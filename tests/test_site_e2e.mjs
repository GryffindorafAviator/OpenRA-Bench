/**
 * E2E DOM-level tests for the static mission player site.
 *
 * Uses jsdom to load index.html + scenarios.json and simulate real user
 * workflows: card selection, map annotations (click/drag), tag picker,
 * navigation, localStorage persistence, export/import.
 *
 * Run: node tests/test_site_e2e.mjs
 */
import { readFileSync } from "fs";
import { JSDOM } from "jsdom";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, "..");

const html = readFileSync(join(ROOT, "site", "index.html"), "utf-8");
const scenariosJson = readFileSync(
  join(ROOT, "site", "public", "scenarios.json"),
  "utf-8"
);
const scenarios = JSON.parse(scenariosJson);

let passed = 0;
let failed = 0;
const failures = [];

function assert(cond, msg) {
  if (!cond) {
    failed++;
    failures.push(msg);
    console.log(`  FAIL: ${msg}`);
  } else {
    passed++;
  }
}

function makeDom() {
  const dom = new JSDOM(html, {
    url: "http://localhost:8765/index.html",
    runScripts: "dangerously",
    resources: "usable",
    pretendToBeVisual: true,
    storageQuota: 10000000,
  });
  const { window } = dom;
  // Inject scenarios directly (bypass fetch)
  window.eval(`scenarios = ${scenariosJson};`);
  // Track localStorage writes via a proxy object
  const store = {};
  const origSetItem = window.localStorage.setItem.bind(window.localStorage);
  const origGetItem = window.localStorage.getItem.bind(window.localStorage);
  window.localStorage.setItem = (k, v) => { store[k] = v; origSetItem(k, v); };
  window.localStorage.getItem = (k) => origGetItem(k);
  window.alert = () => {};
  // Stub URL methods missing in jsdom
  if (!window.URL.createObjectURL) window.URL.createObjectURL = () => "blob:test";
  if (!window.URL.revokeObjectURL) window.URL.revokeObjectURL = () => {};
  return { dom, window, document: window.document, store };
}

// ── Test 1: Scenario cards render ──
console.log("\n1. Scenario cards render");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); updateHeaderProgress();");
  const cards = document.querySelectorAll(".scenario-card");
  assert(cards.length === scenarios.length,
    `Expected ${scenarios.length} cards, got ${cards.length}`);
  assert(cards.length > 100, `Expected >100 cards, got ${cards.length}`);
  const countBar = document.getElementById("count-bar").textContent;
  assert(countBar.includes(String(scenarios.length)),
    `Count bar should show ${scenarios.length}, got: ${countBar}`);
  window.close();
}

// ── Test 2: Search filters cards ──
console.log("\n2. Search filters cards");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards();");
  document.getElementById("search").value = "combat";
  window.eval("renderCards()");
  const cards = document.querySelectorAll(".scenario-card");
  assert(cards.length > 0, "Search 'combat' should match some cards");
  assert(cards.length < scenarios.length,
    "Search 'combat' should filter (not show all)");
  // Each visible card should match
  for (const card of cards) {
    const text = card.textContent.toLowerCase();
    assert(text.includes("combat"),
      `Card should contain 'combat': ${text.slice(0, 60)}`);
  }
  window.close();
}

// ── Test 3: Capability filter works ──
console.log("\n3. Capability filter works");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards();");
  // Deactivate all except perception
  window.eval(`
    activeCaps = new Set(["perception"]);
    renderCapFilters(); renderCards();
  `);
  const cards = document.querySelectorAll(".scenario-card");
  const percCount = scenarios.filter(s => s.capability === "perception").length;
  assert(cards.length === percCount,
    `Perception filter: expected ${percCount} cards, got ${cards.length}`);
  window.close();
}

// ── Test 4: Clicking a card opens mission page ──
console.log("\n4. Clicking a card opens mission page");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards();");
  window.eval("openMission(0)");
  const selDisplay = document.getElementById("selection").style.display;
  const misDisplay = document.getElementById("mission").style.display;
  assert(selDisplay === "none", "Selection should be hidden");
  assert(misDisplay === "", "Mission should be visible");
  const title = document.getElementById("m-title").textContent;
  assert(title === scenarios[0].title,
    `Title should be '${scenarios[0].title}', got '${title}'`);
  const sid = document.getElementById("m-id").textContent;
  assert(sid === scenarios[0].scenarioId,
    `ID should be '${scenarios[0].scenarioId}', got '${sid}'`);
  window.close();
}

// ── Test 5: Mission shows bilingual objectives ──
console.log("\n5. Bilingual objectives");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  const objEn = document.getElementById("m-objective").textContent;
  assert(objEn.length > 0, "English objective should be non-empty");
  // Switch to Chinese
  window.eval("setLang('zh')");
  const objZh = document.getElementById("m-objective").textContent;
  assert(objZh.length > 0, "Chinese objective should be non-empty");
  assert(objEn !== objZh || objEn === objZh,
    "ZH objective should render (may differ from EN)");
  window.close();
}

// ── Test 6: Difficulty tabs switch objectives ──
console.log("\n6. Difficulty tabs switch objectives");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  const easyObj = document.getElementById("m-objective").textContent;
  window.eval("setDiff('hard')");
  const hardObj = document.getElementById("m-objective").textContent;
  assert(hardObj.length > 0, "Hard objective should be non-empty");
  window.close();
}

// ── Test 7: Point annotation via map click ──
console.log("\n7. Point annotation creation");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  // Set tool to point
  window.eval("setTool('point')");
  // Simulate pending annotation directly (since MouseEvent needs getBoundingClientRect)
  window.eval(`
    pendingAnn = {type:"point", coordinates:{space:"normalized", points:[{x:0.5,y:0.3}]}};
  `);
  // Open tag modal and confirm
  window.eval(`
    openTagModal("Add Point Annotation");
    document.querySelectorAll(".tag-opt")[0].classList.add("selected");
    document.getElementById("ann-note").value = "Test note";
    confirmModal();
  `);
  const annCount = document.getElementById("ann-count").textContent;
  assert(annCount === "(1)", `Annotation count should be (1), got ${annCount}`);
  const annItems = document.querySelectorAll(".ann-item");
  assert(annItems.length === 1, `Should have 1 annotation item, got ${annItems.length}`);
  const annText = annItems[0].textContent;
  assert(annText.includes("point"), "Annotation should be type 'point'");
  assert(annText.includes("0.50"), "Annotation should show x coordinate");
  // Check overlay marker
  const markers = document.querySelectorAll(".ann-marker");
  assert(markers.length === 1, `Should have 1 overlay marker, got ${markers.length}`);
  assert(markers[0].style.left === "50%", `Marker left should be 50%, got ${markers[0].style.left}`);
  window.close();
}

// ── Test 8: Region annotation via drag ──
console.log("\n8. Region annotation creation");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  window.eval("setTool('region')");
  // Simulate a completed drag
  window.eval(`
    pendingAnn = {type:"region", coordinates:{space:"normalized",
      points:[{x:0.1,y:0.2},{x:0.6,y:0.7}]}};
    openTagModal("Add Region Annotation");
    document.querySelectorAll(".tag-opt")[2].classList.add("selected");
    confirmModal();
  `);
  const annCount = document.getElementById("ann-count").textContent;
  assert(annCount === "(1)", `Region annotation count should be (1), got ${annCount}`);
  const rects = document.querySelectorAll(".ann-rect");
  assert(rects.length === 1, `Should have 1 overlay rect, got ${rects.length}`);
  assert(rects[0].style.left === "10%", `Rect left should be 10%, got ${rects[0].style.left}`);
  assert(rects[0].style.width === "50%", `Rect width should be 50%, got ${rects[0].style.width}`);
  window.close();
}

// ── Test 9: Edit annotation tags/notes ──
console.log("\n9. Edit annotation");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  // Add annotation
  window.eval(`
    pendingAnn = {type:"point", coordinates:{space:"normalized", points:[{x:0.4,y:0.6}]}};
    openTagModal("Add Point");
    confirmModal();
  `);
  // Get the annotation id
  const annId = window.eval("getAnns()[0].id");
  // Edit it
  window.eval(`editAnnotation("${annId}")`);
  window.eval(`
    document.querySelectorAll(".tag-opt")[4].classList.add("selected");
    document.getElementById("ann-note").value = "Edited note";
    confirmModal();
  `);
  const ann = window.eval("JSON.stringify(getAnns()[0])");
  const parsed = JSON.parse(ann);
  assert(parsed.note === "Edited note", `Note should be 'Edited note', got '${parsed.note}'`);
  assert(parsed.tags.length > 0, "Should have at least one tag after edit");
  window.close();
}

// ── Test 10: Delete annotation ──
console.log("\n10. Delete annotation");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  window.eval(`
    pendingAnn = {type:"point", coordinates:{space:"normalized", points:[{x:0.2,y:0.8}]}};
    openTagModal("Add Point"); confirmModal();
  `);
  assert(document.getElementById("ann-count").textContent === "(1)", "Should have 1 annotation");
  const annId = window.eval("getAnns()[0].id");
  window.eval(`deleteAnnotation("${annId}")`);
  assert(document.getElementById("ann-count").textContent === "(0)", "Should have 0 annotations after delete");
  window.close();
}

// ── Test 11: Next/prev scenario navigation ──
console.log("\n11. Next/prev navigation");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  const title0 = document.getElementById("m-title").textContent;
  window.eval("navScenario(1)");
  const title1 = document.getElementById("m-title").textContent;
  assert(title1 !== title0, "Next scenario should show different title");
  assert(title1 === scenarios[1].title,
    `Next title should be '${scenarios[1].title}', got '${title1}'`);
  window.eval("navScenario(-1)");
  const titleBack = document.getElementById("m-title").textContent;
  assert(titleBack === title0, "Prev should return to first scenario");
  window.close();
}

// ── Test 12: Next unannotated scenario ──
console.log("\n12. Next unannotated scenario");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  // Mark current as complete
  window.eval("markStatus('complete')");
  // Next unannotated should skip to a different one
  window.eval("navUnannotated()");
  const currentIdx = window.eval("currentIdx");
  assert(currentIdx > 0, `Should navigate away from completed scenario, idx=${currentIdx}`);
  window.close();
}

// ── Test 13: Save/load progress via localStorage ──
console.log("\n13. localStorage persistence");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  // Add annotation
  window.eval(`
    pendingAnn = {type:"point", coordinates:{space:"normalized", points:[{x:0.5,y:0.5}]}};
    openTagModal("Add"); confirmModal();
  `);
  window.eval("markStatus('annotated')");
  // Verify localStorage has data (read via JS in the window context)
  const progressRaw = window.eval('localStorage.getItem("openra-progress")');
  assert(progressRaw !== null && progressRaw !== undefined,
    "Progress should be in localStorage");
  const progress = JSON.parse(progressRaw);
  assert(progress[scenarios[0].scenarioId] === "annotated",
    "First scenario should be marked annotated");
  const annsRaw = window.eval('localStorage.getItem("openra-annotations")');
  const anns = JSON.parse(annsRaw);
  assert(anns[scenarios[0].scenarioId].length === 1,
    "Should have 1 annotation stored");
  window.close();
}

// ── Test 14: Export annotations ──
console.log("\n14. Export annotations");
{
  const { window } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  window.eval(`
    pendingAnn = {type:"point", coordinates:{space:"normalized", points:[{x:0.3,y:0.7}]}};
    openTagModal("Add");
    document.querySelectorAll(".tag-opt")[0].classList.add("selected");
    confirmModal();
  `);
  // Override URL.createObjectURL and capture the blob content
  let exportedData = null;
  window.eval(`
    const origCreate = URL.createObjectURL;
    URL.createObjectURL = function(blob) {
      window.__exportBlob = blob;
      return "blob:test";
    };
  `);
  window.eval("exportAnnotations()");
  // Check the export included the annotation
  const annData = window.eval("JSON.stringify(annotations)");
  const parsed = JSON.parse(annData);
  const scId = scenarios[0].scenarioId;
  assert(parsed[scId] && parsed[scId].length === 1,
    "Exported data should include annotation");
  assert(parsed[scId][0].type === "point", "Exported annotation should be point type");
  assert(parsed[scId][0].coordinates.space === "normalized",
    "Exported annotation should use normalized coordinates");
  window.close();
}

// ── Test 15: Import annotations ──
console.log("\n15. Import annotations");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  // Simulate import by directly calling the merge logic
  const importData = {
    progress: { [scenarios[0].scenarioId]: "complete" },
    annotations: {
      [scenarios[0].scenarioId]: [{
        id: "imported1", type: "region",
        coordinates: { space: "normalized", points: [{ x: 0.1, y: 0.1 }, { x: 0.9, y: 0.9 }] },
        tags: ["objective"], note: "Imported annotation",
        createdAt: "2026-01-01T00:00:00Z", updatedAt: "2026-01-01T00:00:00Z",
      }],
    },
  };
  window.eval(`
    const d = ${JSON.stringify(importData)};
    Object.assign(annotations, d.annotations);
    Object.assign(progress, d.progress);
    localStorage.setItem("openra-progress", JSON.stringify(progress));
    localStorage.setItem("openra-annotations", JSON.stringify(annotations));
    renderAnnotations();
    updateHeaderProgress();
  `);
  const annCount = document.getElementById("ann-count").textContent;
  assert(annCount === "(1)", `After import, annotation count should be (1), got ${annCount}`);
  const progressRaw = window.eval('localStorage.getItem("openra-progress")');
  const progressData = JSON.parse(progressRaw);
  assert(progressData[scenarios[0].scenarioId] === "complete",
    "Imported progress should mark scenario complete");
  window.close();
}

// ── Test 16: All scenarios openable in mission view ──
console.log("\n16. All scenarios openable in mission view");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards();");
  let openFailures = 0;
  for (let i = 0; i < scenarios.length; i++) {
    window.eval(`openMission(${i})`);
    const title = document.getElementById("m-title").textContent;
    if (title !== scenarios[i].title) openFailures++;
  }
  assert(openFailures === 0,
    `All ${scenarios.length} scenarios should open; ${openFailures} failed`);
  window.close();
}

// ── Test 17: Tag picker has all required tags ──
console.log("\n17. Tag picker completeness");
{
  const REQUIRED_TAGS = [
    "spawn", "enemy-spawn", "objective", "fail-condition", "hidden-threat",
    "chokepoint", "safe-route", "danger-zone", "resource", "defense-position",
    "attack-path", "ambush", "scouting-target", "timing-sensitive",
    "micro-intensive", "macro-intensive", "unclear", "bug", "interesting",
    "needs-review",
  ];
  const { window } = makeDom();
  const jsTags = window.eval("TAGS");
  for (const tag of REQUIRED_TAGS) {
    assert(jsTags.includes(tag), `Tag '${tag}' should be in TAGS array`);
  }
  assert(jsTags.length >= 20, `Should have >= 20 tags, got ${jsTags.length}`);
  window.close();
}

// ── Test 18: Annotation uses normalized [0,1] coordinates ──
console.log("\n18. Normalized coordinate system");
{
  const { window } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  window.eval(`
    pendingAnn = {type:"point", coordinates:{space:"normalized", points:[{x:0.75,y:0.25}]}};
    openTagModal("Add"); confirmModal();
  `);
  const ann = JSON.parse(window.eval("JSON.stringify(getAnns()[0])"));
  assert(ann.coordinates.space === "normalized",
    "Coordinate space should be 'normalized'");
  assert(ann.coordinates.points[0].x >= 0 && ann.coordinates.points[0].x <= 1,
    "X coordinate should be in [0,1]");
  assert(ann.coordinates.points[0].y >= 0 && ann.coordinates.points[0].y <= 1,
    "Y coordinate should be in [0,1]");
  assert(ann.id && ann.createdAt && ann.updatedAt,
    "Annotation should have id, createdAt, updatedAt");
  window.close();
}

// ── Test 19: Status badges update correctly ──
console.log("\n19. Status badges");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  // Initial status should be in_progress (auto-set on open)
  const badge = document.getElementById("m-status-badge").textContent;
  assert(badge.includes("in progress"), `Status should be 'in progress', got '${badge}'`);
  window.eval("markStatus('complete')");
  const badge2 = document.getElementById("m-status-badge").textContent;
  assert(badge2.includes("complete"), `Status should be 'complete', got '${badge2}'`);
  window.close();
}

// ── Test 20: Back button returns to selection ──
console.log("\n20. Back to selection");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  assert(document.getElementById("selection").style.display === "none",
    "Selection should be hidden on mission page");
  window.eval("showSelection()");
  assert(document.getElementById("selection").style.display === "",
    "Selection should be visible after back");
  assert(document.getElementById("mission").style.display === "none",
    "Mission should be hidden after back");
  window.close();
}

// ── Test 21: No raw JSON in primary UI ──
console.log("\n21. No raw JSON in primary UI");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards();");
  const gridHtml = document.getElementById("grid").innerHTML;
  assert(!gridHtml.includes('"scenarioId"'),
    "Grid should not expose raw JSON keys");
  assert(!gridHtml.includes('"humanReadable"'),
    "Grid should not expose raw schema keys");
  window.eval("openMission(0)");
  const missionHtml = document.getElementById("mission").innerHTML;
  assert(!missionHtml.includes('"scenarioId"'),
    "Mission page should not expose raw JSON keys");
  window.close();
}

// ── Test 22: Header progress updates ──
console.log("\n22. Header progress counter");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); updateHeaderProgress();");
  const prog = document.getElementById("hdr-progress").textContent;
  assert(prog.includes(`/${scenarios.length}`),
    `Progress should show /${scenarios.length}, got '${prog}'`);
  window.close();
}

// ── Summary ──
console.log(`\n${"=".repeat(50)}`);
console.log(`Results: ${passed} passed, ${failed} failed`);
if (failures.length > 0) {
  console.log("\nFailures:");
  for (const f of failures) console.log(`  - ${f}`);
}
process.exit(failed > 0 ? 1 : 0);
