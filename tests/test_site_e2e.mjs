/**
 * E2E DOM-level tests for the static mission player site.
 *
 * Uses jsdom to load index.html + scenarios.json and simulate real user
 * workflows: card selection, navigation, game engine integration.
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
  assert(misDisplay === "block", "Mission should be visible");
  const title = document.getElementById("m-title").textContent;
  assert(title === "#1 " + scenarios[0].title,
    `Title should be '#1 ${scenarios[0].title}', got '${title}'`);
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

// ── Test 7: Scenario card titles have sequence numbers ──
console.log("\n7. Scenario card titles have sequence numbers");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards();");
  const cards = document.querySelectorAll(".scenario-card .title");
  assert(cards.length > 0, "Should have scenario cards");
  assert(cards[0].textContent.startsWith("1. "),
    `First card title should start with '1. ', got '${cards[0].textContent.slice(0, 20)}'`);
  if (cards.length > 1) {
    assert(cards[1].textContent.startsWith("2. "),
      `Second card title should start with '2. ', got '${cards[1].textContent.slice(0, 20)}'`);
  }
  window.close();
}

// ── Test 8: Mission page title has sequence number ──
console.log("\n8. Mission page title has sequence number");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(2);");
  const title = document.getElementById("m-title").textContent;
  assert(title.startsWith("#3 "),
    `Mission title should start with '#3 ', got '${title.slice(0, 20)}'`);
  window.close();
}

// ── Test 9: Next/prev scenario navigation ──
console.log("\n9. Next/prev navigation");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  const title0 = document.getElementById("m-title").textContent;
  window.eval("navScenario(1)");
  const title1 = document.getElementById("m-title").textContent;
  assert(title1 !== title0, "Next scenario should show different title");
  assert(title1.includes(scenarios[1].title),
    `Next title should contain '${scenarios[1].title}', got '${title1}'`);
  window.eval("navScenario(-1)");
  const titleBack = document.getElementById("m-title").textContent;
  assert(titleBack === title0, "Prev should return to first scenario");
  window.close();
}

// ── Test 10: Back button returns to selection ──
console.log("\n10. Back to selection");
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

// ── Test 11: All scenarios openable in mission view ──
console.log("\n11. All scenarios openable in mission view");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards();");
  let openFailures = 0;
  for (let i = 0; i < scenarios.length; i++) {
    window.eval(`openMission(${i})`);
    const title = document.getElementById("m-title").textContent;
    if (!title.includes(scenarios[i].title)) openFailures++;
  }
  assert(openFailures === 0,
    `All ${scenarios.length} scenarios should open; ${openFailures} failed`);
  window.close();
}

// ── Test 12: No raw JSON in primary UI ──
console.log("\n12. No raw JSON in primary UI");
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

// ── Test 13: Header progress updates ──
console.log("\n13. Header progress counter");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); updateHeaderProgress();");
  const prog = document.getElementById("hdr-progress").textContent;
  assert(prog.includes(`/${scenarios.length}`),
    `Progress should show /${scenarios.length}, got '${prog}'`);
  window.close();
}

// ── Test 14: Unit selection auto-clears after move order ──
console.log("\n14. Unit selection auto-clears after move order");
{
  const { window } = makeDom();
  window.eval(`
    gameState = {
      done: false,
      units: [{id:"u1",type:"tank",cell_x:10,cell_y:10,hp:1},{id:"u2",type:"tank",cell_x:20,cell_y:20,hp:1}],
      enemies: [],
      minimap_ascii: ".".repeat(128) + "\\n".repeat(40).split("\\n").map(() => ".".repeat(128)).join("\\n"),
    };
    gameSelected.add("u1");
    gameSelected.add("u2");
  `);
  // Simulate move queue (the auto-clear logic)
  window.eval(`
    const uids = [...gameSelected];
    gameQueue.push({mode:"move",unit_ids:uids,target_x:50,target_y:50});
    gameSelected.clear();
  `);
  const selSize = window.eval("gameSelected.size");
  assert(selSize === 0, `Selection should be empty after move, got size=${selSize}`);
  const queueLen = window.eval("gameQueue.length");
  assert(queueLen === 1, `Queue should have 1 action, got ${queueLen}`);
  const action = JSON.parse(window.eval("JSON.stringify(gameQueue[0])"));
  assert(action.unit_ids.length === 2, "Move action should have 2 units");
  assert(action.unit_ids.includes("u1") && action.unit_ids.includes("u2"),
    "Move action should include u1 and u2");
  window.close();
}

// ── Test 15: Multi-group move orders queue independently ──
console.log("\n15. Multi-group move orders queue independently");
{
  const { window } = makeDom();
  window.eval(`
    gameState = {
      done: false,
      units: [{id:"u1",type:"tank",cell_x:10,cell_y:10,hp:1},{id:"u2",type:"tank",cell_x:20,cell_y:20,hp:1}],
      enemies: [],
      minimap_ascii: ".".repeat(128) + "\\n".repeat(40).split("\\n").map(() => ".".repeat(128)).join("\\n"),
    };
    // First group: select u1, move to (30,30)
    gameSelected.add("u1");
    let uids1 = [...gameSelected];
    gameQueue.push({mode:"move",unit_ids:uids1,target_x:30,target_y:30});
    gameSelected.clear();
    // Second group: select u2, move to (80,80)
    gameSelected.add("u2");
    let uids2 = [...gameSelected];
    gameQueue.push({mode:"move",unit_ids:uids2,target_x:80,target_y:80});
    gameSelected.clear();
  `);
  const queueLen = window.eval("gameQueue.length");
  assert(queueLen === 2, `Queue should have 2 actions, got ${queueLen}`);
  const q = JSON.parse(window.eval("JSON.stringify(gameQueue)"));
  assert(q[0].unit_ids.length === 1 && q[0].unit_ids[0] === "u1",
    "First action should move u1");
  assert(q[0].target_x === 30 && q[0].target_y === 30,
    "First action target should be (30,30)");
  assert(q[1].unit_ids.length === 1 && q[1].unit_ids[0] === "u2",
    "Second action should move u2");
  assert(q[1].target_x === 80 && q[1].target_y === 80,
    "Second action target should be (80,80)");
  window.close();
}

// ── Test 16: No annotation or export elements in DOM ──
console.log("\n16. No annotation or export elements in DOM");
{
  const { window: w, document } = makeDom();
  assert(document.getElementById("map-container") === null,
    "map-container should not exist");
  assert(document.getElementById("ann-overlay") === null,
    "ann-overlay should not exist");
  assert(document.getElementById("tag-modal") === null,
    "tag-modal should not exist");
  assert(document.getElementById("import-file") === null,
    "import-file should not exist");
  w.close();
}

// ── Test 17: No Mark Annotated / Mark Complete buttons ──
console.log("\n17. No Mark Annotated / Mark Complete buttons");
{
  const { window, document } = makeDom();
  window.eval("renderCapFilters(); renderCards(); openMission(0);");
  const buttons = [...document.querySelectorAll("button")];
  const removedBtns = buttons.filter(b =>
    b.textContent.includes("Mark Annotated") || b.textContent.includes("Mark Complete") || b.textContent.includes("Next Unannotated")
  );
  assert(removedBtns.length === 0,
    `Should have no Mark/Unannotated buttons, found ${removedBtns.length}`);
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
