// Option B: Query explorer driven by JSON
async function initQueryExplorer() {
  const range = document.getElementById("queryRange");
  const label = document.getElementById("queryLabel");
  const typeEl = document.getElementById("queryType");
  const descEl = document.getElementById("queryDesc");
  const scoresEl = document.getElementById("queryScores");
  const imgEl = document.getElementById("queryImg");
  const capEl = document.getElementById("queryCaption");

  let data = [];
  try {
    const res = await fetch("data/results.json");
    data = await res.json();
  } catch (e) {
    label.textContent = "Could not load results.json";
    return;
  }

  range.min = 0;
  range.max = Math.max(0, data.length - 1);
  range.value = 0;

  function render(i) {
    const q = data[i];
    label.textContent = `${i + 1} of ${data.length} (${q.name})`;
    typeEl.textContent = q.type ? `Type: ${q.type}` : "";
    descEl.textContent = q.desc || "";
    imgEl.src = q.image || "";
    capEl.textContent = q.caption || "";

    scoresEl.innerHTML = "";
    (q.scores || []).forEach(s => {
      const pill = document.createElement("div");
      pill.className = "score-pill";
      const v = (typeof s.value === "number") ? s.value.toFixed(3) : String(s.value);
      pill.innerHTML = `<span><strong>${s.method}</strong></span><span>${v}</span><span class="small">${s.note || ""}</span>`;
      scoresEl.appendChild(pill);
    });
  }

  range.addEventListener("input", () => render(Number(range.value)));
  render(0);
}

// Option A: Before/After image compare sliders
function initImageCompare() {
  const comps = document.querySelectorAll(".img-compare");

  comps.forEach(comp => {
    const overlay = comp.querySelector(".img-compare__overlay");
    const handle = comp.querySelector(".img-compare__handle");

    let dragging = false;

    function setPos(clientX) {
      const rect = comp.getBoundingClientRect();
      const x = Math.min(Math.max(clientX - rect.left, 0), rect.width);
      const pct = (x / rect.width) * 100;
      overlay.style.width = pct + "%";
      handle.style.left = `calc(${pct}% - 10px)`;
    }

    function onDown(e) {
      dragging = true;
      comp.setPointerCapture?.(e.pointerId);
      setPos(e.clientX);
    }

    function onMove(e) {
      if (!dragging) return;
      setPos(e.clientX);
    }

    function onUp() {
      dragging = false;
    }

    comp.addEventListener("pointerdown", onDown);
    comp.addEventListener("pointermove", onMove);
    window.addEventListener("pointerup", onUp);

    // Set default position
    setPos(comp.getBoundingClientRect().left + comp.getBoundingClientRect().width / 2);
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initQueryExplorer();
  initImageCompare();
});