/* ============================================================
   history.js  —  History jadvali
   - Real-time yangilanish (auto-refresh)
   - Sana/VIN/Confidence bo'yicha saralash
   - CSV / Excel eksport
   ============================================================ */
(() => {
  const $ = (id) => document.getElementById(id);
  const recBody = $("recBody");

  let sortBy = "timestamp";
  let order = "DESC";

  function confBar(c) {
    const pct = Math.round((c || 0) * 100);
    return `<div class="conf-bar"><span style="width:${pct}%"></span></div>` +
           `<small style="color:var(--muted)">${pct}%</small>`;
  }

  async function load() {
    try {
      const r = await fetch(`/api/records?sort_by=${sortBy}&order=${order}&limit=1000`);
      const data = await r.json();
      const rows = data.records || [];
      $("totalCount").textContent = rows.length;

      if (!rows.length) {
        recBody.innerHTML = `<tr><td colspan="5" class="empty">Hali yozuvlar yo'q.</td></tr>`;
        return;
      }
      recBody.innerHTML = rows.map((r) => {
        const img = r.image_path
          ? `<img class="thumb" src="/${r.image_path}" alt="">`
          : `<span style="color:var(--muted)">—</span>`;
        return `<tr>
          <td>${r.id}</td>
          <td>${r.timestamp}</td>
          <td class="vin-cell">${r.detected_vin}</td>
          <td style="display:flex;align-items:center;gap:8px">${confBar(r.confidence)}</td>
          <td>${img}</td>
        </tr>`;
      }).join("");
    } catch (e) {
      recBody.innerHTML = `<tr><td colspan="5" class="empty">Xatolik: ${e}</td></tr>`;
    }
  }

  // Saralash (ustun sarlavhasini bosish)
  document.querySelectorAll("th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const col = th.dataset.sort;
      if (sortBy === col) { order = order === "DESC" ? "ASC" : "DESC"; }
      else { sortBy = col; order = "DESC"; }
      document.querySelectorAll("th[data-sort]").forEach((h) => {
        h.textContent = h.textContent.replace(/[ ▼▲]+$/, "");
      });
      th.textContent += order === "DESC" ? " ▼" : " ▲";
      load();
    });
  });

  $("btnRefresh").addEventListener("click", load);
  $("btnCsv").addEventListener("click", () => { window.location = "/api/export?fmt=csv"; });
  $("btnXlsx").addEventListener("click", () => { window.location = "/api/export?fmt=xlsx"; });

  // Auto-refresh (real-time)
  let timer = setInterval(() => { if ($("autoRefresh").checked) load(); }, 3000);

  load();
})();
