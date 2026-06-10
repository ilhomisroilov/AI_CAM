/* ============================================================
   dashboard.js  —  Dashboard boshqaruvi
   - Connect / Start / Stop tugmalari
   - MJPEG live stream
   - Real-time loglar (polling)
   - Statistika yangilanishi
   ============================================================ */
(() => {
  const $ = (id) => document.getElementById(id);

  const btnConnect = $("btnConnect");
  const btnStart = $("btnStart");
  const btnStop = $("btnStop");
  const btnDisconnect = $("btnDisconnect");
  const videoFeed = $("videoFeed");
  const videoPlaceholder = $("videoPlaceholder");
  const liveBadge = $("liveBadge");
  const camDot = $("camDot");
  const logPanel = $("logPanel");

  let lastLogId = 0;

  async function post(url, body) {
    const opts = { method: "POST" };
    if (body !== undefined) {
      opts.headers = { "Content-Type": "application/json" };
      opts.body = JSON.stringify(body);
    }
    const r = await fetch(url, opts);
    return r.json();
  }

  // Oddiy IPv4 tekshiruvi
  function isValidIp(ip) {
    return /^(\d{1,3})(\.\d{1,3}){3}$/.test(ip) &&
           ip.split(".").every((o) => +o >= 0 && +o <= 255);
  }

  // ---------- Buttons ----------
  btnConnect.addEventListener("click", async () => {
    const ip = ($("cameraIp").value || "").trim();
    if (!isValidIp(ip)) {
      alert("Iltimos, to'g'ri Camera IP kiriting (masalan 192.168.1.10).");
      return;
    }
    btnConnect.disabled = true;
    btnConnect.textContent = "⏳ Connecting...";
    let res;
    try {
      // IP dinamik ravishda backendga yuboriladi -> BLOB Port 2113
      res = await post("/api/camera/connect", { ip });
    } catch (e) {
      res = { ok: false };
    }
    btnConnect.textContent = "🔌 Connect Camera";
    if (res && res.ok) {
      videoFeed.src = "/video_feed?ts=" + Date.now();
      videoPlaceholder.style.display = "none";
      camDot.className = "status-dot on";
      liveBadge.classList.add("on");
      btnStart.disabled = false;
      $("cameraIp").disabled = true;        // ulanish davomida o'zgartirib bo'lmaydi
    } else {
      btnConnect.disabled = false;
      alert(`Kamera ulanmadi (${ip}:2113). Loglarni tekshiring (IP/port/SOPAS sozlamalari).`);
    }
    refreshStatus();
  });

  btnStart.addEventListener("click", async () => {
    const res = await post("/api/processing/start");
    if (res.ok) { btnStart.disabled = true; btnStop.disabled = false; }
    refreshStatus();
  });

  btnStop.addEventListener("click", async () => {
    await post("/api/processing/stop");
    btnStart.disabled = false;
    btnStop.disabled = true;
    refreshStatus();
  });

  btnDisconnect.addEventListener("click", async () => {
    await post("/api/camera/disconnect");
    videoFeed.src = "";
    videoPlaceholder.style.display = "block";
    camDot.className = "status-dot off";
    liveBadge.classList.remove("on");
    btnConnect.disabled = false;
    btnStart.disabled = true;
    btnStop.disabled = true;
    $("cameraIp").disabled = false;        // IP ni qayta tahrirlash mumkin
    refreshStatus();
  });

  $("btnClearLog").addEventListener("click", () => { logPanel.innerHTML = ""; });

  // ---------- Logs polling ----------
  async function pollLogs() {
    try {
      const r = await fetch("/api/logs?since=" + lastLogId);
      const data = await r.json();
      for (const l of data.logs) {
        lastLogId = l.id;
        const div = document.createElement("div");
        div.className = "log-line " + l.level;
        div.innerHTML = `<span class="ts">${l.ts}</span>` +
                        `<span class="lvl">${l.level}</span>` +
                        `<span class="msg"></span>`;
        div.querySelector(".msg").textContent = l.msg;
        logPanel.appendChild(div);
      }
      // 400 satrdan oshsa eskisini tozalaymiz
      while (logPanel.children.length > 400) logPanel.removeChild(logPanel.firstChild);
      if (data.logs.length) logPanel.scrollTop = logPanel.scrollHeight;
    } catch (e) { /* sukut */ }
  }

  // Backend allaqachon ishlayotgan bo'lsa, sahifa qaytganda video qayta yoqilmasligi
  // uchun holatni kuzatamiz (model QAYTA YUKLANMAYDI — u serverda doimiy).
  let videoOn = false;

  // ---------- Status / stats ----------
  async function refreshStatus() {
    try {
      const r = await fetch("/api/status");
      const s = await r.json();
      // Statistika
      $("statFrames").textContent = s.stats.frames;
      $("statDet").textContent = s.stats.detections;
      $("statVin").textContent = s.stats.vins;
      $("statYolo").textContent = s.yolo_ready ? "Ready" : "Not loaded";
      // --- UI ni backend HOLATIGA moslash (navigatsiyadan keyin tiklash) ---
      syncUi(s.camera_connected, s.processing);
    } catch (e) { /* sukut */ }
  }

  // Tugmalar/video holatini backend bilan moslaydi (state restore)
  function syncUi(camConnected, processing) {
    if (camConnected && !videoOn) {
      videoFeed.src = "/video_feed?ts=" + Date.now();   // bir marta yoqamiz
      videoPlaceholder.style.display = "none";
      camDot.className = "status-dot on";
      liveBadge.classList.add("on");
      videoOn = true;
    } else if (!camConnected && videoOn) {
      videoFeed.src = "";
      videoPlaceholder.style.display = "block";
      camDot.className = "status-dot off";
      liveBadge.classList.remove("on");
      videoOn = false;
    }
    btnConnect.disabled = camConnected;
    $("cameraIp").disabled = camConnected;
    btnStart.disabled = (!camConnected) || processing;
    btnStop.disabled = !processing;
  }

  setInterval(pollLogs, 1000);
  setInterval(refreshStatus, 1500);
  pollLogs();
  refreshStatus();
})();
