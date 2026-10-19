/**
 * VisiTrace — Visitor Intelligence System
 * Frontend interactive controller.
 */

// State
let currentFilter = "";
let pollInterval = null;

// Helpers
function formatDateTime(isoString) {
  if (!isoString) return "—";
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return isoString;
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
           ' (' + d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ')';
  } catch (e) {
    return isoString;
  }
}

function showToast(message, isError = false) {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${isError ? 'error' : ''}`;
  toast.innerText = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = '0';
    setTimeout(() => toast.remove(), 200);
  }, 3500);
}

function openModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.add("open");
}

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.classList.remove("open");
}

// Fetch and render Stats
async function fetchStats() {
  try {
    const res = await fetch("/api/stats");
    if (!res.ok) throw new Error("Failed to fetch stats");
    const data = await res.json();

    document.getElementById("kpi-unique-visitors").innerText = data.unique_visitors;
    document.getElementById("kpi-active-visits").innerText = data.active_visits;
    document.getElementById("kpi-total-entries").innerText = data.total_entries;
    document.getElementById("kpi-total-exits").innerText = data.total_exits;
    document.getElementById("kpi-today-visitors").innerText = data.today_unique_visitors;
  } catch (e) {
    console.error("fetchStats error:", e);
  }
}

// Fetch and render Stream Status
async function fetchStreamStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error("Failed to fetch status");
    const data = await res.json();

    const dot = document.getElementById("system-status-dot");
    const statusText = document.getElementById("system-status-text");
    const fpsOverlay = document.getElementById("overlay-fps");
    const tracksOverlay = document.getElementById("overlay-tracks");
    const sourceLabel = document.getElementById("viewport-source-name");

    if (data.is_running) {
      dot.className = "status-dot active";
      statusText.innerText = `Active (${data.source_type})`;
      fpsOverlay.innerText = `FPS: ${data.fps}`;
      tracksOverlay.innerText = `Tracks: ${data.active_tracks}`;
      sourceLabel.innerText = data.source ? data.source.split(/[\\/]/).pop() : "Stream Active";
    } else {
      dot.className = "status-dot";
      statusText.innerText = "Standby (Idle)";
      fpsOverlay.innerText = "FPS: 0.0";
      tracksOverlay.innerText = "Tracks: 0";
      sourceLabel.innerText = "Standby";
    }

    if (data.last_error) {
      const banner = document.getElementById("stream-message-banner");
      banner.style.display = "block";
      banner.style.background = "var(--accent-amber-tint)";
      banner.style.color = "var(--status-error)";
      banner.innerText = `Notice: ${data.last_error}`;
    }
  } catch (e) {
    console.error("fetchStreamStatus error:", e);
  }
}

// Fetch and render Events Timeline
async function fetchEvents() {
  try {
    let url = "/api/events?limit=40";
    if (currentFilter) {
      url += `&event_type=${encodeURIComponent(currentFilter)}`;
    }
    const res = await fetch(url);
    if (!res.ok) throw new Error("Failed to fetch events");
    const events = await res.json();

    const container = document.getElementById("events-list-container");
    const countLabel = document.getElementById("events-count-label");
    countLabel.innerText = `${events.length} events logged`;

    if (events.length === 0) {
      container.innerHTML = `
        <div class="empty-state">
          <div class="empty-state-icon">📡</div>
          <div class="empty-state-text">No visitor events recorded yet.<br>Start stream processing to observe faces.</div>
        </div>
      `;
      return;
    }

    let html = "";
    for (const ev of events) {
      const isEntry = ev.event_type === "entry";
      const badgeClass = isEntry ? "badge-entry" : "badge-exit";
      const badgeText = isEntry ? "Entry" : "Exit";
      const thumbUrl = ev.image_path ? `/${ev.image_path}` : "";

      html += `
        <div class="event-row">
          <img class="event-thumb" src="${thumbUrl}" alt="${ev.visitor_id}" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'48\\' height=\\'48\\' viewBox=\\'0 0 24 24\\' fill=\\'none\\' stroke=\\'%23999\\' stroke-width=\\'1.5\\'><circle cx=\\'12\\' cy=\\'8\\' r=\\'4\\'/><path d=\\'M4 20c0-4 4-6 8-6s8 2 8 6\\'/></svg>'">
          <div class="event-details">
            <div class="event-header-line">
              <span class="visitor-id-tag">${ev.visitor_id}</span>
              <span class="event-type-badge ${badgeClass}">${badgeText}</span>
              <span style="font-size: 0.74rem; color: var(--text-muted); font-family: var(--font-mono);">[${ev.visit_id}]</span>
            </div>
            <div class="event-meta">
              <span class="event-time">${formatDateTime(ev.timestamp)}</span>
              ${ev.confidence ? `<span>Conf: ${(ev.confidence * 100).toFixed(0)}%</span>` : ''}
              ${ev.track_id ? `<span>Track: ${ev.track_id}</span>` : ''}
            </div>
          </div>
        </div>
      `;
    }
    container.innerHTML = html;
  } catch (e) {
    console.error("fetchEvents error:", e);
  }
}

// Fetch and render Visits
async function fetchVisits() {
  try {
    const res = await fetch("/api/visits?limit=15");
    if (!res.ok) throw new Error("Failed to fetch visits");
    const visits = await res.json();

    const tbody = document.getElementById("visits-table-body");
    if (visits.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" class="empty-state">No visit sessions recorded yet.</td></tr>`;
      return;
    }

    let html = "";
    for (const v of visits) {
      const durationStr = v.duration_seconds != null ? `${v.duration_seconds.toFixed(1)}s` : "In progress";
      const statusBadge = v.status === "active" 
        ? `<span class="event-type-badge badge-entry">Active</span>` 
        : `<span class="event-type-badge" style="background: var(--bg-subtle); color: var(--text-secondary);">Completed</span>`;

      html += `
        <tr>
          <td class="font-mono"><strong>${v.visit_id}</strong></td>
          <td class="font-mono">${v.visitor_id}</td>
          <td>${formatDateTime(v.entry_at)}</td>
          <td>${formatDateTime(v.exit_at)}</td>
          <td class="font-mono">${durationStr}</td>
          <td>${statusBadge}</td>
        </tr>
      `;
    }
    tbody.innerHTML = html;
  } catch (e) {
    console.error("fetchVisits error:", e);
  }
}

// Fetch Visitors Directory
async function fetchVisitorsDirectory() {
  try {
    const res = await fetch("/api/visitors?limit=100");
    if (!res.ok) throw new Error("Failed to fetch visitors");
    const visitors = await res.json();

    const tbody = document.getElementById("visitors-directory-body");
    if (visitors.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-state">No visitors registered yet.</td></tr>`;
      return;
    }

    let html = "";
    for (const v of visitors) {
      html += `
        <tr>
          <td class="font-mono"><strong>${v.visitor_id}</strong></td>
          <td>${formatDateTime(v.first_seen_at)}</td>
          <td>${formatDateTime(v.last_seen_at)}</td>
          <td class="font-mono">${v.total_visits}</td>
          <td class="font-mono">${v.quality_score ? v.quality_score.toFixed(1) : '—'}</td>
        </tr>
      `;
    }
    tbody.innerHTML = html;
    openModal("modal-visitors");
  } catch (e) {
    showToast("Failed to load visitors directory: " + e.message, true);
  }
}

// Fetch Diagnostics / Health
async function fetchDiagnostics() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    document.getElementById("health-json-view").innerText = JSON.stringify(data, null, 2);
    openModal("modal-health");
  } catch (e) {
    showToast("Failed to load diagnostics: " + e.message, true);
  }
}

// Fetch Configuration
async function fetchConfig() {
  try {
    const res = await fetch("/api/config");
    const data = await res.json();
    document.getElementById("config-json-view").innerText = JSON.stringify(data, null, 2);
    openModal("modal-config");
  } catch (e) {
    showToast("Failed to load config: " + e.message, true);
  }
}

// Start Stream Action
async function handleStartStream() {
  const typeSelect = document.getElementById("select-source-type");
  const pathInput = document.getElementById("input-source-path");
  const banner = document.getElementById("stream-message-banner");

  const sourceType = typeSelect.value;
  const sourcePath = pathInput.value.trim();

  if (!sourcePath) {
    showToast("Please enter a valid video path or RTSP URL", true);
    return;
  }

  banner.style.display = "none";
  showToast("Initiating stream connection...");

  try {
    const res = await fetch("/api/streams/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        source_type: sourceType,
        source_path: sourcePath,
        loop: true
      })
    });

    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.detail || data.message || "Failed to start stream");
    }

    showToast("Stream started successfully");
    // Refresh MJPEG stream element to reconnect
    const img = document.getElementById("live-stream-img");
    img.src = `/api/streams/preview?t=${Date.now()}`;
    refreshAllData();
  } catch (e) {
    showToast(e.message, true);
    banner.style.display = "block";
    banner.style.background = "var(--accent-amber-tint)";
    banner.style.color = "var(--status-error)";
    banner.innerText = `Error: ${e.message}`;
  }
}

// Stop Stream Action
async function handleStopStream() {
  showToast("Stopping stream...");
  try {
    const res = await fetch("/api/streams/stop", { method: "POST" });
    const data = await res.json();
    showToast(data.message || "Stream stopped");
    refreshAllData();
  } catch (e) {
    showToast(e.message, true);
  }
}

// Refresh all live data
function refreshAllData() {
  fetchStats();
  fetchStreamStatus();
  fetchEvents();
  fetchVisits();
}

// Event Listeners
document.addEventListener("DOMContentLoaded", () => {
  // Navigation & buttons
  document.getElementById("btn-refresh").addEventListener("click", () => {
    refreshAllData();
    showToast("Dashboard refreshed");
  });

  document.getElementById("btn-show-visitors").addEventListener("click", fetchVisitorsDirectory);
  document.getElementById("btn-show-health").addEventListener("click", fetchDiagnostics);
  document.getElementById("btn-show-config").addEventListener("click", fetchConfig);

  // Stream controls
  document.getElementById("btn-start-stream").addEventListener("click", handleStartStream);
  document.getElementById("btn-stop-stream").addEventListener("click", handleStopStream);

  document.getElementById("btn-load-sample").addEventListener("click", () => {
    document.getElementById("select-source-type").value = "file";
    document.getElementById("input-source-path").value = "data/sample_videos/video_sample1.mp4";
    showToast("Loaded Katomaran sample video path");
  });

  document.getElementById("select-source-type").addEventListener("change", (e) => {
    const pathInput = document.getElementById("input-source-path");
    if (e.target.value === "rtsp") {
      pathInput.placeholder = "rtsp://username:password@192.168.1.100:554/stream1";
      if (!pathInput.value.startsWith("rtsp")) {
        pathInput.value = "rtsp://127.0.0.1:8554/live";
      }
    } else {
      pathInput.placeholder = "e.g. data/sample_videos/video_sample1.mp4";
      pathInput.value = "data/sample_videos/video_sample1.mp4";
    }
  });

  // Filter pills
  document.querySelectorAll(".filter-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      document.querySelectorAll(".filter-pill").forEach(p => p.classList.remove("active"));
      pill.classList.add("active");
      currentFilter = pill.getAttribute("data-filter");
      fetchEvents();
    });
  });

  // Initial load
  refreshAllData();

  // Periodic polling every 2 seconds
  pollInterval = setInterval(refreshAllData, 2000);
});
