/* ═══════════════════════════════════════════════════════
   20 Newsgroups IR — Frontend JavaScript
   ═══════════════════════════════════════════════════════ */

let prChart = null;

// ── Poll index status on load ─────────────────────────────────────────────
async function checkStatus() {
  const badge = document.getElementById('status-badge');
  try {
    const res = await fetch('/status');
    const data = await res.json();
    if (data.ready) {
      badge.textContent = '● ' + data.message;
      badge.className = 'badge badge-ready';
      document.getElementById('search-btn').disabled = false;
      loadCategories();
    } else {
      badge.textContent = '● ' + data.message;
      badge.className = 'badge badge-loading';
      setTimeout(checkStatus, 3000);
    }
  } catch {
    badge.textContent = '● Connection error';
    badge.className = 'badge badge-error';
    setTimeout(checkStatus, 5000);
  }
}

async function loadCategories() {
  try {
    const res = await fetch('/categories');
    const data = await res.json();
    const sel = document.getElementById('category-filter');
    data.categories.forEach(cat => {
      const opt = document.createElement('option');
      opt.value = cat;
      opt.textContent = cat;
      sel.appendChild(opt);
    });
  } catch { /* ignore */ }
}

// ── Search submission ─────────────────────────────────────────────────────
document.getElementById('search-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const btn = document.getElementById('search-btn');
  const overlay = document.getElementById('loading-overlay');

  btn.disabled = true;
  overlay.style.display = 'flex';

  const form = e.target;
  const formData = new FormData(form);

  try {
    const res = await fetch('/search', { method: 'POST', body: formData });
    const data = await res.json();
    if (data.error) {
      alert('⚠️ ' + data.error);
      return;
    }
    renderResults(data);
    renderMetrics(data.evaluation, data.method);
    renderPRCurve(data.pr_curves, data.method);
    document.getElementById('results-section').style.display = 'grid';
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
  } catch (err) {
    alert('Request failed: ' + err);
  } finally {
    btn.disabled = false;
    overlay.style.display = 'none';
  }
});

// ── Render search results ─────────────────────────────────────────────────
function renderResults(data) {
  const list = document.getElementById('results-list');
  const info = document.getElementById('results-info');

  info.textContent = `Showing ${data.results.length} results for "${data.query}" · method: ${data.method.toUpperCase()}`;
  list.innerHTML = '';

  if (data.results.length === 0) {
    list.innerHTML = '<p style="color:#6b7280;margin:.5rem 0">No results found.</p>';
    return;
  }

  data.results.forEach((doc, idx) => {
    const item = document.createElement('div');
    item.className = 'result-item';
    item.innerHTML = `
      <div class="result-header">
        <span class="result-rank">#${idx + 1}</span>
        <span class="result-category">${escHtml(doc.category)}</span>
        <span class="result-score">score: ${doc.score}</span>
      </div>
      <div class="result-snippet">${escHtml(doc.snippet)}…</div>
    `;
    list.appendChild(item);
  });
}

// ── Render metrics table ──────────────────────────────────────────────────
function renderMetrics(evalData, selectedMethod) {
  const tbody = document.getElementById('metrics-tbody');
  tbody.innerHTML = '';

  if (!evalData || evalData.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" style="color:#6b7280">No evaluation data.</td></tr>';
    return;
  }

  const methodOrder = { 'TFIDF': 0, 'BM25': 1, 'SBERT': 2, 'RRF': 3 };
  evalData.sort((a, b) => (methodOrder[a.method] ?? 99) - (methodOrder[b.method] ?? 99));

  evalData.forEach(row => {
    const tr = document.createElement('tr');
    const isSelected = row.method.toLowerCase() === selectedMethod.toLowerCase();
    if (isSelected) tr.classList.add('highlight');
    tr.innerHTML = `
      <td><strong>${row.method}</strong></td>
      <td>${row.precision_at_k}</td>
      <td>${row.recall_at_k}</td>
      <td>${row.map}</td>
      <td>${row.ndcg}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Render Precision-Recall curve ─────────────────────────────────────────
function renderPRCurve(prCurves, method) {
  const ctx = document.getElementById('pr-chart').getContext('2d');
  if (prChart) prChart.destroy();

  const curveData = prCurves[method];
  if (!curveData || curveData.recalls.length === 0) return;

  const labels = curveData.recalls.map(r => r.toFixed(3));
  const data   = curveData.precisions;

  prChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [{
        label: `${method.toUpperCase()} P-R Curve`,
        data,
        borderColor: '#4f46e5',
        backgroundColor: 'rgba(79,70,229,.1)',
        borderWidth: 2,
        pointRadius: 3,
        fill: true,
        tension: 0.3,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: 'Recall', font: { size: 11 } },
          ticks: { maxTicksLimit: 6, font: { size: 10 } },
        },
        y: {
          title: { display: true, text: 'Precision', font: { size: 11 } },
          min: 0, max: 1,
          ticks: { font: { size: 10 } },
        }
      },
      plugins: {
        legend: { labels: { font: { size: 11 } } },
        tooltip: {
          callbacks: {
            title: (items) => `Recall: ${items[0].label}`,
            label:  (item) => `Precision: ${item.formattedValue}`,
          }
        }
      }
    }
  });
}

// ── Utility ───────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Boot ──────────────────────────────────────────────────────────────────
document.getElementById('search-btn').disabled = true;
checkStatus();
