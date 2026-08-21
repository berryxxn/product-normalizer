const textarea = document.getElementById("product-names");
const submitBtn = document.getElementById("submit-btn");

const states = {
  empty: document.getElementById("empty-state"),
  loading: document.getElementById("loading-state"),
  error: document.getElementById("error-state"),
  clusters: document.getElementById("clusters-list"),
};

function showState(name) {
  for (const key in states) {
    states[key].classList.toggle("hidden", key !== name);
  }
}

function renderClusters(clusters) {
  states.clusters.innerHTML = "";

  clusters.forEach((cluster, index) => {
    const entry = document.createElement("div");
    entry.className = "cluster-entry";

    const membersList = cluster.members
      .map((member) => `<li>${escapeHtml(member)}</li>`)
      .join("");

    entry.innerHTML = `
      <div class="cluster-index">Kelompok ${index + 1}</div>
      <div class="cluster-canonical">${escapeHtml(cluster.canonical_name)}</div>
      <ul class="cluster-members">${membersList}</ul>
      <span class="cluster-similarity">Kemiripan: ${(cluster.similarity * 100).toFixed(0)}%</span>
    `;

    states.clusters.appendChild(entry);
  });

  showState("clusters");
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

async function handleSubmit() {
  const names = textarea.value
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  if (names.length === 0) {
    showState("empty");
    return;
  }

  showState("loading");

  try {
    const response = await fetch("/normalize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });

    if (!response.ok) {
      throw new Error(`Server merespons dengan status ${response.status}`);
    }

    const data = await response.json();
    renderClusters(data.clusters);
  } catch (err) {
    document.getElementById("error-message").textContent =
      "Gagal memproses daftar produk: " + err.message;
    showState("error");
  }
}

submitBtn.addEventListener("click", handleSubmit);
