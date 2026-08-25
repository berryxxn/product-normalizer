const textarea = document.getElementById("product-names");
const submitBtn = document.getElementById("submit-btn");
const submitSpinner = document.getElementById("submit-spinner");
const submitLabel = document.getElementById("submit-label");
const emptyState = document.getElementById("empty-state");
const loadingState = document.getElementById("loading-state");
const errorState = document.getElementById("error-state");
const errorMessage = document.getElementById("error-message");
const resultsPanel = document.getElementById("results-panel");
const resultsSummary = document.getElementById("results-summary");
const clustersList = document.getElementById("clusters-list");

const STATES = [emptyState, loadingState, errorState, resultsPanel];

function showState(state) {
  for (const el of STATES) {
    el.classList.toggle("hidden", el !== state);
  }
}

function parseNames(raw) {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

function setLoading(isLoading) {
  submitBtn.disabled = isLoading || textarea.value.trim() === "";
  submitSpinner.classList.toggle("hidden", !isLoading);
  submitLabel.textContent = isLoading ? "Memproses..." : "Cocokkan Produk";
}

function renderClusters(clusters) {
  clustersList.innerHTML = "";
  resultsSummary.textContent = `${clusters.length} kelompok produk ditemukan`;

  clusters.forEach((cluster, index) => {
    const card = document.createElement("article");
    card.className = "cluster-card";

    const header = document.createElement("div");
    header.className = "cluster-header";

    const badge = document.createElement("span");
    badge.className = "cluster-badge";
    badge.textContent = `Kelompok ${index + 1}`;

    const similarity = document.createElement("span");
    similarity.className = "cluster-similarity";
    similarity.textContent = `${Math.round(cluster.similarity * 100)}% kemiripan`;

    header.append(badge, similarity);

    const canonical = document.createElement("h2");
    canonical.className = "cluster-canonical";
    canonical.textContent = cluster.canonical_name;

    const membersList = document.createElement("ul");
    membersList.className = "cluster-members";
    for (const member of cluster.members) {
      const li = document.createElement("li");

      const memberText = document.createElement("span");
      memberText.className = "member-text";
      memberText.textContent = member;
      li.appendChild(memberText);

      if (member === cluster.canonical_name) {
        const canonicalBadge = document.createElement("span");
        canonicalBadge.className = "canonical-badge";
        canonicalBadge.title = "Nama kanonis";
        canonicalBadge.textContent = "✓";
        li.appendChild(canonicalBadge);
      }

      membersList.appendChild(li);
    }

    card.append(header, canonical, membersList);
    clustersList.appendChild(card);
  });
}

async function handleSubmit() {
  const names = parseNames(textarea.value);

  if (names.length === 0) {
    errorMessage.textContent = "Masukkan minimal satu nama produk sebelum mencocokkan.";
    showState(errorState);
    return;
  }

  setLoading(true);
  showState(loadingState);

  try {
    const response = await fetch("/normalize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ names }),
    });

    if (!response.ok) {
      throw new Error(`status ${response.status}`);
    }

    const data = await response.json();
    renderClusters(data.clusters);
    showState(resultsPanel);
  } catch (err) {
    errorMessage.textContent = "Gagal memproses permintaan. Periksa koneksi Anda dan coba lagi.";
    showState(errorState);
  } finally {
    setLoading(false);
  }
}

submitBtn.addEventListener("click", handleSubmit);
textarea.addEventListener("input", () => {
  submitBtn.disabled = textarea.value.trim() === "";
});

setLoading(false);
