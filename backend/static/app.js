const textarea = document.getElementById("product-names");
const submitBtn = document.getElementById("submit-btn");
const emptyState = document.getElementById("empty-state");
const loadingState = document.getElementById("loading-state");
const errorState = document.getElementById("error-state");
const errorMessage = document.getElementById("error-message");
const clustersList = document.getElementById("clusters-list");

const STATES = [emptyState, loadingState, errorState, clustersList];

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

function renderClusters(clusters) {
  clustersList.innerHTML = "";

  clusters.forEach((cluster, index) => {
    const entry = document.createElement("article");
    entry.className = "cluster";

    const header = document.createElement("div");
    header.className = "cluster-header";

    const badge = document.createElement("span");
    badge.className = "cluster-index";
    badge.textContent = `No. ${String(index + 1).padStart(2, "0")}`;

    const confidence = document.createElement("span");
    confidence.className = "cluster-confidence";
    confidence.textContent = `${Math.round(cluster.similarity * 100)}% cocok`;

    header.append(badge, confidence);

    const canonicalLabel = document.createElement("p");
    canonicalLabel.className = "cluster-canonical-label";
    canonicalLabel.textContent = "Nama kanonik";

    const canonical = document.createElement("h2");
    canonical.className = "cluster-canonical";
    canonical.textContent = cluster.canonical_name;

    const membersLabel = document.createElement("p");
    membersLabel.className = "cluster-members-label";
    membersLabel.textContent = `${cluster.members.length} varian nama ditemukan`;

    const membersList = document.createElement("ul");
    membersList.className = "cluster-members";
    for (const member of cluster.members) {
      const li = document.createElement("li");
      li.textContent = member;
      membersList.appendChild(li);
    }

    entry.append(header, canonicalLabel, canonical, membersLabel, membersList);
    clustersList.appendChild(entry);
  });
}

async function handleSubmit() {
  const names = parseNames(textarea.value);

  if (names.length === 0) {
    errorMessage.textContent = "Masukkan minimal satu nama produk sebelum mencocokkan.";
    showState(errorState);
    return;
  }

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
    showState(clustersList);
  } catch (err) {
    errorMessage.textContent = "Gagal memproses permintaan. Periksa koneksi Anda dan coba lagi.";
    showState(errorState);
  }
}

submitBtn.addEventListener("click", handleSubmit);
