import { useState, type FormEvent } from 'react';
import './App.css';

interface Cluster {
  canonical_name: string;
  members: string[];
  similarity: number;
}

interface NormalizeResponse {
  clusters: Cluster[];
}

function App() {
  const [inputText, setInputText] = useState('');
  const [clusters, setClusters] = useState<Cluster[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const names = inputText
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line.length > 0);

    if (names.length === 0) {
      setError('Masukkan minimal satu nama produk');
      return;
    }

    setLoading(true);
    setError(null);
    setClusters([]);

    try {
      const response = await fetch('/normalize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ names }),
      });

      if (!response.ok) {
        throw new Error(`Server merespons dengan status ${response.status}`);
      }

      const data: NormalizeResponse = await response.json();
      setClusters(data.clusters);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Gagal memproses daftar produk');
    } finally {
      setLoading(false);
    }
  };

  const handlePaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const text = e.clipboardData.getData('text');
    setInputText(text);
  };

  return (
    <main className="app">
      <header className="app-header">
        <h1>Normalisasi Nama Produk</h1>
        <p>Tempel daftar nama produk dari berbagai supplier, satu nama per baris.</p>
      </header>

      <form onSubmit={handleSubmit} className="input-panel">
        <label htmlFor="product-names">Daftar nama produk</label>
        <textarea
          id="product-names"
          rows={12}
          placeholder="Indomie Goreng 85gr&#10;Mie Goreng Indomi 85g&#10;INDOMIE GRNG 85 GR&#10;Sari Roti Tawar 400gr&#10;Roti Tawar Sari Roti 400g&#10;Teh Botol Sosro 400ml&#10;Teh Kotak Sosro 400 ml"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          onPaste={handlePaste}
          disabled={loading}
        />
        <div className="input-footer">
          <span className="input-hint">Tip: Tempel langsung dari Excel, WhatsApp, atau nota supplier</span>
          <button type="submit" disabled={loading || inputText.trim() === ''} className="submit-btn">
            {loading ? (
              <>
                <span className="spinner" aria-hidden="true"></span>
                Memproses...
              </>
            ) : (
              'Cocokkan Produk'
            )}
          </button>
        </div>
      </form>

      {error && (
        <div className="error-state" role="alert">
          <svg className="error-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <p>{error}</p>
        </div>
      )}

      {clusters.length > 0 && (
        <section className="results-panel" aria-label="Hasil normalisasi">
          <div className="results-summary">
            <span>{clusters.length} kelompok produk ditemukan</span>
          </div>
          <div className="clusters-grid">
            {clusters.map((cluster, index) => (
              <article key={index} className="cluster-card">
                <header className="cluster-header">
                  <span className="cluster-badge">Kelompok {index + 1}</span>
                  <span className="cluster-similarity">{Math.round(cluster.similarity * 100)}% kemiripan</span>
                </header>
                <h2 className="cluster-canonical">{cluster.canonical_name}</h2>
                <ul className="cluster-members" role="list">
                  {cluster.members.map((member, i) => (
                    <li key={i}>
                      <span className="member-text">{member}</span>
                      {i === 0 && <span className="canonical-badge" title="Nama kanonis">✓</span>}
                    </li>
                  ))}
                </ul>
              </article>
            ))}
          </div>
        </section>
      )}

      {clusters.length === 0 && !loading && !error && (
        <div className="empty-state">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8" />
            <polyline points="14 2 14 8 20 8" />
            <line x1="16" y1="13" x2="8" y2="13" />
            <line x1="16" y1="17" x2="8" y2="17" />
            <polyline points="10 9 9 9 8 9" />
          </svg>
          <p>Belum ada hasil. Masukkan daftar nama produk lalu klik "Cocokkan Produk".</p>
        </div>
      )}
    </main>
  );
}

export default App;