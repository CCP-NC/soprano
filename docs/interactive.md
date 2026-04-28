# Interactive NMR Explorer

An interactive WebAssembly-powered app for exploring NMR properties from `.magres` files — no installation required. Runs entirely in the browser via [marimo](https://marimo.io) and [Pyodide](https://pyodide.org).

**Features:**
- Upload a `.magres` file
- Visualise the crystal structure
- Select atom subsets by element, label, index, box, sphere, or bond distance
- Inspect MS tensor properties (IUPAC, Haeberlen, Herzfeld-Berger conventions)
- Simulate NMR spectra with adjustable broadening and reference shielding
- Explore relative Euler angles between MS and EFG tensors

```{raw} html
<div style="position: relative; padding-bottom: 0; height: 900px; margin: 1rem 0; border: 1px solid #e0e0e0; border-radius: 6px; overflow: hidden;">
  <iframe
    src="../_static/apps/magres_analysis.html"
    style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: none;"
    allow="cross-origin-isolated"
    loading="lazy"
    title="Interactive NMR Properties Explorer"
  ></iframe>
</div>
```

:::{note}
First load fetches Python packages (~50 MB) via Pyodide. Allow a minute on first visit. File is processed locally in your browser; nothing is sent to a server.
:::
