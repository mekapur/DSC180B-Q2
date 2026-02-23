/**
 * charts.js
 * Interactive histogram visualization using Plotly.
 * Compares real vs. synthetic distributions and computes Total Variation distance.
 */

(function () {
  'use strict';

  /**
   * Render a histogram comparison chart using Plotly
   * @param {string} containerId - ID of the container div
   * @param {string} datasetId - ID of the dataset (e.g., 'popular_browsers', 'ram_utilization')
   * @param {string} method - Method name (e.g., 'MST', 'PerTable_DPSGD')
   * @param {boolean} normalize - If true, show probability; else raw counts
   */
  window.renderHistogram = function (containerId, datasetId, method, normalize = false) {
    const container = document.getElementById(containerId);
    if (!container) {
      console.error('renderHistogram: container not found:', containerId);
      return;
    }

    // Fetch histogram data
    fetch('data/histograms.json')
      .then((r) => r.json())
      .then((data) => {
        const dataset = data.find((d) => d.id === datasetId);
        if (!dataset) {
          console.error('renderHistogram: dataset not found:', datasetId);
          container.innerHTML = '<p>Dataset not found</p>';
          return;
        }

        const real = dataset.real;
        const synth = dataset.methods[method];

        if (!synth) {
          console.error('renderHistogram: method not found:', method);
          container.innerHTML = '<p>Method not found</p>';
          return;
        }

        // Normalize if requested
        let realNorm = real;
        let synthNorm = synth;

        if (normalize) {
          const realSum = real.reduce((a, b) => a + b, 0);
          const synthSum = synth.reduce((a, b) => a + b, 0);
          realNorm = real.map((x) => x / (realSum || 1));
          synthNorm = synth.map((x) => x / (synthSum || 1));
        }

        // Build x-axis labels
        let xLabels;
        if (dataset.type === 'categorical') {
          xLabels = dataset.categories;
        } else if (dataset.type === 'numeric_binned') {
          xLabels = dataset.bin_edges.slice(0, -1).map((e, i) => {
            return `${e}-${dataset.bin_edges[i + 1]}`;
          });
        } else {
          xLabels = Array.from({ length: realNorm.length }, (_, i) => `Bin ${i}`);
        }

        // Create Plotly traces
        const trace1 = {
          x: xLabels,
          y: realNorm,
          name: 'Real',
          type: 'bar',
          marker: { color: 'rgba(37, 99, 235, 0.7)' }, // blue
        };

        const trace2 = {
          x: xLabels,
          y: synthNorm,
          name: `Synthetic (${method})`,
          type: 'bar',
          marker: { color: 'rgba(220, 38, 38, 0.7)' }, // red
        };

        const layout = {
          title: `${dataset.name} - Real vs. ${method}`,
          xaxis: { title: 'Category / Bin' },
          yaxis: { title: normalize ? 'Probability' : 'Count' },
          barmode: 'group',
          hovermode: 'x unified',
          margin: { t: 40, r: 20, b: 40, l: 50 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
        };

        const config = {
          responsive: true,
          displayModeBar: false,
        };

        Plotly.react(containerId, [trace1, trace2], layout, config);
      })
      .catch((e) => {
        console.error('renderHistogram: error loading data:', e);
        container.innerHTML = '<p>Error loading histogram data</p>';
      });
  };

  /**
   * Compute Total Variation distance between two distributions
   * @param {number[]} p - Real distribution (normalized)
   * @param {number[]} q - Synthetic distribution (normalized)
   * @returns {number} TV distance
   */
  window.computeTotalVariation = function (p, q) {
    if (p.length !== q.length) {
      console.error('computeTotalVariation: distributions must have same length');
      return NaN;
    }

    const sum = p.reduce((acc, pi, i) => acc + Math.abs(pi - q[i]), 0);
    return sum / 2;
  };
})();
