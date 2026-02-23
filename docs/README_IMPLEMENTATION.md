# Multi-Page Website Upgrade: Implementation Guide

## Overview

This document describes the upgraded multi-page website with scrollytelling and interactive histogram visualizations.

## New File Structure

```
docs/
  index.html                    (Home/overview page, updated)
  methods.html                  (Technical methods explanation)
  results.html                  (Quantitative evaluation results)
  explore.html                  (Interactive scrollytelling page)
  takeaways.html                (Discussion and conclusions)
  
  partials/
    header.html                 (Shared nav header, injected into all pages)
    footer.html                 (Shared footer, injected into all pages)
  
  js/
    scrolly.js                  (Scrollytelling module: IntersectionObserver, panel updates)
    charts.js                   (Interactive histogram visualization using Plotly)
  
  data/
    results.json                (Query metadata and scores, existing)
    histograms.json             (NEW: histogram bins and counts for all methods)
    figures.json                (NEW: step metadata and asset paths)
  
  style.css                     (Updated with design system, scrollytelling, and new layouts)
  app.js                        (Updated with partial injection, routing, and shared initialization)
```

## Features Implemented

### 1. Multi-Page Architecture with Partials

- **Header and footer** are loaded dynamically via fetch and injected into all pages
- **Active nav highlighting** automatically marks the current page link based on `data-page` attribute on `<body>`
- **Relative paths** work everywhere (compatible with GitHub Pages subpaths)

### 2. Scrollytelling Page (`explore.html`)

**Layout:**
- **Left column:** Narrative steps stacked vertically
- **Right column:** Sticky figure panel that updates as user scrolls

**Five Steps:**
1. **Project Overview** – Workflow diagram placeholder
2. **Query Explorer** – Interactive slider to browse 21 benchmark queries (reused from original)
3. **Interactive Histogram** – Compare real vs. synthetic distributions with Plotly chart
4. **Image Compare Slider** – Drag-to-compare real and synthetic plots
5. **Summary Scoreboard** – Query pass rates for all methods

**Technology:**
- IntersectionObserver detects when a step enters the viewport (40% threshold)
- On active step, `setPanel(stepId)` updates the sticky panel with new content
- Progress indicator shows "Step X of N" and a vertical progress bar
- Smooth fade-in animations between panels

### 3. Interactive Histogram Visualization (`charts.js`)

**Powered by Plotly** (via CDN: `https://cdn.plot.ly/plotly-2.30.0.min.js`)

**Features:**
- **Dataset selector:** Choose between popular_browsers, ram_utilization, display_vendors
- **Method selector:** Compare Real vs. each of 4 methods (MST, Per-Table DP-SGD, Wide-Table DP-VAE, Private Evolution)
- **Normalize toggle:** Switch between raw counts and probability distributions
- **Total Variation metric:** Automatically computed and displayed
- **Pass/Fail indicator:** Shows whether the method passed (TV below threshold)
- **Responsive:** Adapts to container width

**Data Format** (`histograms.json`):
```json
{
  "id": "dataset_id",
  "name": "Human-readable name",
  "description": "...",
  "type": "categorical" or "numeric_binned",
  "categories": [...],      // for categorical
  "bin_edges": [...],       // for numeric
  "real": [counts...],
  "methods": {
    "MST": [...],
    "PerTable_DPSGD": [...],
    "Wide_DPVAE": [...],
    "PrivateEvolution": [...]
  },
  "tv_distances": {
    "MST": 0.0403,
    ...
  },
  "metrics": {
    "passed": {"MST": 1, ...},
    "threshold": 0.15
  }
}
```

### 4. Design System & Styling

**CSS Variables:**
- `--maxw`: 1100px (main container width)
- `--pad`: 18px (padding)
- `--border`: #e6e6e6
- `--text`: #111
- `--muted`: #555
- `--bg`: #fff
- `--card`: #fafafa

**New Components:**
- `.scrolly-container`: Two-column layout (1fr + 1fr on desktop, stacks on mobile)
- `.step`: Narrative sections with left border highlight on active
- `.scrolly-panel`: Sticky figure panel
- `.score-card`: Card component for displaying query pass/fail scores
- `.figure-placeholder`: Placeholder for images (gray dashed box with text)
- `.callout`: Highlighted information box
- Active nav indicator: Underline on current page link

**Responsive:**
- Desktop (1200px+): Two-column scrollytelling
- Tablet/mobile (<1200px): Single column, sticky panel becomes static

### 5. Page-Specific Content

**index.html** – Home page
- Hero section with project overview
- Motivation section (privacy risks, differential privacy basics)
- Quick navigation cards linking to Methods, Results, Explore
- Key findings summary
- Next steps callout

**methods.html** – Technical details
- What is differential privacy (plain language)
- Approach 1: Training-based DP-SGD + VAE
- Approach 2: Training-free Private Evolution
- Baseline methods (Per-Table DP-SGD, MST)
- Evaluation framework (21 queries, metrics, thresholds)

**results.html** – Quantitative evaluation
- Overall pass rates scoreboard
- Key insights (what worked, what failed, trade-offs)
- Query-by-query breakdown (aggregate, distribution, ranking, row-level)
- Failure modes analysis (sparsity, independence, category mismatch, DP noise)
- Evaluation methodology and pass thresholds table
- Recommendations for practitioners

**explore.html** – Interactive scrollytelling
- Narrative steps with sticky figure panel
- Reuses Query Explorer (results.json slider)
- Interactive Plotly histogram
- Image compare slider
- Summary scoreboard
- Smooth scrolling experience with progress indicators

**takeaways.html** – Discussion
- Key takeaways from the study
- Comparative insights (per-table vs. wide-table, MST vs. Private Evolution)
- Recommendations for practice (when to use each method, hybrid strategy, privacy configuration, data preprocessing)
- Future directions (relational DP, workload-specific DP, improved LLMs, quantization)
- Conclusion and resources

## How to Use Locally

### Start a Server

```bash
cd /path/to/docs
python3 -m http.server 8000
```

Then visit `http://localhost:8000` in your browser.

### Test Individual Pages

- **Home:** `http://localhost:8000/index.html` (or just `/`)
- **Methods:** `http://localhost:8000/methods.html`
- **Results:** `http://localhost:8000/results.html`
- **Explore:** `http://localhost:8000/explore.html`
- **Takeaways:** `http://localhost:8000/takeaways.html`

### Test JSON Data

```bash
curl http://localhost:8000/data/results.json
curl http://localhost:8000/data/histograms.json
curl http://localhost:8000/data/figures.json
```

## GitHub Pages Deployment

The site is designed to work on GitHub Pages. Simply push the `docs/` folder to your repository, and GitHub will serve it automatically.

**Important:**
- Use relative paths (no leading `/`)
- All assets referenced should be in the `docs/` folder
- No build step needed (plain HTML, CSS, vanilla JS)

## Customizing Content

### Update Query Explorer Data

Edit [docs/data/results.json](docs/data/results.json):
- Add or modify query entries
- Update scores and thresholds
- Point to actual query figure images (currently placeholders)

### Update Histogram Data

Edit [docs/data/histograms.json](docs/data/histograms.json):
- Replace sample bins/counts with real evaluation data from your CSV files
- Add new datasets (copy the structure)
- Update TV distances and pass thresholds

### Update Page Content

All page content is in HTML files (methods.html, results.html, explore.html, takeaways.html).
- Edit text, lists, and tables directly
- Add new sections as needed
- Maintain the `<div id="siteHeader"></div>` and `<div id="siteFooter"></div>` placeholders

### Add Images

Place images in `docs/assets/` and reference them:
```html
<img src="assets/your-image.png" alt="Description" loading="lazy" />
```

For slider/placeholder images, update the paths in the HTML or scrolly.js panel rendering.

## Architecture Details

### Partial Injection (app.js)

The `injectPartials()` function runs on every page load:
1. Fetches `partials/header.html` and inserts into `#siteHeader`
2. Fetches `partials/footer.html` and inserts into `#siteFooter`
3. Calls `setActiveNav()` to highlight the current page link

### Scrollytelling (scrolly.js)

The `IntersectionObserver` monitors all `.step` elements:
- Threshold: 40% of step must be in viewport
- When a step becomes active, `setPanel(stepId)` is called
- Panel content is regenerated based on step ID
- Lazy initialization: Query Explorer and Histogram Chart are only set up when their steps activate

### Routing (app.js)

Each page has a `data-page` attribute on the `<body>` tag:
```html
<body data-page="explore">
```

On page load, the DOMContentLoaded event listener:
1. Injects partials
2. Detects the page via `data-page` attribute
3. Runs page-specific initialization (e.g., scrollytelling on explore.html)

### Charts (charts.js)

`window.renderHistogram(containerId, datasetId, method, normalize)`:
1. Fetches histograms.json
2. Finds the dataset
3. Retrieves real and synthetic counts
4. Normalizes if requested
5. Creates Plotly bar chart with dual traces (real + synthetic)
6. Displays Total Variation distance and pass/fail status

## Placeholder Images

The site references image placeholders in several places:
- `assets/hero.jpeg` – Hero section image (replace with your workflow diagram)
- `assets/slider/q_browserhist_real.png` – Browser histogram real data
- `assets/slider/q_browserhist_synth.png` – Browser histogram synthetic data

**To add your images:**
1. Save them in `docs/assets/` or `docs/assets/slider/`
2. Update image references in HTML files
3. For the explore page compare slider, update scrolly.js panel rendering

## Browser Compatibility

- Modern browsers (Chrome, Firefox, Safari, Edge)
- Requires:
  - ES6 JavaScript (async/await, template literals, etc.)
  - CSS Grid and Flexbox
  - Fetch API
  - IntersectionObserver API
  - Plotly.js from CDN

## Accessibility

- **Keyboard navigation:** Use Tab to navigate links and controls
- **Aria labels:** Image compare slider has `role="slider"` and `aria-label`
- **Semantic HTML:** Proper heading hierarchy (h1, h2, h3, etc.)
- **Color contrast:** All text meets WCAG AA standards
- **Lazy loading:** Images marked with `loading="lazy"`

## Performance Optimizations

- **Lazy image loading:** `loading="lazy"` on img tags
- **Partial-first loading:** Header/footer fetched once on all pages
- **Plotly on-demand:** Chart library only loaded on explore.html
- **Minimal JS:** No frameworks, pure vanilla JS
- **Efficient CSS:** Design system with variables, minimal redundancy

## Troubleshooting

**Issue:** "Could not load results.json" message in Query Explorer
- **Fix:** Ensure `docs/data/results.json` exists and is valid JSON
- **Check:** `curl http://localhost:8000/data/results.json`

**Issue:** Header/footer not appearing
- **Fix:** Check that `docs/partials/header.html` and `footer.html` exist
- **Check:** Browser console for fetch errors

**Issue:** Histogram chart not rendering
- **Fix:** Ensure `docs/data/histograms.json` is valid and has correct dataset IDs
- **Check:** Plotly library loaded (CDN must be accessible)

**Issue:** Scrollytelling panels not updating
- **Fix:** Check that `.step` elements have `data-step` attributes
- **Check:** Browser console for JavaScript errors in scrolly.js

## Files Reused from Original Site

- **app.js:** Original `initQueryExplorer()` and `initImageCompare()` functions preserved and enhanced
- **style.css:** Original styles preserved, new styles appended
- **data/results.json:** Unchanged, used by Query Explorer
- **assets/hero.jpeg:** Unchanged, still used on index and explore pages

## Next Steps for Completion

1. **Add real images:**
   - Replace `assets/hero.jpeg` with actual workflow diagram
   - Upload actual query comparison plots for slider examples

2. **Update histogram data:**
   - Extract real bin counts from evaluation CSV files
   - Update `docs/data/histograms.json` with actual distribution data

3. **Expand query explorer:**
   - Add more queries to `docs/data/results.json` (currently has 2 placeholders)
   - Add corresponding query figures to `docs/assets/slider/`

4. **Customize styling:**
   - Update colors in `:root` CSS variables
   - Adjust spacing and breakpoints as needed
   - Add custom branding/logo

5. **Deploy to GitHub Pages:**
   - Push `docs/` folder to GitHub
   - Pages will be live at `https://username.github.io/repo-name`

## Questions?

Refer to the code comments in:
- **app.js** – Partial injection and routing logic
- **scrolly.js** – Scrollytelling and panel management
- **charts.js** – Histogram rendering with Plotly
- **style.css** – Design tokens and component styling

All files are well-commented for future maintenance.
