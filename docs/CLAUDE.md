# CLAUDE.md

Persistent context file for AI agents working on this repository. Keep concise, full-breadth, and comprehensive.

## Repository meta

- **Tooling**: Use `uv` (not raw `python3`/`pip`) for dependency management and running scripts. The Q1 submodule already uses `uv` with a `pyproject.toml`.
- **Language**: Python (PyTorch, Opacus). LaTeX for reports (XeLaTeX + BibTeX, custom `dsc180reportstyle`).
- **Structure**:
  - `dsc-180a-q1/` — Git submodule (Q1 code). Remote: `git@github.com-jktrns:jktrns/dsc-180a-q1`. Contains DP-VAE implementation, notebooks, and telemetry data.
  - `docs/` — Papers (TXT dumps), query benchmark (JSON/SQL), LaTeX reports, and this file.
  - `docs/queries/` — 22 SQL queries (JSON format with `question`, `sql`, `db_id` fields) representing the Intel DCA benchmark workload.
  - `docs/papers/` — Reference papers (text dumps) and reading notes.

## Project overview

- **Course**: DSC 180A/B (UCSD Data Science Capstone), two-quarter sequence. Q1 = Fall 2025, Q2 = Winter 2026.
- **Team**: Mehak Kapur, Hana Tjendrawasi, Jason Tran (jat037, GitHub: jktrns), Phuc Tran. Mentor: Yu-Xiang Wang.
- **Topic**: Differentially private synthetic data generation — comparing **training-based** (DP-SGD) vs. **training-free** (Private Evolution) methods on real telemetry data.
- **Central research question**: Can training-free methods match the statistical fidelity of training-based approaches on real analytical workloads?

## Q1 work (dsc-180a-q1/)

- Built a **DP-VAE** (differentially private variational autoencoder) trained with DP-SGD on a 152,356-row synthetic telemetry table (product family A-F/Others, event labels, timestamps).
- Hyperparams: batch 256, lr 3e-3, noise multiplier 1.0, clip norm 1.0, latent dim 16, up to 30 epochs, privacy budget epsilon=4.0, delta=1e-5.
- Key results: categorical marginals within ~8% absolute deviation, downstream logistic regression accuracy 0.380 (vs 0.381 real-only baseline). Rare classes (product E, event error/reset) underrepresented due to DP-SGD gradient clipping bias.
- Libraries: PyTorch + Opacus for DP-SGD.
- Report: `docs/q1-report.tex`.

## Q2 proposal (docs/q2-proposal.tex)

- **Title**: "Training-based versus training-free differential privacy for data synthesis"
- **Data**: Intel Driver and Client Applications (DCA) telemetry corpus — ~30 interrelated tables, keyed on `guid` (per-client). Categories: client metadata, power/thermal, battery/mobility, application behavior, web browsing, network/memory, display.
- **Benchmark**: 22 SQL queries from Intel analysts covering aggregate stats with joins, ranked top-k, geographic/demographic breakdowns, histograms/distributions, and complex multi-way pivots.
- **Evaluation metric**: Query discrepancy Delta_j using appropriate distance per result type (relative error for scalars, Spearman's rho for rankings, total variation for histograms). Aggregate score = fraction of queries passing tolerance thresholds.
- **Training-based approach**: DP-SGD with PyTorch + Opacus (continuation of Q1 work, now on real DCA data).
- **Training-free approach**: Private Evolution (PE) using black-box API access to foundation models (e.g., LLMs for tabular).
- **Research questions**: (1) Which method scores higher under matched epsilon/delta? (2) Does error compound across multi-table joins? (3) Which preserves minority classes better? (4) Downstream classifier comparability? (5) Wall-clock cost comparison?
- **Expected outputs**: Technical report, SQL benchmark suite, two synthetic DCA datasets (one DP-SGD, one PE), project website, open-source code.

## DCA data infrastructure

### Storage and access

- **Total size**: 9.1 TB, stored in HDSI Industry Data Repository.
- **Format**: Parquet files (readable with `pyarrow.parquet`). Each table in a subfolder, possibly split into multiple Parquet files up to 6.2 GB each.
- **Manifest files**: `{table_name}-manifest.json` contain URLs, file sizes, row counts, schema, and total stats. Originally exported from Amazon Redshift.
- **Transfer**: Via Globus. Collection = "HDSI Industry Data Repository", Path = `/projects/dca/`. Requires Globus account (UCSD SSO for UCSD users) and membership group invitation. Use Globus Connect Personal to transfer to local device. Guides folder in the repository contains data dictionary, table summary, and SOPs. Contacts: Sophia Tran (sotran@ucsd.edu), Rick Wagner (rick@sdsc.edu). See `docs/globus-documentation.txt` for full setup instructions.
- **Reading**: `import pyarrow.parquet as pq; dataset = pq.ParquetDataset('table_folder/'); df = dataset.read().to_pandas()`. Supports filtering during read.

### Three database schemas

**1. university_analysis_pad** (data dictionary metadata):
- `data_dictionary` (2,248 rows, 40 KiB) — column-level metadata: table_name, column_name, data_type, ordinal_position, description.
- `data_dictionary_collector_il` (24 rows) — input library descriptions.
- `data_dictionary_collector_inputs` (321 rows) — collector input definitions.
- `data_dictionary_tables` (116 rows) — table-level descriptions.

**2. university_prod** (main telemetry data, ~115 tables):
Key tables relevant to the SQL benchmark:
- `system_sysinfo_unique_normalized` (1M rows, 74 MiB) — client attributes: chassis type, country, OEM, RAM, CPU family/generation, graphics, screen size, persona.
- `batt_acdc_events` (972M rows, 34 GiB) — battery AC/DC state durations and battery percentage.
- `batt_info` (163M rows, 10 GiB) — battery chemistry, capacity, charge cycles.
- `hw_pack_run_avg_pwr` (222M rows, 18 GiB) — running average system power histograms (Watts), daily.
- `hw_metric_stats` (449M rows, 8.7 GiB) — daily measurements of hardware metrics.
- `hw_metric_hourly_stats` (509M rows, 9.5 GiB) — hourly hardware metric measurements.
- `hw_cpu_frequency` (669M rows, 4.3 GiB) — IA frequency (MHz) and number of active cores.
- `os_c_state` (2B rows, 67 GiB) — CPU C-state residency (C0, C1, C3, C6, C7) at daily granularity.
- `os_p_state` (6.3B rows, 48 GiB) — CPU clock frequency when in C0, daily.
- `os_memsam_avail_percent` (174M rows, 15 GiB) — available memory percentage, daily.
- `os_network_consumption` (2.9B rows, 55 GiB) — hourly network bytes/sec breakdown.
- `frgnd_backgrnd_apps` (361M rows, 4.9 GiB) — foreground apps + top N background processes.
- `frgnd_system_usage_by_app` (16B rows, 338 GiB) — hardware metrics per application.
- `fps` (764M rows, 26 GiB) — DirectX FPS statistics for foreground game/app.
- `userwait_v2` (2.2B rows, 59 GiB) — user wait incidents (>1s) with CPU/memory/disk/network context.
- `web_cat_usage_v2` (171M rows, 6.4 GiB) — browser usage by category (social, productivity, etc.).
- `machinegeo` (128M rows, 6.3 GiB) — client geographic info (country).
- `power_acdc_usage_v4` (17.7B rows, 342 GiB) — summarized data by AC/DC + display on/off events.
- `power_s_state` (547M rows, 21 GiB) — system power state changes (shutdown, system up, suspend, resume).
- `wifi_ap` (151M rows, 7.7 GiB) — WiFi access point details.
- `wifi_card` (69M rows, 7.0 GiB) — WiFi card details.
- `wifi_state` (775M rows, 15 GiB) — WiFi connection state changes.
- Largest tables: `eventlog_item_hist` (83B rows, 2.8 TB), `driver_files_hist` (77B rows, 232 GiB), `plist_process_resource_util_hist` (35B rows, 1.6 TB).

**3. public** (gaming/FPS analytics, ~80+ tables):
- Various `gaming_*` tables for FPS analysis, game config, segment embeddings.
- `ucsd_apps_execlass_*` — app classification tables.
- Sample/mapping tables for country/chassis subsets.

### DCA Data Dictionary (dca-dictionary.pdf, 144 pages)

Intel's System Usage Report (SUR) Data Dictionary. Three layers:
1. **Collector Layer**: Raw data collected by Input Libraries (ILs) on host systems into SQLite DB. 22 ILs including ACPI-BATTERY, FGWND (foreground window), FPS, HWIL (hardware MSRs/MMIOs), MOUSE-UX, OSIL (OS perf counters), PROCESS, QNR_SAMPLER (CPU/GPU metrics), WIFI, etc.
2. **ATL Layer**: Analyzer Task Library queries SQLite, generates flat files for upload. Groups metrics into categories (CPU, memory, network, WiFi, mouse cursor, application usage, OS events, system specs).
3. **ETL Layer**: Cloud processing into analysis tables. 115 tables documented with full column-level details.

Common columns across most ETL tables: `load_ts`, `batch_id`, `audit_zip`, `audit_internal_path`, `guid` (unique client identifier), `interval_start_utc`, `interval_end_utc`, `ts` (collection timestamp), `dt` (date, ETL-generated from ts).

### Tables referenced in the 22-query SQL benchmark

The SQL queries in `docs/queries/` reference a "reporting" schema with normalized/derived tables:
- `reporting.system_sysinfo_unique_normalized` — system attributes (chassistype, countryname_normalized, sysinfo_ram, persona, cpu_codename, etc.)
- `reporting.system_psys_rap_watts` — system power (avg_psys_rap_watts, nrs)
- `reporting.system_pkg_C0` — C0 residency (avg_pkg_c0, nrs)
- `reporting.system_pkg_avg_freq_mhz` — clock frequency (avg_avg_freq_mhz, nrs)
- `reporting.system_pkg_temp_centigrade` — temperature (avg_temp_centigrade, nrs)
- `reporting.system_batt_dc_events` — battery usage (num_power_ons, duration_mins)
- `reporting.system_memory_utilization` — RAM usage (avg_percentage_used, nrs, sysinfo_ram)
- `reporting.system_frgnd_apps_types` — foreground app usage by type
- `reporting.system_userwait` — user wait times
- `reporting.system_web_cat_pivot_*` — browsing by 28 categories
- `reporting.system_network_consumption` — network bytes
- `reporting.system_display_devices` — connected displays
- `reporting.system_hw_pkg_power` — processor power draw

These "reporting" tables appear to be pre-joined/aggregated views built on top of the raw `university_prod` tables, keyed on `guid` and optionally `dt`.

## Q2 deliverables and deadlines

- **Report checkpoint**: Sun Feb 15 (complete title/abstract/intro/methods + appendix with proposal)
- **Code checkpoint**: Sun Feb 15 (documented, reproducible, README with env setup)
- **Poster checkpoint**: Sun Feb 15 (draft PDF)
- **Website checkpoint**: Sun Feb 22 (skeleton/outline on GitHub Pages)
- **Presentation checkpoint**: Week 8 check-in with TA
- **Final submissions**: Sun Mar 8 (report, code, website, poster — submit poster print to Triton Print)
- **Capstone showcase**: Fri Mar 13 (poster session)

---

## Differential privacy — full technical depth

### Definition and intuition (Dwork & Roth Ch. 1-2)

DP is a **definition, not an algorithm**. It is a promise from a data curator: "You will not be affected, adversely or otherwise, by allowing your data to be used, no matter what other studies or information sources are available."

**Formal definition**: A randomized mechanism M with domain N^|X| is (epsilon, delta)-DP if for all S in Range(M) and all x, y with ||x - y||_1 <= 1: Pr[M(x) in S] <= exp(epsilon) * Pr[M(y) in S] + delta. Databases are represented as histograms x in N^|X|, where x_i counts records of type i. Distance = L1 norm.

**Key parameters**:
- epsilon controls the privacy-utility tradeoff. Small epsilon (< 1) = strong privacy. For small epsilon, exp(epsilon) ~ 1 + epsilon, so outputs are nearly indistinguishable.
- delta is additive slack. Must be negligible (less than inverse polynomial in database size). delta = 1/||x||_1 is dangerous (permits publishing a few complete records).

**Privacy loss random variable**: L(xi) = ln(Pr[M(x)=xi] / Pr[M(y)=xi]). For (epsilon, delta)-DP, |L| <= epsilon with probability >= 1 - delta.

**Distinction between (epsilon, 0) and (epsilon, delta)**:
- (epsilon, 0)-DP: For every run, the output is (almost) equally likely on every neighboring database *simultaneously*.
- (epsilon, delta)-DP: For any *fixed pair* of neighboring databases, the privacy loss is bounded with high probability 1-delta. But given an output, there *may* exist some database where that output is much more likely.

### Key properties

1. **Post-processing immunity** (Proposition 2.1): If M is (epsilon, delta)-DP, then f(M) is also (epsilon, delta)-DP for any data-independent function f. Once released, no analysis can make the output less private.
2. **Composition** (Theorem 3.16): k mechanisms with parameters (epsilon_i, delta_i) compose to (sum(epsilon_i), sum(delta_i))-DP. "Epsilons and deltas add up."
3. **Advanced composition** (Theorem 3.20): k applications of (epsilon, delta)-DP mechanisms yield (epsilon', k*delta + delta')-DP where epsilon' = sqrt(2k * ln(1/delta')) * epsilon + k*epsilon*(exp(epsilon)-1). Saves sqrt(log(1/delta)) factor over naive composition.
4. **Group privacy** (Theorem 2.2): For groups of size k, (epsilon, 0)-DP implies (k*epsilon, 0)-DP. Privacy degrades linearly with group size.

### Why privacy requires randomness

Any non-trivial deterministic mechanism can always be distinguished on some pair of neighboring databases (hybrid argument). Thus randomness is essential for privacy guarantees that hold against arbitrary auxiliary information.

### Fundamental Law of Information Recovery

Overly accurate answers to too many questions will destroy privacy. This applies to ALL privacy methods, not just DP. Reconstruction attacks (Section 8.1) show that if noise magnitude is bounded by E, an adversary can reconstruct the database to within 4E positions by querying all subsets.

### Basic mechanisms (Dwork & Roth Ch. 3)

**Laplace mechanism**: For f: N^|X| -> R^k with L1-sensitivity Delta_f = max_{x~y} ||f(x)-f(y)||_1, add noise Y_i ~ Lap(Delta_f / epsilon) to each coordinate. Achieves (epsilon, 0)-DP. Error: O(k * Delta_f / epsilon) for k queries.

**Gaussian mechanism**: Add noise N(0, sigma^2) where sigma >= c * Delta_2_f / epsilon for c^2 > 2ln(1.25/delta). Achieves (epsilon, delta)-DP. Advantage: same noise type as statistical error; sum of Gaussians is Gaussian.

**Exponential mechanism**: For utility function u: N^|X| x R -> R with sensitivity Delta_u, select r in R with probability proportional to exp(epsilon * u(x,r) / (2*Delta_u)). Achieves (epsilon, 0)-DP. Utility guarantee: output has score >= OPT - O((Delta_u/epsilon) * ln|R|) with high probability.

**Report Noisy Max**: Add Lap(1/epsilon) to each of m counting queries, report the index of the maximum. Is (epsilon, 0)-DP regardless of m (information minimization principle — only releasing the index, not all counts).

**Randomized response**: Flip coin; if tails respond truthfully, if heads flip again and respond randomly. Is (ln 3, 0)-DP. Earliest example of privacy by randomized process.

### Sensitivity (L1 vs L2)

- **L1-sensitivity**: Delta_f = max_{x~y} ||f(x)-f(y)||_1. Used with Laplace mechanism.
- **L2-sensitivity**: Delta_2_f = max_{x~y} ||f(x)-f(y)||_2. Used with Gaussian mechanism. For histograms/counting queries, L1-sensitivity = 1 (one person changes at most one count). A fixed set of m counting queries has worst-case L1-sensitivity m.

### Sparse Vector Technique (Dwork & Roth Sec. 3.6)

For a stream of sensitivity-1 queries, only a few are "above threshold." AboveThreshold adds noise to the threshold and to each query value, reporting only above/below. Privacy cost scales with the number of above-threshold queries c, not the total number of queries. Accuracy: O(c * log(k) / epsilon) where k = total queries.

NumericSparse extends this to also release the numeric values of above-threshold queries at a constant factor loss in accuracy.

### Query release with correlated error (Dwork & Roth Ch. 4-5)

**SmallDB** (offline): Uses exponential mechanism to select a small synthetic database (size O(log|Q| / alpha^2)) that approximates all queries in Q to within alpha. For normalized queries, error scales as O((log|X| * log|Q| / (epsilon * ||x||_1))^{1/3}). Can answer exponentially many queries in the database size with (epsilon, 0)-DP.

**Private Multiplicative Weights** (online): Maintains a hypothesis database. When a query reveals large error (above threshold), uses the noisy answer to update the hypothesis via multiplicative weights. Only "hard" queries cost privacy. Convergence: at most 4*log|X|/alpha^2 update steps. Combined with NumericSparse, achieves same accuracy as SmallDB but online.

**Key insight**: By rethinking the computational goal (e.g., correlated noise, answering in batches), one can do far better than methodically replacing each step with independent noise.

### Iterative Construction (IC) Mechanism (Dwork & Roth Ch. 5)

Generalization: reduces query release to the simpler problem of "distinguishing" (finding the query with largest discrepancy). Uses any T(alpha)-database update algorithm + any private distinguisher. The exponential mechanism serves as a canonical distinguisher. The reduction shows: up to small factors, the information complexity of query release = information complexity of agnostic learning.

**Median Mechanism**: Maintains a set of databases (an alpha-net). Each update removes at least half. Converges in log|N_alpha(Q)| steps. Works for arbitrary low-sensitivity queries, not just linear.

**Game-theoretic view**: Query release = computing approximate min-max strategies in a zero-sum game between a "database player" and a "query player." Multiplicative weights = Alice playing no-regret learning. Boosting for queries = Bob playing no-regret learning.

### Lower bounds (Dwork & Roth Ch. 8)

- **Reconstruction attacks**: With noise always bounded by E, adversary reconstructs database to within 4E positions (Theorem 8.1). With linear number of queries and noise o(sqrt(n)), adversary reconstructs almost the entire database (Theorem 8.2).
- **Packing lower bound** for (epsilon, 0)-DP (Theorem 8.8): For k sensitivity-1 queries, noise must be Omega(min(k/epsilon, d/epsilon)) where d = |X|. This is tight (matching Laplace mechanism).
- **Separation (epsilon,0) vs (epsilon,delta)**: The packing bound is linear in k for (epsilon,0) but sqrt(k) for (epsilon,delta) via advanced composition. Proven separation for k ~ n queries.

### Computational complexity (Dwork & Roth Ch. 9)

Under cryptographic assumptions (one-way functions):
- **Hard to syntheticize**: Some distributions of databases + queries are hard to produce synthetic databases for, even though inefficient algorithms (SmallDB) succeed. The construction uses digital signatures: rows of the synthetic DB must be valid signatures, which can't be forged.
- **Traitor tracing connection**: Hardness of general synopsis generation (not just synthetic databases) is tightly connected to traitor tracing schemes. If efficient query release existed, it would break traitor tracing.
- **Computational DP**: With computationally bounded adversaries, secure multiparty computation can simulate a trusted curator. This provably buys accuracy in the distributed setting: summing n bits requires Omega(sqrt(n)) error in the multi-party model vs O(1) with a trusted curator.

### DP and Machine Learning (Dwork & Roth Ch. 11)

- **PAC learning under DP**: Any class learnable non-privately is also privately learnable with the same sample complexity (up to log|C|/(epsilon*alpha) term). Via exponential mechanism with quality score = empirical error.
- **SQ (statistical query) model**: Any SQ-learnable class can be learned privately via Laplace mechanism on the queries. Preserves computational efficiency.
- **Online learning**: Randomized Weighted Majority (multiplicative weights) with update parameter eta = epsilon/(sqrt(32T*ln(1/delta))) is automatically (epsilon, delta)-DP. Privacy is "free" — regret bound O(sqrt(ln(1/delta)*ln(k)) / (epsilon*sqrt(T))), nearly matching non-private bound.
- **Empirical risk minimization**: Reduce to learning from expert advice. LinearLearner achieves error O(sqrt(ln(1/delta)*ln(d/beta)) / (epsilon*sqrt(T))).

### Additional models (Dwork & Roth Ch. 12)

- **Local model**: Each individual randomizes their own data (generalized randomized response). Equivalent to SQ model up to polynomial factors. Strictly weaker than centralized model (single counting query needs Theta(sqrt(n)) error vs O(1) centrally).
- **Pan-private streaming**: Internal state must also be DP (protects against intrusions/subpoenas). Event-level and user-level privacy. Counter algorithm achieves O(log^{2.5}(T)/epsilon) error, and Omega(log T) is a lower bound.
- **Continual observation**: Binary tree structure for counting — noise scales with O(log^{1.5} T) not O(T).

### DP and Mechanism Design (Dwork & Roth Ch. 10)

- DP implies 2*epsilon-approximate dominant strategy truthfulness (Proposition 10.1). Composes well, provides group privacy (collusion resistance), works without money.
- Digital goods auctions: exponential mechanism achieves OPT - O(log(n)/epsilon) revenue.
- Equilibrium selection in large games: DP mechanisms can compute approximate correlated/Nash equilibria, yielding O(1/n^{1/4})-approximately truthful mechanisms.
- Punishing exponential mechanism: combine exponential mechanism (good utility) with commitment mechanism (strict truthfulness) to get exactly truthful mechanisms without money.
- Privacy-aware agents: modeling cost of privacy as c_i(epsilon) = epsilon * v_i. Sensitive surveyor's problem. In the "sensitive value model" (where value for privacy correlates with the private data), no individually rational direct revelation mechanism achieves nontrivial accuracy.

---

## DP-SGD — complete technical details (Abadi et al., 2016)

### Algorithm (Algorithm 1 in the paper)

At each step t:
1. **Sample a lot** L_t with sampling probability q = L/N (each example independently with probability q).
2. **Compute per-example gradients**: g_t(x_i) = nabla_{theta_t} L(theta_t, x_i) for each x_i in L_t.
3. **Clip gradients**: g_bar_t(x_i) = g_t(x_i) / max(1, ||g_t(x_i)||_2 / C). Bounds L2 norm to C.
4. **Aggregate with noise**: g_tilde_t = (1/L)(sum g_bar_t(x_i) + N(0, sigma^2 * C^2 * I)).
5. **Descent**: theta_{t+1} = theta_t - eta_t * g_tilde_t.

### Moments accountant

The key innovation. Tracks the log moment generating function alpha_M(lambda) = log E[exp(lambda * c(o; M, d, d'))] where c is the privacy loss.

**Properties** (Theorem 2):
- *Composability*: For sequential mechanisms M_1,...,M_k: alpha_M(lambda) <= sum alpha_{M_i}(lambda).
- *Tail bound*: delta = min_lambda exp(alpha_M(lambda) - lambda * epsilon).

For the Gaussian mechanism with random sampling, the moment bound is: alpha(lambda) <= q^2 * lambda * (lambda+1) / ((1-q) * sigma^2) + O(q^3 * lambda^3 / sigma^3).

**Main result** (Theorem 1): Algorithm is (epsilon, delta)-DP if sigma >= c_2 * q * sqrt(T * log(1/delta)) / epsilon. Saves sqrt(log(T/delta)) factor over strong composition theorem. Practical example: q=0.01, sigma=4, delta=10^-5, T=10000 gives epsilon=1.26 (vs 9.34 with strong composition).

### Experimental findings

- MNIST: 97% accuracy at (8, 10^-5)-DP; 95% at (2, 10^-5); 90% at (0.5, 10^-5).
- CIFAR-10: 73% at (8, 10^-5)-DP (vs 86% non-private baseline, ~13% gap). Used pre-trained convolutional layers from CIFAR-100 (treated as public).
- **Lot size**: Optimal ~ sqrt(N). Too small = too many epochs consume budget; too large = more per-step privacy cost.
- **Gradient clip norm C**: Median of unclipped gradient norms is a good heuristic.
- **Network size**: Counterintuitively, more hidden units don't hurt accuracy — larger networks may be more noise-tolerant.
- **Generalization benefit**: DP-SGD training shows small train-test gap (consistent with theoretical result that DP training generalizes well).

### Implementation notes

- Per-example gradient computation required (not just batch gradient). TensorFlow `per_example_gradient` operator.
- Sanitizer (clip + noise) + PrivacyAccountant (tracks spending).
- DP-PCA: Project inputs onto principal directions of noisy covariance matrix A^T A + Gaussian noise. Reduces dimensionality (784 -> 60 for MNIST), improves accuracy AND training speed.

---

## Private Evolution (PE) — complete technical details (Lin et al., 2024)

### Problem formulation: DPSDA

**DP Wasserstein Approximation (DPWA)**: Given private dataset S_priv, design an (epsilon, delta)-DP algorithm that outputs synthetic dataset S_syn minimizing Wasserstein distance W_p(S_priv, S_syn).

**DPSDA**: Solve DPWA with only black-box API access to foundation models. API queries must also be DP (API provider is untrusted).

### Algorithm

Two APIs required:
- **RANDOM_API(n)**: Generate n random samples from the foundation model.
- **VARIATION_API(S)**: Generate variations of each sample in S (similar images/texts).

**PE loop** (Algorithm 1):
1. S_0 = RANDOM_API(N_syn) — initial population.
2. For t = 1..T:
   a. **DP Nearest Neighbor Histogram** (Algorithm 2): Each private sample x votes for its nearest synthetic candidate in embedding space. Add Gaussian noise N(0, sigma * I_n) to histogram. Threshold: max(histogram - H, 0).
   b. Normalize histogram to distribution P_t.
   c. Resample N_syn candidates from P_t (with replacement).
   d. S_t = VARIATION_API(resampled candidates).
3. Return S_T.

### Distance function

d(x, z) = ||Phi(x) - Phi(z)||_2 where Phi is an embedding network (e.g., Inception, CLIP). **Lookahead**: Average embedding of k variations of z instead of z itself, to anticipate the effect of VARIATION_API.

### Privacy analysis

Remarkably simple (no subsampling, no complex composition):
1. Sensitivity of histogram = 1 (each private sample contributes exactly one vote).
2. Each iteration = Gaussian mechanism with noise multiplier sigma.
3. T iterations = adaptive composition of T Gaussian mechanisms.
4. Equivalent to single Gaussian mechanism with noise multiplier sigma/sqrt(T) (Dong et al., 2022 Corollary 3.3).
5. Compute (epsilon, delta) from standard Gaussian mechanism formula (Balle & Wang, 2018).

Releasing all intermediate sets S_1,...,S_T satisfies the same DP (protects from API provider too).

### Convergence theory

**Non-private** (Theorem 1): Converges in T >> d * log(D/eta) / log(L) iterations, where d = embedding dimension, D = diameter, eta = target Wasserstein error, L ~ number of variations per point. Nearly tight: need at least Omega(d/log L) iterations.

**Private** (Theorem 2): With multiplicity B (B identical copies of each private point), converges when B >> H >> sigma * log(T*L*N_priv), where sigma >> sqrt(T) * log(1/delta) / epsilon. PE discovers every cluster of size >> d * log(1/delta) / epsilon in O(d) iterations, comparable to SOTA DP clustering.

### Key experimental results

**CIFAR-10** (ImageNet public data): FID <= 7.9 at epsilon = 0.67 (prior SOTA DP-Diffusion needed epsilon = 32). Downstream classification: 84.8% with ensemble of 5 classifiers.

**Camelyon17** (medical, large distribution shift from ImageNet): 80.33% accuracy at (10, 3e-6)-DP (prior SOTA 91.1%). PE works even under large distribution shifts because foundation model support spans the entire sample space.

**High-resolution** (512x512 cat images, N=100): Used Stable Diffusion. PE captures key characteristics despite small dataset and high resolution — a regime where no prior training-based method had results.

**Unlimited samples**: Call VARIATION_API on S_syn repeatedly. Classifier accuracy improves with more samples (89.13% at 1M samples with ensemble).

---

## Aug-PE for Text (Xie et al., 2024)

### Challenges of extending PE to text

- Text is discrete (vs continuous pixel space) — harder to control variation diversity.
- Variable length (vs fixed image dimensions).
- Original PE yields unsatisfactory text quality.

### Design innovations

**RANDOM_API**: Direct prompting with class labels. Pseudo-class approach: generate subcategories per class (e.g., Steakhouse, Bistros for restaurants) to encourage diversity.

**VARIATION_API** (two methods):
- *Paraphrasing*: "Please rephrase the below sentences: {input}". Works well for GPT-2.
- *Fill-in-the-blanks*: Mask p% of tokens, ask LLM to fill. Works well for instruction-tuned models (GPT-3.5). Higher mask probability = more diversity.

**Adaptive text lengths**: Add "with {targeted_word} words" to prompt. Adjust targeted_word = max(original_word + N(0, sigma_word^2), min_word). Captures fat-tailed length distributions.

**Sample selection improvements over PE**:
- Rank-based selection (top N_syn by probability) instead of random sampling — eliminates redundancy.
- Generate L-1 variations per selected sample — larger, more diverse candidate pool.
- Retain selected samples in next iteration's dataset — preserves high-quality candidates.
- Use self-embedding (K=0) for NN voting rather than averaged variation embedding (K>0) — more representative for text.

### Results

- Under same generator (GPT-2 series): Aug-PE achieves comparable or better accuracy than DP-finetuning baselines on Yelp and OpenReview.
- With GPT-3.5 (where DP-finetuning is infeasible): significantly outperforms on challenging datasets (OpenReview, PubMed). E.g., OpenReview area classification: 41.9% (Aug-PE GPT-3.5) vs 38.6% (DP-FT GPT-2-Large) at epsilon=1.
- **Efficiency**: 12.7x-65.7x speedup over DP-finetuning. DP-SGD finetuning GPT-2-Large on Yelp = 1764 GPU hours; Aug-PE L=2 = 27 hours.
- **Membership inference attacks**: Aug-PE exhibits lower AUC scores than DP-finetuning baselines, indicating stronger empirical privacy.
- **Scaling**: Performance improves with more powerful LLMs (GPT-3.5, Mixtral-8x7B, LLaMA-2). Also improves with more synthetic samples.

---

## PE for Tabular Data (Swanberg et al., 2025)

### Key innovation: Workload-aware distance function

No good tabular embeddings exist (unlike images/text). Instead, define distance based on the query workload:

Wdist_psi(x, c) = sum_i |psi_i(x) - psi_i(c)|

where psi_i are predicates corresponding to the workload queries W. Low workload-aware distance => low workload error.

### Two approaches evaluated

**1. Adapted Private Evolution**: PE with workload-aware distance on NYC Taxi data using Gemini 1.0 Pro.
- Without DP, it converges and 1-way marginals match.
- **With DP, PE fails to beat even simple baselines** (independent marginals, DP workload). Optimal setting: only 1 iteration (marginal gains from iterating are outweighed by privacy cost of composition).

**2. One-shot LLM-generated public data**: Prompt Gemini to generate records matching column names/datatypes (one shot, no privacy cost). Use as public data in existing algorithms:
- PMW_pub: Uses Gemini data to initialize the generating distribution.
- MST modified: Uses Gemini data in the generation step instead of Private-PGM.
- JAM: Privately decides whether to measure each marginal on public or private data.

### Key findings

- **JAM with Gemini data performs best overall, BUT equally well with uniform random data** — the Gemini data doesn't actually help. JAM is simply using private data for the queries rather than the public data.
- **Tabular DP synthesis is much more mature** than image/text. Marginal-based methods (MST, AIM, JAM) are hard to beat.
- **LLMs capture 1-way marginals** reasonably but are **inaccurate on k-way marginals** — the cross-column correlations that matter for workload queries.
- **Takeaway**: API access to LLMs does not (yet) improve DP synthetic tabular data beyond established baselines. May change as foundation models improve and are trained on more tabular data.

---

## VAE (Variational Autoencoder) — as used in Q1

**Architecture** (from q1-report.tex): Encoder maps 13D input -> 128-unit hidden -> 16D mean + 16D log-variance. Decoder mirrors. Produces logits for categorical attributes + numeric head for timestamp.

**Loss**: L = L_rec + L_kl. L_rec = CE(x_cat, x_hat_cat) + MSE(x_time, x_hat_time). L_kl = KL(q_theta(z|x) || N(0,I)).

**Generation**: Sample z ~ N(0,I), decode to logits, argmax for categories, invert timestamp scaler, clip to observed range.

**With DP-SGD**: Per-sample gradients clipped to norm C=1.0, Gaussian noise with sigma=1.0*C added at each step. Privacy tracked via Opacus privacy accountant.

---

## Key references

- **Dwork & Roth (2014)**: "The Algorithmic Foundations of Differential Privacy" — foundational DP textbook. Covers Laplace/Gaussian/exponential mechanisms, composition theorems (basic + advanced), sparse vector technique, query release (SmallDB, multiplicative weights, boosting), lower bounds (reconstruction attacks, packing arguments), computational complexity (hard-to-syntheticize distributions, traitor tracing), mechanism design, machine learning, local/pan-private/continual observation models. Notes in `docs/papers/dwork-roth-notes-incomplete.md`.
- **Abadi et al. (2016)**: "Deep Learning with Differential Privacy" — introduces DP-SGD (per-sample gradient clipping + Gaussian noise) and the moments accountant (tracks log MGF of privacy loss, composes linearly, yields tighter bounds than strong composition by sqrt(log(T/delta)) factor). MNIST 97% at (8,10^-5)-DP, CIFAR-10 73% at (8,10^-5)-DP.
- **Lin et al. (2024)**: "DP Synthetic Data via Foundation Model APIs 1: Images" — introduces Private Evolution. Training-free, API-only. DP NN histogram with Gaussian noise. CIFAR-10 FID 7.9 at epsilon=0.67 (48x improvement over DP-Diffusion). Convergence proof: O(d) iterations, comparable to SOTA DP clustering.
- **Xie et al. (2024)**: "DP Synthetic Data via Foundation Model APIs 2: Text" — Aug-PE. Innovations: fill-in-the-blanks variation, adaptive text lengths, rank-based selection, multi-variation generation. Competitive with DP-finetuning at same model; superior with GPT-3.5. 12-66x faster than DP-SGD finetuning.
- **Swanberg et al. (2025)**: "Is API Access to LLMs Useful for Generating Private Synthetic Tabular Data?" — PE adapted for tabular with workload-aware distance. One-shot Gemini data as public data for PMW_pub/JAM. **Negative result**: API access doesn't beat established marginal-based baselines (MST, JAM). Tabular DP synthesis is more mature.
- **Ghalebikesabi et al. (2023)**: DP-Diffusion. Fine-tunes diffusion models with DP-SGD. SOTA before PE. Epsilon=32 for CIFAR-10 FID~7.9.

## DSC180B-Q2 submodule (teammates' repo)

**WARNING**: Code quality is low and AI-generated. Do NOT trust or learn from it. Treat as a status report only.

### What they did

- **5 tables downloaded** from Globus (not all 19 needed): `os_network_consumption_v2`, `frgnd_system_usage_by_app`, `userwait_v2`, `system_sysinfo_unique_normalized`, and `ucsd_apps_execlass_final`.
- **7 of 24 queries implemented** as standalone Python/DuckDB scripts (queries 2-5, 10-12). All query against raw Parquet files directly.
- **Preprocessing pipeline for query 2 only**: build training table -> freeze clip bounds -> apply mappings/buckets -> DP-SGD training. Hardcoded absolute paths (`/Users/hanatjendrawasi/...`).
- **DP-SGD training attempted for query 2 only**: Trained on a bucketed/mapped version of network consumption data (37,044 rows, 4 categorical + 5 numeric columns).
- **DP-VAE code exists** but unclear if it was actually run on the DCA data.
- **No Private Evolution implementation** at all.

### What they did NOT do

- Did not build or even identify the "reporting" schema tables that the 22 queries actually reference. The queries use `reporting.system_*` tables which are **pre-joined/aggregated views** that don't exist in the raw `university_prod` data.
- Did not download or address 14 of the 19 required reporting tables.
- Did not implement 17 of 24 queries.
- No evaluation framework (no comparison of real vs synthetic query results).
- No reproducibility (hardcoded local paths, no environment setup, no README with setup instructions beyond a vague overview).

---

## The "reporting" schema gap — critical issue

The 24 SQL queries in `docs/queries/` reference 19 distinct `reporting.system_*` tables. These are NOT raw `university_prod` tables. They appear to be **pre-aggregated views** that someone at Intel built. The raw tables and reporting tables have different schemas:

**Example**: Query 1 references `reporting.system_psys_rap_watts` with columns `nrs`, `avg_psys_rap_watts`. The raw table `hw_pack_run_avg_pwr` has very different columns (histogram bins, event names, etc.). Someone aggregated the raw data into per-guid summary statistics.

**The 19 reporting tables needed** (mapped to likely raw source tables where identifiable):

| Reporting table | Likely raw source | Status |
|---|---|---|
| system_sysinfo_unique_normalized | system_sysinfo_unique_normalized | Direct (same table, 1M rows) |
| system_psys_rap_watts | hw_pack_run_avg_pwr / power_acdc_usage_v4 | Must build aggregation |
| system_pkg_C0 | os_c_state / power_acdc_usage_v4 | Must build aggregation |
| system_pkg_avg_freq_mhz | hw_cpu_frequency / os_p_state | Must build aggregation |
| system_pkg_temp_centigrade | hw_metric_stats | Must build aggregation |
| system_batt_dc_events | batt_acdc_events | Must build aggregation |
| system_memory_utilization | os_memsam_avail_percent | Must build aggregation |
| system_frgnd_apps_types | frgnd_system_usage_by_app | Must build aggregation |
| system_userwait | userwait_v2 | Must build aggregation |
| system_web_cat_usage | web_cat_usage_v2 | Must build aggregation |
| system_web_cat_pivot_duration | web_cat_usage_v2 (pivoted) | Must build pivot |
| system_network_consumption | os_network_consumption / v2 | Must build aggregation |
| system_display_devices | ??? (display_devices in ETL) | Need to find raw table |
| system_hw_pkg_power | hw_metric_stats / hw_pack_run_avg_pwr | Must build aggregation |
| system_mods_power_consumption | ??? (plist_process_resource_util?) | Need to find raw table |
| system_mods_top_blocker_hist | ??? (mods_sleepstudy tables) | Need to find raw table |
| system_os_codename_history | machinemaster_json? | Need to find raw table |
| system_cpu_metadata | system_sysinfo_unique_normalized? | May be derivable from sysinfo |
| system_on_off_suspend_time_day | power_s_state / power_idle_state | Must build aggregation |

### Teammates' suggested download (DO NOT FOLLOW)

They suggested downloading these 5 tables:
- `/projects/dca/university_prod/os_network_consumption_v2/` (13 GiB)
- `/projects/dca/university_prod/frgnd_system_usage_by_app/` (338 GiB)
- `/projects/dca/university_prod/userwait_v2/` (59 GiB)
- `/projects/dca/public/ucsd_apps_execlass_final/` (19 MiB)
- `/projects/dca/university_analysis_pad/system_sysinfo_unique_normalized/` (74 MiB)

**Total: ~410 GiB. This is wrong.** They picked these because it's what they used in DSC180B-Q2 (queries 2-5, 10-12). Problems:
- `frgnd_system_usage_by_app` is 338 GiB — absurd for a local machine.
- `userwait_v2` is 59 GiB — large and only needed for 3 queries.
- `ucsd_apps_execlass_final` is from the `public` schema and is NOT referenced by any of the 24 benchmark queries.
- These 5 tables only cover ~8 of 24 queries, and all from just 2 query types (network/app usage).

### Actual data strategy (10 queries, ~5-10 GiB download)

**Key insight: You do NOT need to download entire tables.** Each table is split into multiple parquet files (e.g., 8 files of ~1 GiB each). You can download just the first 1-2 parquet files per table to get a representative sample. Then filter all tables to only the guids present in your sysinfo sample for consistency.

**Selected 10 queries** covering all 5 proposal query types from only 6 raw tables:

| Query Type | Query File | Reporting Tables Used |
|---|---|---|
| Aggregate stats + joins | `avg_platform_power_c0_freq_temp_by_chassis` | sysinfo + psys_rap + C0 + freq + temp (5-way join) |
| Aggregate stats + joins | `server_exploration_1` | network + sysinfo |
| Ranked top-k | `most_popular_browser_in_each_country_by_system_count` | web_cat_usage + sysinfo |
| Geographic/demographic | `Xeon_network_consumption` | network + sysinfo |
| Geographic/demographic | `pkg_power_by_country` | hw_pkg_power + sysinfo |
| Histogram/distribution | `ram_utilization_histogram` | memory_utilization |
| Histogram/distribution | `popular_browsers_by_count_usage_percentage` | web_cat_usage |
| Complex pivot | `persona_web_cat_usage_analysis` | web_cat_pivot_duration + sysinfo |
| Geographic (stretch) | `battery_power_on_geographic_summary` | batt_dc_events + sysinfo |
| Demographic (stretch) | `battery_on_duration_cpu_family_gen` | cpu_metadata + batt_dc_events |

**Download plan — just first 1-2 parquet files per table:**

| Globus Path | Full Size | Download | Files to grab |
|---|---|---|---|
| `university_analysis_pad/system_sysinfo_unique_normalized/` | 74 MiB | **All 8 files** | Anchor table, need full set |
| `university_analysis_pad/data_dictionary/` | 40 KiB | **All** | Schema reference |
| `university_prod/web_cat_pivot/` | 53 MiB | **All** | Tiny, just grab it |
| `university_prod/web_cat_usage_v2/` | 6.4 GiB | **1 file (~1-2 GiB)** | `0000_part_00.parquet` |
| `university_prod/hw_metric_stats/` | 8.7 GiB | **1 file (~1-2 GiB)** | `0000_part_00.parquet` |
| `university_prod/os_network_consumption_v2/` | 13 GiB | **1 file (~1-2 GiB)** | `0000_part_00.parquet` |
| `university_prod/os_memsam_avail_percent/` | 15 GiB | **1 file (~1-2 GiB)** | `0000_part_00.parquet` |
| `university_prod/batt_acdc_events/` (stretch) | 34 GiB | **1 file (~1-2 GiB)** | `0000_part_00.parquet` |

**Also download manifests** (a few KB each) for all target tables — they contain column schemas:
- `{table_name}-manifest.json` for each table above

**Estimated total download: ~5-10 GiB** (vs 43 GiB full or 410 GiB teammates' plan).

**After downloading**, filter all event tables to only guids present in `system_sysinfo_unique_normalized` using DuckDB. This ensures a coherent sample.

### Processing pipeline

```
data/raw/{table_name}/*.parquet     <-- partial Parquet from Globus (gitignored)
        |
        v  (DuckDB aggregation SQL scripts in src/)
data/reporting/{table_name}.parquet <-- per-guid aggregated reporting tables (gitignored)
        |
        v  (benchmark SQL queries from docs/queries/)
data/results/real/{query_name}.csv  <-- ground truth query results
data/results/synth/{query_name}.csv <-- synthetic data query results
```

**Step 1**: Download manifests + first parquet files via Globus Connect Personal  
**Step 2**: Inspect schemas with DuckDB (verify hw_metric_stats has PSYS_RAP, PKG_C0, AVG_FREQ, TEMP metrics)  
**Step 3**: Build reporting tables — DuckDB SQL scripts that aggregate raw per-event data into per-guid summaries matching the `reporting.system_*` schema  
**Step 4**: Run 10 benchmark queries on reporting tables -> ground truth CSVs  
**Step 5**: Apply DP-SGD and Private Evolution to reporting tables -> synthetic versions  
**Step 6**: Run same queries on synthetic data -> compare with ground truth  

### December 2024 update — pre-built reporting tables (CRITICAL DISCOVERY)

The folder `/projects/dca/university_prod/dca_update_dec_2024/` (added Jan 16, 2025) contains **pre-aggregated derived tables** that Intel built — the exact "reporting" schema tables the 24 queries need. This was added AFTER the teammates did their Q1 work, so they never saw it.

**Contents of `dca_update_dec_2024/`:**

| Folder | Likely reporting table | Compressed size | Verdict |
|---|---|---|---|
| `system_cpu_metadata` | system_cpu_metadata | **42 MiB** | DOWNLOAD — tiny, unlocks 2 queries |
| `system_os_codename_history` | system_os_codename_history | **18 MiB** | DOWNLOAD — tiny, unlocks 2 blocker queries |
| `guids_on_off_suspend_time_day` | system_on_off_suspend_time_day | **17 MiB** | DOWNLOAD — tiny, unlocks on/off/sleep query |
| `reporting_soc_cpu_power_usage_metrics_filter` | psys_rap / C0 / freq / temp | **987 B** | DOWNLOAD — basically empty? Check if stub |
| `reporting_soc_cpu_power_histo_metrics_filter` | Power histogram metrics | unknown | Check size |
| `mods_sleepstudy_top_blocker_hist` | system_mods_top_blocker_hist | **1.88 GiB** | MAYBE — moderate, unlocks 2 blocker queries |
| `display_devices` | system_display_devices | **6.15 GiB** | SKIP for now — large, only 2 display queries |
| `apps_execlass_combined` | App classification labels | **13.3 GiB** (2 files) | SKIP — large, not directly in benchmark queries |
| `__tmp_fgnd_apps_date` | system_frgnd_apps_types | **21.6 GiB** (4 files) | SKIP — still huge even pre-aggregated |
| `frgnd_v2_daily_summary` | system_frgnd_apps_types (daily) | unknown | Check size — may be smaller alternative |
| `__tmp_batt_dc_events` | system_batt_dc_events | unknown | Check size |
| `__tmp_soc_cpu_power_sysinfo` | CPU power + sysinfo join | unknown | Check size |
| `mods_sleepstudy_power_estimation_data_13wks` | system_mods_power_consumption | unknown | Check size — needed for 3 power queries |
| `mods_sleepstudy_recent_usage_instance` | Sleep study usage | unknown | Low priority |
| `mods_sleepstudy_scenario_instance_13wks` | Sleep study scenarios | unknown | Low priority |
| `_tmp_guids_versions_timestamps` | GUID versioning metadata | unknown | Low priority |
| `reporting_athena_editions` | Athena edition filtering | unknown | Low priority |
| `reporting_sysinfo_athena` | Sysinfo for Athena devices | unknown | Low priority |
| `system_os_version_current` | Current OS version | unknown | Low priority |

**Format warning:** Unlike `university_prod` (Parquet), the update files are **gzipped text** (`.txt000.gz`, `.txt001.gz`, etc.), NOT Parquet. Some are still large — e.g., `__tmp_fgnd_apps_date` is 4 files totaling ~21.6 GiB compressed. Need to check sizes of each table before downloading blindly.

**HTTPS endpoint exists** but requires Globus auth: `https://g-3d1295.0ed28.75bc.data.globus.org/projects/dca/university_prod/dca_update_dec_2024/{table}/{file}?download=1`. Cannot be accessed programmatically without an authenticated session.

**If the smaller tables are confirmed small, this changes everything.** These pre-aggregated tables could be a fraction of the raw event tables. If confirmed:
- **All 24 queries may be feasible**, not just 10
- **Total download could be under 5 GiB** for the entire update folder
- **No need to build the reporting schema ourselves** — Intel already did it
- The original plan's tables (hw_metric_stats, os_network_consumption_v2, etc.) are still useful as fallbacks but may not be the primary path

**Recommended revised strategy:**
1. Download ALL of `dca_update_dec_2024/` — likely small since these are aggregated
2. Download `university_analysis_pad/system_sysinfo_unique_normalized/` (74 MiB) as anchor
3. Download `university_analysis_pad/data_dictionary/` (40 KiB) for schema reference
4. Download manifests for the update tables to check sizes before full transfer
5. Inspect schemas with DuckDB to confirm column names match what the 24 queries expect
6. Only fall back to raw `university_prod` tables if the update tables don't have what we need

### Intel's actual ETL SQL (docs/queries/scratch reporting analytics queries.sql)

**CRITICAL DISCOVERY:** This 1,797-line SQL file contains Intel's actual CREATE TABLE + INSERT statements for building every `reporting.system_*` table from raw `university_prod` data. This is the definitive source for how to construct reporting tables.

**Key revelation — source tables differ from what we assumed:**
- `system_hw_pkg_power` comes from `hw_pack_run_avg_pwr` (NOT `hw_metric_stats`). Uses `rap_22` as max. We don't have `hw_pack_run_avg_pwr`, but `hw_metric_stats` filtered to `IA_POWER` metrics has the needed `nrs` and `mean` columns as a substitute.
- `system_pkg_C0`, `system_pkg_avg_freq_mhz`, `system_pkg_temp_centigrade`, `system_psys_rap_watts` ALL come from `power_acdc_usage_v4_hist` (NOT `hw_metric_stats`). They aggregate by `(guid, dt, event_name)` with weighted stats from `nrs`, `avg_val`, `min_val`, `max_val`, `percentile_50th`. Our `hw_metric_stats` has `name`, `nrs`, `mean`, `min`, `max` — similar but without `event_name` breakdown or median. Substitute works because the benchmark queries aggregate across all events anyway.
- `system_display_devices` is a direct copy from `university_prod.display_devices` (no aggregation needed).
- `system_userwait` ETL confirmed: `acdc = UPPER(substring(ac_dc_event_name,1,2))` — "AC_DISPLAY_ON"→"AC", "DC_DISPLAY_OFF"→"DC", "UN_DISPLAY_ON"→"UN". Table keeps BOTH `ac_dc_event_name` and derived `acdc`.
- `system_network_consumption` ETL: `GROUP BY guid, dt, input_description` with `SUM(nr_samples) → nrs`, weighted avg `SUM(nr_samples * avg_bytes_sec)/SUM(nr_samples) → avg_bytes_sec`. Column rename: `input_description` → `input_desc`.

### Reporting Schema table definition (docs/queries/Reporting Schema table definition.md)

Official Intel documentation defining the exact schema for every `reporting.system_*` table. Key findings for building reporting tables:

**Column name mismatches between query SQL and reporting schema:**
- `system_userwait`: schema says `total_duration_in_ms` and `ac_dc_event_name`; queries use `total_duration_ms` and `acdc`. When building, alias to match what queries expect.
- `system_network_consumption`: schema says `input_desc` and `nrs`; raw data has `input_description` and `nr_samples`. Rename during build.
- `system_web_cat_pivot_duration`: schema uses SHORT column names (`education`, `finance`, `mail`, `news`, `private`, `reference`, `search`, `shopping`, `unclassified`, `recreation_travel`). Raw `web_cat_pivot` uses LONG names (`education_education`, `finance_banking_and_accounting`, etc.). Must alias.

**Event-name breakdown in hw_metric tables:**
- `system_psys_rap_watts`, `system_pkg_C0`, `system_pkg_avg_freq_mhz`, `system_pkg_temp_centigrade` all have `event_name` column (AC_DISPLAY_ON, DC_DISPLAY_OFF, etc.) with per-event stats (min, avg, median, max). Our raw `hw_metric_stats` has daily aggregates without event_name breakdown. Queries aggregate across all events anyway, so this doesn't break anything.

**`system_memory_utilization` ETL (CORRECTED from Intel's SQL):**
- Raw `average` column = **average free memory in MB** (NOT a percentage!). Previous assumption was wrong.
- `sysinfo_ram = ram * 2^10` (converts sysinfo `ram` from GB → MB)
- `avg_free_ram = SUM(sample_count * average) / SUM(sample_count)` (weighted avg free MB)
- `utilized_ram = sysinfo_ram - avg_free_ram`
- `avg_percentage_used = ROUND((sysinfo_ram - avg_free_ram) * 100 / sysinfo_ram)`
- Must JOIN with sysinfo on guid, filter `ram != 0`, GROUP BY `guid, dt, ram`.

**`system_batt_dc_events` is per-guid-per-dt** (not per-guid):
- Has `guid`, `dt`, `duration_mins`, `num_power_ons`, plus 6 battery percentage columns. Our `__tmp_batt_dc_events` must be aggregated per `(guid, dt)`.

**`system_mods_power_consumption`** (lines 182-201):
- Power values are in mW (milliwatts). Categories: cpu, display, disk, mbb (mainboard), network, soc, loss, other, total.

### CONFIRMED: hw_metric_stats has all 4 metrics (risk resolved)

Verified from `notebooks/01-data-exploration.ipynb`. The `name` column in `hw_metric_stats` contains:

| Reporting table needed | hw_metric_stats `name` value | Rows | Guids |
|---|---|---|---|
| system_psys_rap_watts | `HW::PACKAGE:RAP:WATTS:` + `HW:::PSYS_RAP:WATTS:` | 4,846 | 816 |
| system_pkg_C0 | `HW::PACKAGE:C0_RESIDENCY:PERCENT:` | 945,500 | 8,943 |
| system_pkg_avg_freq_mhz | `HW::CORE:AVG_FREQ:MHZ:` | 12,844 | 613 |
| system_pkg_temp_centigrade | `HW::CORE:TEMPERATURE:CENTIGRADE:` | 13,091 | 622 |
| system_hw_pkg_power | `HW::PACKAGE:IA_POWER:WATTS:` (+ GT, DRAM, etc.) | 318,791 | ~800 |

**Note:** PSYS_RAP has very few rows (4,846 across 816 guids in our 1/8th sample). The 5-way chassis join query will work but may have limited guid overlap. The PKG_C0 metric dominates with 945K rows / 8,943 guids.

All 43 distinct metric names are documented in the notebook output. Other notable metrics: CPI (14M rows), C-state residencies (C2-C10, ~1M rows each), memory bandwidth, GT frequency.

### CONFIRMED: all data downloads are valid

From exploration notebook, verified:
- **100% guid overlap** between all event tables and sysinfo (every event-table guid exists in sysinfo)
- **network consumption**: `input_description` contains exact strings the queries need (`OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::` and `BYTES SENT/SEC`)
- **web_cat_pivot**: all 29 columns (guid + 28 categories) match persona query SQL exactly
- **os_memsam_avail_percent**: `average` = % available (need `100 - average` for % used), `sample_count` = nrs, sysinfo_ram via JOIN
- **sysinfo**: `chassistype`, `countryname_normalized`, `persona`, `cpu_family`, `processornumber`, `ram`, `os` all present with expected values
- **update .gz files**: all parseable by DuckDB `read_csv(auto_detect=true)`

### Current query coverage: 21 feasible, 3 permanently infeasible

**Feasible (21):**

*Original 13 (from first data download):*
1. `Xeon_network_consumption` — geographic/demographic
2. `avg_platform_power_c0_freq_temp_by_chassis` — aggregate stats + 5-way join
3. `battery_on_duration_cpu_family_gen` — demographic breakdown
4. `battery_power_on_geographic_summary` — geographic breakdown
5. `mods_blockers_by_osname_and_codename` — aggregate stats + join
6. `most_popular_browser_in_each_country_by_system_count` — ranked top-k
7. `on_off_mods_sleep_summary_by_cpu_marketcodename_gen` — demographic breakdown
8. `persona_web_cat_usage_analysis` — complex multi-way pivot
9. `pkg_power_by_country` — geographic breakdown
10. `popular_browsers_by_count_usage_percentage` — histogram/distribution
11. `ram_utilization_histogram` — histogram/distribution
12. `server_exploration_1` — aggregate stats + join
13. `top_mods_blocker_types_durations_by_osname_and_codename` — aggregate stats + join

*Newly unlocked (8, from second data download):*
14. `display_devices_connection_type_resolution_durations_ac_dc` — aggregate stats + join
15. `display_devices_vendors_percentage` — aggregate stats
16. `userwait_top_10_wait_processes` — ranked top-k
17. `userwait_top_10_wait_processes_wait_type_ac_dc` — ranked top-k + pivot
18. `userwait_top_20_wait_processes_compare_ac_dc_unknown_durations` — ranked top-k + pivot
19. `top_10_applications_by_app_type_ranked_by_focal_time` — ranked top-k
20. `top_10_applications_by_app_type_ranked_by_system_count` — ranked top-k
21. `top_10_applications_by_app_type_ranked_by_total_detections` — ranked top-k

All 5 proposal query types covered: aggregate stats with joins (6), ranked top-k (7), geographic/demographic breakdowns (4), histograms/distributions (2), complex multi-way pivots (1), plus blocker analysis (1).

**Permanently infeasible (3):** `ranked_process_classifications`, `top_10_processes_per_user_id_ranked_by_total_power_consumption`, `top_20_most_power_consuming_processes_by_avg_power_consumed` — all require `system_mods_power_consumption`. The only pre-aggregated source (`mods_sleepstudy_power_estimation_data_13wks`) is a stub with 10K rows and 1 guid. The raw alternative (`plist_process_resource_util_hist`, 1.6 TB, 320 files × 5.5 GiB) does NOT have the needed columns (`user_id`, `total_power_consumption`) — it has CPU/IO/memory resource metrics, not power watts. `__tmp_soc_cpu_power_sysinfo` (218 KB, 427 guids) is a sysinfo table, no power consumption data. No viable path exists.

**Note on stub data:** The `mods_sleepstudy_power_estimation_data_13wks` stub (10K rows, 1 guid) has the correct schema and the 3 queries don't reference `guid` at all (they aggregate by `user_id`/`app_id` only). The queries WILL execute and produce rankings, but the data represents a single client's power profile — acceptable for pipeline testing but not population-level analysis.

### All 4 partial queries now unlocked (total: 13 feasible)

1. **`mods_sleepstudy_top_blocker_hist.txt000.gz`** (1.88 GiB, 92M rows) — DOWNLOADED. Unlocks both blocker queries.
2. **`__tmp_batt_dc_events.txt000.gz`** (12 MiB) — DOWNLOADED. Pre-aggregated battery events. Unlocks both battery queries.

### Data file sizes on disk (actual)

| File | Size | Rows | Guids |
|---|---|---|---|
| system_sysinfo_unique_normalized (8 parquets) | 77.5 MiB | 1,000,000 | 1,000,000 |
| web_cat_pivot (8 parquets) | 55.4 MiB | 512,077 | 512,077 |
| data_dictionary (1 parquet) | 40 KiB | 2,248 | — |
| web_cat_usage_v2 (1/8 parquets) | 864 MiB | 21,354,922 | 64,276 |
| hw_metric_stats (1/8 parquets) | 1.17 GiB | 56,196,901 | 28,896 |
| os_network_consumption_v2 (1/8 parquets) | 1.81 GiB | 121,843,286 | 37,224 |
| os_memsam_avail_percent (1/8 parquets) | 2.03 GiB | 21,688,089 | 69,552 |
| system_cpu_metadata.txt000.gz | 42.5 MiB | 1,000,000 | — |
| system_os_codename_history.txt000.gz | 17.6 MiB | 639,223 | — |
| guids_on_off_suspend_time_day.txt000.gz | 16.8 MiB | 1,582,017 | — |
| mods_sleepstudy_top_blocker_hist.txt000.gz | 1.89 GiB | 92,460,980 | — |
| mods_sleepstudy_recent_usage_instance.txt000.gz | 12 KiB | 106 | — |
| __tmp_batt_dc_events.txt000.gz | 12 MiB | ~49K | — |
| mods_sleepstudy_power_estimation_data_13wks.txt000.gz | 218 KB | 10,000 | 1 (STUB — queries work but single-client) |
| __tmp_soc_cpu_power_sysinfo.txt000.gz | 17 KB | 427 | 427 (sysinfo-only, no power data) |
| display_devices.txt000.gz | 6.16 GiB | 220,997,262 | 209,239 |
| userwait_v2/0000_part_00.parquet | 4.89 GiB | 175,223,880 | 38,142 |
| __tmp_fgnd_apps_date.txt003.gz | 1.53 GiB | 56,755,998 | 55,830 |
| **Grand total** | **~20.7 GiB** | | |

### Second download round — unlocking 8 more queries (COMPLETED)

Downloaded 3 additional files to bring coverage from 13 to 21 queries:

**`system_userwait`** (unlocked 3 queries) — CONFIRMED
- Downloaded: `userwait_v2/0000_part_00.parquet` (4.89 GiB, 175M rows, 38,142 guids)
- This is raw event-level data. Must aggregate to build reporting table.
- **Column mapping** (raw → reporting): `proc_name_current` → `proc_name`, `ac_dc_event_name` → `acdc`, `SUM(duration_ms)` → `total_duration_ms`, `COUNT(*)` → `number_of_instances`. Aggregation: `GROUP BY guid, proc_name_current, event_name, ac_dc_event_name`.
- 28 raw columns; only 6 needed for reporting table.

**`system_display_devices`** (unlocked 2 queries) — CONFIRMED
- Downloaded: `display_devices.txt000.gz` (6.16 GiB, 221M rows, 209,239 guids)
- Pre-aggregated from `dca_update_dec_2024/`. All 7 needed columns present: `connection_type`, `resolution_heigth` (sic), `resolution_width`, `duration_ac`, `duration_dc`, `vendor_name`, `guid`.
- 22 columns total; query columns all verified.

**`system_frgnd_apps_types`** (unlocked 3 queries) — CONFIRMED
- Downloaded: `__tmp_fgnd_apps_date.txt003.gz` (1.53 GiB, 56.8M rows, 55,830 guids)
- All 5 needed columns present: `app_type`, `exe_name`, `totalsecfocal_day`, `lines_per_day`, `guid`.
- 9 columns total. Has some malformed rows in `company_short` (embedded tabs in addresses); use `ignore_errors=true` when reading with DuckDB.
- `frgnd_v2_daily_summary` was checked on Globus: 28.5 GiB (5 files) — WORSE than `__tmp_fgnd_apps_date`. Ruled out. Raw `frgnd_system_usage_by_app` (337.5 GiB) not feasible.

### Dead-end investigations (for posterity)

**`system_mods_power_consumption`** (3 queries permanently infeasible):
- `mods_sleepstudy_power_estimation_data_13wks.txt000.gz` — 218 KB, 10,000 rows, **1 guid only**. Stub/test data. Has the correct schema (`user_id`, `app_id`, `total_power_consumption`) and the queries don't reference `guid`, so they technically execute — but single-client data is not meaningful for population-level DP synthesis.
- `plist_process_resource_util_hist` — 1.6 TB (320 parquet files × 5.5 GiB each). Schema checked via manifest: has `proc_name`, `cpu_user_sec`, `cpu_kernel_sec`, IO/memory metrics — but **NO `user_id` or `total_power_consumption`** columns. Wrong data source entirely (process resource utilization ≠ sleep study power estimation).
- `__tmp_soc_cpu_power_sysinfo` — 17 KB, 427 guids. Just another sysinfo table (chassis, CPU code, persona). No power consumption data.
- `mods_sleepstudy_scenario_instance_13wks` — 1.21 GiB on Globus. Not downloaded; same mods_sleepstudy family but contains scenario instances, not power per process.
- **Conclusion**: No viable path to multi-guid `system_mods_power_consumption` data. All 3 queries (`ranked_process_classifications`, `top_10_processes_per_user_id_ranked_by_total_power_consumption`, `top_20_most_power_consuming_processes_by_avg_power_consumed`) are dropped from the benchmark.

### userwait_v2 schema (from manifest + verified)

16 files across 8 partitions (each partition split into `_part_00` + `_part_01`). We downloaded `0000_part_00` (4.89 GiB, 175M rows, 38K guids).

Full table: 62.8 GiB total, 2,233,128,469 rows across 16 files.

Raw columns (28): `load_ts`, `batch_id`, `audit_zip`, `audit_internal_path`, `guid`, `interval_start_utc`, `interval_end_utc`, `interval_local_start`, `interval_local_end`, `dt`, `ts`, `event_name`, `duration_ms`, `pid_current`, `proc_name_ts_current`, `proc_name_current`, `proc_pkg_current`, `captioned_current`, `windowed_mode_current`, `non_responsive_current`, `ac_dc_event_name`, `pid_next`, `proc_name_ts_next`, `proc_name_next`, `proc_pkg_next`, `captioned_next`, `windowed_mode_next`, `non_responsive_next`.

Reporting table aggregation SQL:
```sql
SELECT guid, 
       UPPER(proc_name_current) AS proc_name,
       event_name,
       ac_dc_event_name AS acdc,
       SUM(duration_ms) AS total_duration_ms,
       COUNT(*) AS number_of_instances
FROM userwait_v2
GROUP BY guid, proc_name_current, event_name, ac_dc_event_name
```

### Globus directory structure (from live listing)

```
/projects/dca/
├── guides/                              # DCA data dictionary, table summary, SOPs
├── university_analysis_pad/             # Metadata schema
│   ├── data_dictionary/                 # 1 parquet, 2248 rows (column-level metadata)
│   ├── data_dictionary_collector_il/    # 1 parquet
│   ├── data_dictionary_collector_inputs/# 1 parquet
│   ├── data_dictionary_tables/          # 1 parquet
│   ├── system_sysinfo_unique_normalized/# 8 parquets (~10 MiB each, 1M rows total)
│   └── *-manifest.json                  # Schema/size manifests for each table
├── university_prod/                     # Main telemetry data (~115 tables)
│   ├── dca_update_dec_2024/             # NEW DATA added Jan 2025
│   ├── batt_acdc_events/                # 34 GiB, multi-parquet
│   ├── hw_metric_stats/                 # 8.7 GiB, multi-parquet
│   ├── os_network_consumption_v2/       # 13 GiB, multi-parquet
│   ├── os_memsam_avail_percent/         # 15 GiB, multi-parquet
│   ├── web_cat_usage_v2/               # 6.4 GiB, multi-parquet
│   ├── web_cat_pivot/                   # 53 MiB
│   ├── userwait_v2/                     # 59 GiB (large)
│   ├── frgnd_system_usage_by_app/       # 338 GiB (massive)
│   ├── power_acdc_usage_v4_hist/        # 342 GiB (massive)
│   ├── plist_process_resource_util_hist/ # 1.6 TB (enormous)
│   ├── eventlog_item_hist/              # 2.8 TB (largest)
│   ├── ... (~100 more tables)
│   └── *-manifest.json
└── public/                              # Gaming/FPS analytics (~80+ tables)
    ├── ucsd_apps_execlass_final/         # 19 MiB (NOT used by benchmark queries)
    ├── gaming_*/                         # Various FPS analysis tables
    └── ...
```

Each table folder contains split parquet files like `0000_part_00.parquet`, `0001_part_00.parquet`, etc. (up to ~6.2 GiB per file). Manifests contain URLs, file sizes, row counts, and full column schemas.

## Style notes

- Academic tone, LaTeX formatting conventions. Use `\citep{}` / `\citet{}` for citations.
- Code should follow best practices per DSC 180 methodology lessons (reproducible, documented, build scripts, library code in .py not notebooks).
- GitHub repo must be public with README describing environment setup and reproduction steps.

### Writing style rules (MANDATORY for all output)

- NEVER use bolds (no `**text**` in markdown, no bold in any output).
- NEVER use em dashes. Use commas, periods, or semicolons instead.
- Use clear, spartan, informative language.
- Avoid metaphors, cliches, generalizations.
- Avoid setup language: "in conclusion", "in closing", "in summary", "moreover", etc.
- Avoid constructions like "not just X, but also Y".
- Avoid unnecessary adjectives and adverbs.
- Avoid asterisks, hashtags in prose.
- Avoid these words: delve, embark, enlightening, esteemed, shed light, craft, crafting, imagine, realm, game-changer, unlock, discover, skyrocket, abyss, not alone, in a world where, revolutionize, disruptive, utilize, utilizing, dive deep, tapestry, illuminate, unveil, pivotal, intricate, elucidate, hence, furthermore, realm, however, harness, exciting, groundbreaking, cutting-edge, remarkable, it remains to be seen, glimpse into, navigating, landscape, stark, testament, moreover, boost, skyrocketing, opened up, powerful, inquiries, ever-evolving.

### Jupyter notebook style rules (MANDATORY)

- NEVER leave comments in code cells. No inline comments, no block comments.
- Do not spam `print()` with newlines and f-strings for output.
- Use `from IPython.display import display, Markdown, HTML` to render formatted output.
- Use `display(Markdown(...))` for section headers, tables, and explanatory text within code cells.
- Use `display(df)` or `display(df.head())` to show dataframes instead of `print(df.to_string())`.

---

## Agent #2 notes (context handoff)

### What Agent #1 accomplished

1. Initialized the repo, added both submodules (`dsc-180a-q1`, `DSC180B-Q2`), fixed SSH aliases for portability.
2. Read every reference paper in full and wrote comprehensive notes into this file (DP foundations, DP-SGD, PE, Aug-PE, PE-tabular).
3. Identified the critical "reporting schema gap" — the 24 queries reference `reporting.system_*` tables that don't exist in raw `university_prod`.
4. Discovered the `dca_update_dec_2024/` folder containing pre-aggregated tables from Intel, resolving most gaps.
5. Devised and executed a data download strategy: ~8 GiB total across 7 parquet tables + 6 gzipped text files from the update folder.
6. Created `notebooks/01-data-exploration.ipynb` — confirmed all data is valid, 100% guid overlap, correct column names/values, 13 of 24 queries feasible.
7. Set up `uv` with `pyproject.toml` (duckdb, pandas, ipykernel, jupyter). Created `.gitignore`, `data/README.md`, pushed initial commit to `git@github.com-jktrns:jktrns/dsc-180b-q2.git`.

### Synthesis methodology (decided in Agent #2 session)

The unit of synthesis is the guid. Each guid is one client system. All 19 reporting tables are different measurements of the same set of clients. DP guarantees are per-person, so the privacy unit must align with the data unit.

Synthesizing each reporting table independently would destroy cross-table correlations. Most benchmark queries JOIN multiple tables on guid (e.g., server exploration joins network consumption with sysinfo, the 5-way chassis query joins sysinfo with four hw_metric tables). If synthetic tables have no relationship to each other, JOINs produce either zero rows (mismatched guids) or random associations (correct guids but uncorrelated attributes). Queries that measure cross-table relationships (like "average network consumption by chassis type") would return meaningless results.

The correct approach: build a single wide table with one row per guid containing all attributes and pre-aggregated metrics from every reporting table. Train the DP-VAE on this wide table. The synthetic output preserves the joint distribution: a synthetic guid with chassis="Notebook" will tend to have the network/memory/power patterns that real notebooks have.

For multi-row-per-guid tables (userwait has rows per proc_name, web_cat_usage has rows per browser/category, frgnd_apps has rows per exe_name), expand categorical breakdowns into separate columns in the wide table. For example, instead of one browser duration column, create chrome_duration, edge_duration, firefox_duration columns per guid. The number of columns does not matter as long as the privacy unit (guid) stays consistent.

Steps to build the wide training table:
1. For each reporting table, write a guid-level aggregation that produces the columns the benchmark queries reference. Examples: `SUM(nrs)` for network, `SUM(nrs * avg_percentage_used) / SUM(nrs)` for memory, `COUNT(*)` for battery power-ons.
2. LEFT JOIN all guid-level aggregations onto sysinfo (the anchor table with 1M guids).
3. Result: one row per guid, ~50-200 columns depending on how many categorical breakdowns are expanded.
4. Train DP-VAE (and later PE) on this wide table.
5. Generate synthetic wide table rows.
6. Decompose synthetic wide table back into the 19 reporting table schemas. Run original benchmark SQL unchanged.

This approach was chosen over three alternatives:
- Independent per-table synthesis: destroys cross-table correlations, JOINs fail.
- Guid-level aggregation with lossy regeneration of fine-grained rows: unnecessary complexity since queries only consume aggregated statistics anyway.
- Two-stage conditional synthesis: guid attributes first, then event rows conditioned on guid. More complex, deferred to future work.

The research question from the proposal ("Does error compound across multi-table joins?") is directly addressed by this design. The wide-table approach should preserve joint distributions, while an independent baseline would show how much error compounds when correlations are lost.

### Current repo state

- `notebooks/01-data-exploration.ipynb` — data validation notebook (run successfully, outputs preserved).
- `notebooks/02-query-feasibility.ipynb` — 147/147 column checks passed, all 24 queries verified.
- `notebooks/03-build-reporting-tables.ipynb` — 19 reporting parquet files, ~11.5 GiB.
- `notebooks/04-run-benchmark-queries.ipynb` — 24/24 queries executed, ground truth CSVs in `data/results/real/`.
- `notebooks/05-dp-sgd.ipynb` — DP-VAE training + synthetic generation + benchmark evaluation (complete, see results below).
- `pyproject.toml` — deps: duckdb, pandas, ipykernel, jupyter, torch, opacus, scikit-learn, matplotlib
- `docs/` — CLAUDE.md, papers, queries (24 JSON), q1-report.tex, q2-proposal.tex, chat logs
- `data/models/` — `dp_vae_checkpoint.pt` (model weights + training history), `transformer.pkl` (sklearn ColumnTransformer), `training_curves.png`
- `data/reporting/` — 19 real reporting parquets + `wide_training_table.parquet` + `synth_wide_training_table.parquet`
- `data/reporting/synthetic/` — 12 synthetic reporting parquets (decomposed from synthetic wide table)
- `data/results/real/` — ground truth CSVs for all 24 queries
- `data/results/synthetic/` — synthetic result CSVs for 8 evaluated queries
- `data/` — gitignored parquet/gz files (~20.7 GiB total), tracked manifests and README

### Completed work (Agent #2)

1. Expanded data coverage from 13 to 24 queries (~20.7 GiB downloaded).
2. Created `notebooks/02-query-feasibility.ipynb` (147/147 column checks passed, all 24 queries verified).
3. Created `notebooks/03-build-reporting-tables.ipynb` (19 reporting parquet files, ~11.5 GiB).
4. Created `notebooks/04-run-benchmark-queries.ipynb` (24/24 queries executed successfully, ground truth CSVs saved).
5. Discovered Intel's actual ETL SQL (`scratch reporting analytics queries.sql`), corrected memory utilization formula.
6. Updated `data/README.md` with complete download/reproduction instructions.

### Completed work (Agent #3)

1. Created `notebooks/05-dp-sgd.ipynb` — full DP-SGD pipeline:
   - Built guid-level wide training table: 1,000,000 rows x 70 columns (9 categorical, 59 numeric, 1 guid). LEFT JOIN of all reporting tables onto sysinfo anchor. Categoricals top-k binned (k=50), numerics clipped at 99.9th percentile + log1p transformed + StandardScaled. Final feature matrix: 1M x 307 (248 one-hot + 59 scaled numeric).
   - DP-VAE architecture: encoder 307→512→512→(64 mean, 64 logvar), 9 categorical decoder heads (one per column, cross-entropy loss), 1 numeric decoder head (64→59, MSE loss), KL divergence regularization. Total parameters: 505,971.
   - Training: batch size 4096, 20 epochs, Adam lr=1e-3, max_grad_norm=1.0, Opacus `make_private_with_epsilon` auto-calibrated noise multiplier to reach ε=4.0 at δ=1e-5. Final ε=3.996. Training time: 359.7 minutes on CPU (~18 min/epoch).
   - Generation: 1M synthetic rows via z~N(0,I), softmax sampling for categoricals, inverse StandardScale + expm1 for numerics.
   - Decomposition: 12 synthetic reporting tables reconstructed from synthetic wide table.
   - Evaluation: 8 benchmark queries executed on synthetic data, compared with ground truth.

2. Model checkpoint saved to `data/models/dp_vae_checkpoint.pt` (survives kernel restarts).

### DP-SGD benchmark results (notebook 05)

8 of 21 feasible queries evaluated (those whose reporting tables are fully reconstructable from the wide table). 13 skipped: userwait (3), frgnd_apps (3), display (2), blocker (2), on_off/battery needing cpu_metadata.generation (2), server_exploration needing model_normalized/#ofcores (1).

Overall accuracy: 2/49 columns within 10% relative error, 3/49 within 25%, 6/49 within 50%.

One strong result: browser ranking query — 42/50 countries got the correct most-popular browser. The model learned the joint distribution between countryname_normalized and browser preference well enough to preserve rankings.

All continuous metric columns failed badly (>99% relative error):

| Metric | Real | Synthetic | Issue |
|---|---|---|---|
| avg_psys_rap_watts | 4.29 | 0.002 | ~816 guids with data out of 1M (0.08%) |
| avg_pkg_c0 | 42.6% | 0.022% | ~9K guids (0.9%) |
| avg_freq_mhz | 2693 | 0.007 | ~613 guids (0.06%) |
| avg_temp_centigrade | 44.7°C | 0.003 | ~622 guids (0.06%) |
| avg_bytes_received | 7.36e16 | 1.12 | ~37K guids (3.7%) |
| avg_duration (battery) | 144 min | 0.11 min | sparse |
| avg_percentage_used (memory) | 42.6% | 0.0% | ~70K guids (7.0%) |

Root cause: sparsity in the wide table. Most metric columns are zero for 93-99.9% of guids. The VAE learns that these columns are almost always 0 and generates near-zero values. The KL regularization pushes the latent distribution toward N(0,I), and the MSE loss optimizes for the mode (0) on sparse columns.

Secondary issue: system counts are inflated. The 5-way chassis join returns ~104 real guids (only those with ALL 5 metrics), but synthetic data returns ~163K guids because the model generates small positive values for everyone, so the INNER JOIN matches far more rows.

Decomposition artifact: popular_browsers percent_instances = 33.33% for all browsers in synthetic (should be ~52%/33%/14% for chrome/edge/firefox) because the reconstruction creates exactly 1 row per guid per browser, losing real instance-count variation.

### Why the wide-table DP-SGD approach failed

The wide table design (one row per guid, all metrics as columns) preserves cross-table correlations in theory, but creates extreme zero-inflation in practice. When only 0.06-7% of guids have nonzero values for a column, the column's distribution is essentially a point mass at 0 with a tiny tail. A VAE with MSE loss collapses to generating near-zero values.

This is a known failure mode. Potential mitigations (not yet implemented):
1. Train only on guids with sufficient coverage (e.g., present in >= 3 metric tables). Reduces sparsity but loses the population-level synthesis goal.
2. Two-stage model: first a Bernoulli model for zero/nonzero per column, then a conditional model for nonzero values only.
3. Per-table independent synthesis: loses cross-table correlations but eliminates sparsity. Useful as a baseline.
4. Mixture model or zero-inflated loss function instead of plain MSE.
5. Restrict the wide table to only the subset of guids and columns relevant to each query cluster.

For the report, this negative result directly answers research question (2) from the proposal: "Does error compound across multi-table joins?" The answer is yes, dramatically, when sparsity makes the joint distribution nearly degenerate.

### What needs to happen next (priority order)

1. Implement Private Evolution on DCA data (no PE code exists yet).
2. Write report checkpoint: title/abstract/intro/methods + DP-SGD results section documenting the sparsity failure.
3. Build poster draft.
4. Deliverables due Sun Feb 15: report checkpoint, code checkpoint, poster checkpoint.
5. Website due Sun Feb 22 (skeleton on GitHub Pages).
6. Final everything due Sun Mar 8; capstone showcase Fri Mar 13.

### Grading structure (Q2)

- Checkpoints: 10%
- Report (graded by mentor): 40%
- Website (graded by TA): 7.5%
- Poster + presentation (graded by mentor + TA): 10%
- Code artifact (graded by TA): 7.5%
- Ethics: 5%
- Participation: 15% + 5% overall

### Key constraints

- Use `uv`, not raw `python3`/`pip`.
- LaTeX reports use XeLaTeX + BibTeX with custom `dsc180reportstyle`.
- Code must be in `.py` library files, not just notebooks.
- Git remote uses SSH alias `github.com-jktrns` locally; `.gitmodules` uses standard `github.com` for portability.
- Data files are gitignored; only manifests and README tracked.
- 21/24 queries feasible. 3 permanently infeasible (mods_power_consumption — stub data only, raw source lacks needed columns).
- Data files total ~20.7 GiB on disk.
- `__tmp_fgnd_apps_date.txt003.gz` needs `ignore_errors=true` when reading with DuckDB (some rows have malformed `company_short` with embedded tabs).
