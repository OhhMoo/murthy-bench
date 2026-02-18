# Molecular Simulation Trading System
## Complete Pipeline, Physical Models & Development Roadmap

---

## Executive Summary

This system directly implements computational algorithms from **Frenkel & Smit's "Understanding Molecular Simulation"** onto financial markets — not as a loose metaphor, but as precise mathematical transplants. Every module maps a specific physical model to a financial analog with justified, rigorous correspondence. The core engine uses **Langevin dynamics** for pairs trading on equities and crypto, extended with Monte Carlo sampling, free energy estimation, and enhanced sampling techniques across the full pipeline.

---

## Part I: Physical Models Used — Step by Step

---

### Step 1: Pair Universe Construction — *Cointegration as Molecular Bonding*

**Physical Model: Effective Pair Potential / Lennard-Jones Bonding Criterion**

In molecular simulation, two atoms are considered "bonded" or "interacting" if their effective pair potential V(r) has a stable minimum at some separation r*. The bond exists as long as the system energy keeps the pair near that minimum. Weak or absent potentials mean no persistent interaction.

We map this directly: two assets form a "bound pair" if their log-price spread has a stable potential minimum — i.e., it is cointegrated. The Engle-Granger and Johansen cointegration tests are the financial equivalent of computing whether V(r) has a well-defined minimum. The hedge ratio β plays the role of the equilibrium bond length r*.

**Why this model:** Just as simulating random uncorrelated atoms wastes compute, trading uncorrelated asset pairs destroys capital. The binding criterion filters the universe to only interacting pairs — those where the physics of mean-reversion actually applies.

**F&S Reference:** Chapter 2 (Potential energy functions), Chapter 4 (Pair correlations and structure).

```
Spread: x(t) = log(P_A(t)) - β · log(P_B(t))
Bond exists ↔ x(t) is I(0) [stationary]
β = hedge ratio = equilibrium bond length r*
```

---

### Step 2: Langevin Parameter Estimation — *The Core Physics Engine*

**Physical Model: Langevin Equation (Brownian particle in a harmonic potential)**

The Langevin equation describes a massive particle subject to:
- [ ] A **deterministic restoring force** (friction / mean-reversion)
- [ ] A **stochastic thermal force** (market noise / random kicks from other traders)
- [ ] An **external potential** (the fundamental spread equilibrium)

The full equation implemented:

```
dx(t) = -γ · x(t) · dt  +  σ · dW(t)
```

Where:
- [ ] `x(t)` = spread deviation from equilibrium
- [ ] `γ` = friction coefficient = **mean-reversion speed** (estimated via MLE on OU process)
- [ ] `σ` = noise amplitude = **volatility of the spread**
- [ ] `dW(t)` = Wiener increment (market microstructure noise)
- [ ] Equilibrium `x* = 0` (after centering the spread)

This is the **Ornstein-Uhlenbeck process**, which is the exact solution to a Langevin equation in a harmonic potential V(x) = ½γx². The stationary distribution is:

```
p(x) ∝ exp(-V(x) / T_eff)  =  exp(-γx² / 2σ²)   [Boltzmann distribution]
```

**Numerical Integration:** We use the **Euler-Maruyama scheme** (stochastic analog of the Verlet integrator) for forward simulation:

```
x(t + Δt) = x(t) - γ · x(t) · Δt  +  σ · √Δt · ε,   ε ~ N(0,1)
```

**Parameter Estimation (MLE on discretized OU):**

Given discrete observations at intervals Δt, the exact conditional distribution is Gaussian:

```
x(t+Δt) | x(t)  ~  N( x(t)·e^(-γΔt),  σ²(1 - e^(-2γΔt)) / 2γ )
```

Parameters γ, σ are estimated by maximizing the log-likelihood over a rolling window of observed spreads.

**Why this model:** Langevin dynamics is the simplest physically-grounded model for a mean-reverting system in a noisy bath. It has exact analytical solutions, making parameter estimation tractable, and the physical intuition is directly tradeable: γ tells you how fast to expect reversion, σ tells you how much noise to tolerate, and T_eff = σ²/2γ tells you the "temperature" of the pair — how energetically disordered it currently is.

**F&S Reference:** Chapter 9 (Brownian dynamics), Section 9.3 (Langevin equation), Section 9.4 (Numerical integration schemes).

---

### Step 3: Effective Temperature Estimation — *Fluctuation-Dissipation Theorem*

**Physical Model: Fluctuation-Dissipation Theorem (FDT)**

In statistical mechanics, the FDT states that the spontaneous fluctuations of a system at equilibrium are fundamentally related to its response to external perturbations. For a Langevin system:

```
T_eff = σ² / (2γ)
```

This is not a metaphor — it is the exact formula for the effective temperature of a Brownian particle from FDT. In the trading context:

| Physics | Finance |
|---|---|
| High T (hot bath) | High-volatility, low-mean-reversion regime |
| Low T (cold bath) | Tight, fast-reverting, high-confidence pair |
| T → ∞ | Pair is breaking down, cease trading |
| T → 0 | Near-mechanical arbitrage opportunity |

T_eff is used to **dynamically rescale position sizing** via Boltzmann weighting:

```
Position size ∝ exp(-ΔF / T_eff)
```

where ΔF is the free energy cost of the current spread displacement.

**Why this model:** T_eff provides a physically principled, dimensionless measure of pair quality that automatically accounts for the interplay of noise and restoring force. It replaces ad hoc volatility scaling with a thermodynamically grounded risk measure.

**F&S Reference:** Chapter 3 (Monte Carlo simulations), Section on thermal averages and fluctuations.

---

### Step 4: Free Energy Calculation — *Signal Generation via F = U - TS*

**Physical Model: Helmholtz Free Energy in a Harmonic Potential**

The free energy F = U - TS captures the total "tradeable energy" of a configuration, penalized by entropic uncertainty. For a harmonic potential:

```
F(x) = ½ · γ · x²          [potential energy U(x)]
ΔF   = F(x_current) - F(0) = ½ · γ · x²   [free energy displacement]
```

**Entry signal:** Enter a trade when the free energy displacement exceeds transaction costs:

```
Enter long spread  ↔  x(t) < -z_threshold · √(T_eff / γ)
Enter short spread ↔  x(t) > +z_threshold · √(T_eff / γ)
```

The threshold is set in units of the thermal length scale `ξ = √(T_eff / γ)` (the RMS fluctuation of the OU process), making it physically dimensionless.

**Exit signal:** Exit when x(t) returns to within ε of equilibrium — i.e., when the particle returns to the potential well minimum.

**Stop-loss:** When |x(t)| > x_max, the particle has "escaped the potential well" — the pair bond is breaking. Exit and flag for re-estimation. This is the trading equivalent of bond dissociation.

**Why this model:** Using free energy rather than raw z-score separates *real alpha* (exploiting genuine free energy gradients) from *noise alpha* (chasing random fluctuations). A spread can be large in absolute terms but small relative to T_eff — the free energy formulation automatically discounts such signals.

**F&S Reference:** Chapter 7 (Free energy calculations), Chapter 8 (The chemical potential).

---

### Step 5: Ergodicity Testing — *Detecting Regime Breaks*

**Physical Model: Ergodicity and Sampling Quality (from Enhanced Sampling literature)**

In molecular simulation, a system is ergodic if the time average of any observable equals its ensemble average. Ergodicity breaking occurs when the system gets trapped in a metastable state and fails to sample the full configuration space — the simulation is producing biased, non-representative results.

In markets, ergodicity breaking means: the pair is no longer sampling its theoretical stationary distribution p(x). It is trapped in a new regime. This is the physical signal that a structural break has occurred before it becomes obvious in price levels.

**Test implemented:** The ratio of time-averaged variance to ensemble-predicted variance:

```
Ergodicity ratio E = <x²>_time / (σ² / 2γ)

E ≈ 1.0  →  pair is ergodic, trading normally
E >> 1.0 →  trapped in extended displacement (trending regime)
E << 1.0 →  trapped in compressed range (pre-breakout?)
```

Alerts are triggered when E drifts significantly from 1.0, prompting re-estimation or position reduction.

**Why this model:** Traditional financial models have no principled way to distinguish "the spread is large but will revert" from "the pair relationship has changed." Ergodicity testing provides exactly this: a physics-derived detector for when the theoretical model has stopped being applicable.

**F&S Reference:** Chapter 3 (ergodicity and detailed balance), Chapter 13 (enhanced sampling methods).

---

### Step 6: Detailed Balance Violation Scoring — *Arbitrage Detection*

**Physical Model: Detailed Balance Condition**

In equilibrium statistical mechanics, detailed balance requires that the probability flux between any two states is equal in both directions:

```
p(x_A) · P(x_A → x_B)  =  p(x_B) · P(x_B → x_A)
```

This is the equilibrium condition. When detailed balance is violated, there is a net probability current — the system is being driven out of equilibrium by an external force. In markets, this external force is an **arbitrage**.

We estimate transition probabilities from the empirical spread time series and compute the net probability current:

```
J(x_A → x_B) = p(x_A)·P(x_A→x_B) - p(x_B)·P(x_B→x_A)
```

Large |J| indicates directional probability flow — a persistent mispricing the market has not yet corrected.

**Why this model:** Detailed balance violations are precisely the mathematical condition for non-equilibrium market states. This converts arbitrage detection from a heuristic search into a principled measurement.

**F&S Reference:** Section 3.3 (detailed balance and reversibility), Chapter 13 (non-equilibrium methods).

---

### Step 7: HPC Parameter Sweep — *Parallel Tempering*

**Physical Model: Parallel Tempering (Replica Exchange Monte Carlo)**

Parallel tempering runs multiple copies (replicas) of a system simultaneously at different temperatures T_1 < T_2 < ... < T_N, periodically swapping configurations between replicas if the exchange satisfies the Metropolis criterion:

```
P(swap i↔j) = min(1, exp(-(β_i - β_j)(U_j - U_i)))
```

High-temperature replicas explore broadly (avoiding local minima); low-temperature replicas exploit (converging to the best solution). Swaps allow good solutions found by hot replicas to propagate down to cold replicas.

**Trading application:** Run the Langevin trading engine simultaneously across a grid of (γ, σ, β, window_length) parameter sets. Use the parallel tempering exchange criterion to propagate high-performing parameter configurations toward the "cold" (production) replica. This solves the problem of overfitting to a single parameter set found by naive grid search.

**Implementation:** Ray remote functions, one actor per replica, exchange coordination via Ray shared memory.

**Why this model:** Parallel tempering is provably better than random search or grid search for multimodal optimization landscapes — exactly the kind of landscape produced by parameter optimization in non-stationary financial time series.

**F&S Reference:** Chapter 13 (advanced Monte Carlo techniques), Section 13.2 (parallel tempering).

---

### Step 8: Tail Risk & Drawdown Estimation — *Umbrella Sampling*

**Physical Model: Umbrella Sampling**

Umbrella sampling is an enhanced sampling technique for estimating the probability of rare events that would almost never occur in standard simulations. A series of biasing potentials (umbrellas) are added to force the system to sample high-energy, low-probability regions:

```
V_biased(x) = V(x)  +  w_i · (x - x_i)²
```

The true probability is recovered via the WHAM (Weighted Histogram Analysis Method) reweighting. This gives the full free energy profile F(x) including the rare tails.

**Trading application:** Estimate the probability distribution of extreme spread displacements — the financial "energy barriers." This gives:
- [ ] Probability of gap events exceeding any threshold
- [ ] Expected drawdown under stress scenarios
- [ ] VaR / CVaR computed from the full physical distribution rather than a Gaussian assumption

**Why this model:** Standard VaR assumes Gaussian tails. Umbrella sampling recovers the *actual* tail shape of the spread distribution, including non-Gaussian fat tails that arise from microstructure effects and regime changes.

**F&S Reference:** Chapter 7 (free energy methods), Section 7.3 (umbrella sampling and WHAM).

---

## Part II: Innovation Points

---

### Innovation 1: Direct Algorithm Transplant (Not Analogy)

Previous econophysics work (Stanley, Mantegna, Bouchaud) uses *concepts* from physics — scaling laws, random matrix theory, criticality — but does not directly implement the computational algorithms. This project implements the **exact numerical schemes** from Frenkel & Smit: the Euler-Maruyama integrator, the Metropolis acceptance criterion, the WHAM reweighting procedure. The innovation is methodological fidelity: the code is a direct port of molecular simulation algorithms to financial data, not a re-derivation from scratch.

### Innovation 2: T_eff as a Dynamic Trading Parameter

Using the fluctuation-dissipation theorem to extract an **effective temperature** T_eff = σ²/2γ from live market data and feeding it back into position sizing is, to our knowledge, novel in systematic trading. This converts a physics theorem into a real-time risk adjuster: the system automatically trades smaller in hot markets and larger in cold markets, with the exact same functional form as the Boltzmann factor.

### Innovation 3: Ergodicity Breaking as a Leading Indicator

Financial risk management typically detects regime changes after the fact — via realized volatility spikes, drawdown thresholds, or correlation breakdowns. Monitoring the ergodicity ratio E = <x²>_time / (σ²/2γ) provides a **leading indicator**: the pair begins failing to sample its theoretical distribution *before* it produces large losses. This is an early warning system derived from statistical mechanics, not from financial heuristics.

### Innovation 4: Detailed Balance as Arbitrage Quantification

Existing arbitrage detection methods (cointegration spread z-score, pairs divergence monitors) measure *magnitude* of displacement. Detailed balance violation scoring measures the **net probability current** — the directionality and persistence of the mispricing. This is a strictly stronger signal because it distinguishes a symmetric large fluctuation (which will revert) from an asymmetric driven deviation (which is genuine arbitrage).

### Innovation 5: Parallel Tempering for Live Parameter Maintenance

Applying parallel tempering to the continuous problem of parameter maintenance in non-stationary markets is novel. Rather than re-optimizing parameters periodically (which creates discontinuities and overfitting), the HPC layer runs a continuously operating parallel tempering chain. Production always runs on the "cold" replica, but benefits from exploration done by hot replicas in real time.

### Innovation 6: Unified Physical State Vector

Every signal in this system derives from a single physical state vector `(x, γ, σ, T_eff, E, J)` — spread, friction, noise, temperature, ergodicity ratio, and probability current. This unified representation means all modules speak the same language, enabling clean composition and making the system interpretable: every trade has a complete physical justification.

---

## Part III: Complete Pipeline with Molecular Simulation Models

```
┌──────────────────────────────────────────────────────────────────┐
│  DATA INGESTION                                                  │
│  Equities: Alpaca / Polygon  →  1min OHLCV bars                 │
│  Crypto:   CCXT WebSocket    →  Tick aggregation → OHLCV        │
│  Output:   Normalized MarketState objects (timestamp, OHLCV)    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  PAIR SCREENING  [Model: Effective Pair Potential]               │
│  - Compute log-price spreads for all asset pairs                 │
│  - Engle-Granger / Johansen cointegration test                  │
│  - Accept pairs where spread is I(0)  →  bound pair universe    │
│  - Estimate β (hedge ratio = equilibrium bond length)           │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  LANGEVIN ENGINE  [Model: OU Process / Brownian Dynamics]        │
│  - Rolling MLE estimation of (γ, σ) from spread time series     │
│  - Compute T_eff = σ²/2γ  [Fluctuation-Dissipation Theorem]    │
│  - Compute thermal length ξ = √(T_eff/γ)                        │
│  - Forward simulate via Euler-Maruyama for scenario generation  │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  FREE ENERGY SIGNAL  [Model: Harmonic Potential / Helmholtz F]  │
│  - Compute ΔF(x) = ½γx²                                         │
│  - Entry: |x| > z* · ξ  AND  ΔF > transaction cost             │
│  - Exit:  |x| < ε · ξ   (return to well minimum)               │
│  - Stop:  |x| > x_max   (potential well escape / bond break)   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  ERGODICITY MONITOR  [Model: Ergodic Sampling Theory]            │
│  - Compute E = <x²>_rolling / (σ²/2γ)                          │
│  - E ≈ 1: pair healthy, full position allowed                   │
│  - E > threshold: reduce position, re-estimate parameters       │
│  - E sustained >> 1: pair flagged for removal                   │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  DETAILED BALANCE SCORER  [Model: Detailed Balance / NESS]      │
│  - Build empirical transition matrix P(x_i → x_j) from data    │
│  - Compute probability current J for each transition            │
│  - High |J| → directional arb opportunity, boost signal         │
│  - J ≈ 0  → symmetric fluctuation, standard sizing             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  PORTFOLIO / RISK LAYER  [Model: Boltzmann Weighting + WHAM]    │
│  - Position size = base_size · exp(-ΔF / T_eff)                 │
│  - Umbrella sampling for tail risk on portfolio spread dist.    │
│  - Correlation-adjusted total exposure limit                    │
│  - Phase diagram monitoring: (γ, σ) across all active pairs    │
└──────────────────────────┬───────────────────────────────────────┘
                           │
          ┌────────────────┼──────────────────┐
          │                │                  │
┌─────────▼──────┐  ┌──────▼──────┐  ┌───────▼──────────┐
│  BACKTEST      │  │  PAPER      │  │  HPC             │
│  Event loop    │  │  TRADING    │  │  SIMULATION      │
│  Historical    │  │  Alpaca     │  │  Parallel        │
│  bar replay    │  │  paper +    │  │  tempering over  │
│  Full metrics  │  │  CCXT sand  │  │  param space     │
└────────────────┘  └─────────────┘  └──────────────────┘
```

---

## Part IV: Week-by-Week Development Timeline

**Total Duration: 12 Weeks**
**Team assumption: 1 quantitative developer, 1 infrastructure engineer (or solo with context switching)**

---

### Week 1 — Foundation & Data Infrastructure

**Goal:** Reliable, unified data pipeline for both equities and crypto.

**Tasks:**
- [ ] Set up project repo structure: `src/data/`, `src/physics/`, `src/signals/`, `src/execution/`, `src/hpc/`, `tests/`, `configs/`
- [ ] Implement `MarketState` dataclass: unified normalized container for OHLCV + metadata
- [ ] Integrate Alpaca API: historical minute bars download, rate-limit handling, local caching to DuckDB
- [ ] Integrate CCXT: connect to Binance, implement tick → OHLCV aggregation, handle WebSocket reconnects
- [ ] Write data normalization layer: align timestamps, handle missing bars, corporate action adjustments for equities
- [ ] Unit tests for data layer: assert shape, dtype, no NaN in OHLCV, timestamp monotonicity

**Deliverables:**
- [ ] `DataManager` class with `.get_bars(symbol, start, end, freq)` interface
- [ ] DuckDB database with 2 years of minute bars for 50 equity symbols + 10 crypto pairs
- [ ] Test suite: 100% pass rate on data layer

---

### Week 2 — Pair Screening Module

**Goal:** Implement the molecular bonding criterion — identify cointegrated pairs.

**Tasks:**
- [ ] Implement `PairScreener` class
- [ ] Log-price spread computation with rolling hedge ratio β via OLS / Kalman filter
- [ ] Engle-Granger cointegration test (via `statsmodels`)
- [ ] Johansen cointegration test for multi-asset spreads
- [ ] Pair ranking by: cointegration p-value, stationarity (ADF), half-life, correlation
- [ ] Persistence layer: save accepted pairs + parameters to DuckDB
- [ ] Cross-asset screening: equities vs equities, crypto vs crypto, equities vs crypto ETFs

**Deliverables:**
- [ ] `PairScreener` with `.screen_universe(symbols)` returning `List[PairConfig]`
- [ ] Ranked pair universe: top 20 equity pairs, top 10 crypto pairs
- [ ] Visualization: spread time series + ADF statistic for top pairs

---

### Week 3 — Core Langevin Physics Engine

**Goal:** Implement the central physical model — OU parameter estimation and Langevin dynamics.

**Tasks:**
- [ ] Implement `LangevinPair` class (the core engine — all other modules depend on this)
- [ ] MLE estimation of (γ, σ) from discretized OU conditional likelihood
- [ ] Compute T_eff = σ²/2γ from fluctuation-dissipation theorem
- [ ] Compute thermal length ξ = √(T_eff/γ)
- [ ] Compute half-life = ln(2)/γ
- [ ] Euler-Maruyama forward integrator for scenario simulation
- [ ] Rolling estimation: re-estimate parameters every N bars on a sliding window
- [ ] Validation: simulate synthetic OU paths, verify MLE recovers known parameters
- [ ] Unit tests: parameter recovery within tolerance for known (γ, σ)

**Deliverables:**
- [ ] `LangevinPair` class: `.fit(spread_series)`, `.simulate(n_steps, n_paths)`, `.state` property
- [ ] `PhysicalState` dataclass: `(x, gamma, sigma, T_eff, xi, half_life, timestamp)`
- [ ] Validation notebook: MLE accuracy vs sample size, parameter stability plots

---

### Week 4 — Free Energy Signal Generation

**Goal:** Convert physical state into entry/exit/stop signals via Helmholtz free energy.

**Tasks:**
- [ ] Implement `FreeEnergySignal` class
- [ ] Compute ΔF(x) = ½γx² for current spread displacement
- [ ] Dynamic threshold: z_threshold in units of ξ (thermal length scale)
- [ ] Transaction cost model: fixed + proportional costs, minimum ΔF to enter
- [ ] Signal states: `FLAT`, `ENTRY_LONG`, `ENTRY_SHORT`, `EXIT`, `STOP`
- [ ] Signal generation loop: consume `PhysicalState` stream, emit `Signal` objects
- [ ] Position sizing: base_size · exp(-ΔF / T_eff) [Boltzmann factor]
- [ ] Integrate stop-loss: bond dissociation criterion |x| > x_max

**Deliverables:**
- [ ] `FreeEnergySignal` class with `.generate(physical_state)` → `Signal`
- [ ] Signal stream: timestamped entry/exit/stop events with physical justification attached
- [ ] Parameter sensitivity analysis: P&L vs z_threshold, x_max for top pairs

---

### Week 5 — Ergodicity Monitor & Detailed Balance Scorer

**Goal:** Implement the two diagnostic modules for regime detection and arbitrage identification.

**Tasks:**
- [ ] Implement `ErgodicityMonitor`
  - [ ] Rolling variance estimator for <x²>_time
  - [ ] Ergodicity ratio E = <x²>_time / (σ²/2γ)
  - [ ] Alert logic: E thresholds for position reduction and pair removal
  - [ ] State machine: `ERGODIC`, `WEAKLY_ERGODIC`, `ERGODICITY_BROKEN`
- [ ] Implement `DetailedBalanceScorer`
  - [ ] Discretize spread into bins: empirical transition matrix P(x_i → x_j)
  - [ ] Probability current: J(i→j) = π_i · P_ij - π_j · P_ji
  - [ ] Net current score: scalar summary of |J| across all transitions
  - [ ] Integration with signal: boost position size when J is high and aligned with signal direction
- [ ] Unit tests: known non-equilibrium process produces high J; pure OU produces J ≈ 0

**Deliverables:**
- [ ] `ErgodicityMonitor` with `.update(x)` → `ErgodicitState`
- [ ] `DetailedBalanceScorer` with `.score(spread_history)` → float ∈ [0,1]
- [ ] Backtest comparison: strategy with vs without these modules on 2023-2024 data

---

### Week 6 — Backtesting Engine

**Goal:** Full historical simulation with realistic execution modeling.

**Tasks:**
- [ ] Implement event-driven backtest engine (bar-by-bar, no look-ahead)
- [ ] Order types: market, limit, stop-limit
- [ ] Transaction cost model: commission + slippage (half-spread estimate)
- [ ] Portfolio accounting: positions, cash, margin, realized/unrealized P&L
- [ ] Handle corporate actions, stock splits, delistings (equities)
- [ ] Handle exchange downtime, API rate limits, funding rates (crypto)
- [ ] Performance metrics: Sharpe, Sortino, max drawdown, Calmar, hit rate, avg holding time, half-life vs actual holding time comparison
- [ ] Run full backtest on top 10 equity pairs + top 5 crypto pairs, 2 years of data

**Deliverables:**
- [ ] `BacktestEngine` with `.run(pair_configs, signals, start, end)` → `BacktestResult`
- [ ] Full performance report for all pairs
- [ ] Physical attribution report: what fraction of P&L came from free energy vs noise alpha

---

### Week 7 — Umbrella Sampling & Risk Module

**Goal:** Implement tail risk estimation from the physical distribution.

**Tasks:**
- [ ] Implement `UmbrellaSampler`
  - [ ] Define umbrella windows along spread coordinate
  - [ ] Biased simulation: add harmonic bias potential w_i(x - x_i)² to Langevin integrator
  - [ ] WHAM (Weighted Histogram Analysis Method) reweighting
  - [ ] Recover full free energy profile F(x) including tails
- [ ] Extract tail risk metrics:
  - [ ] P(|x| > threshold) for arbitrary threshold (better than Gaussian VaR)
  - [ ] Expected max drawdown under given horizon
  - [ ] CVaR at 95%, 99%, 99.9% confidence
- [ ] Portfolio-level: combine umbrella samples across all pairs for correlated tail risk

**Deliverables:**
- [ ] `UmbrellaSampler` with `.estimate_tail_distribution(pair)` → `TailRiskProfile`
- [ ] Comparison: Gaussian VaR vs umbrella sampling CVaR on historical stress events
- [ ] Risk report: full tail profile for each active pair

---

### Week 8 — Paper Trading Infrastructure

**Goal:** Connect the full signal pipeline to live market data and simulated execution.

**Tasks:**
- [ ] Implement `LiveDataFeed` class: Alpaca WebSocket (equities) + CCXT WebSocket (crypto)
- [ ] Implement real-time `PhysicalState` updater: streaming MLE re-estimation
- [ ] Order management system (OMS): track open orders, fills, positions in memory + DuckDB
- [ ] Alpaca paper account integration: submit, modify, cancel orders via REST API
- [ ] CCXT sandbox integration for crypto paper trading
- [ ] Reconciliation loop: compare OMS state with broker state every N seconds
- [ ] Alerting: email/Slack notification on ergodicity break, position stop, parameter change

**Deliverables:**
- [ ] Live system running on paper accounts for both equities and crypto
- [ ] Real-time dashboard (Streamlit): active pairs, physical states, P&L, ergodicity ratios
- [ ] Latency profiling: signal generation latency from bar close to order submission

---

### Week 9 — HPC Infrastructure & Parallel Tempering

**Goal:** Build the distributed parameter exploration layer using parallel tempering.

**Tasks:**
- [ ] Set up Ray cluster: local first, then AWS/GCP spot instance cluster
- [ ] Implement `ParallelTemperingOptimizer`
  - [ ] N replicas, each running Langevin engine at different "meta-temperatures" T_meta
  - [ ] T_meta controls exploration width in (γ, σ, β, window) parameter space
  - [ ] Metropolis exchange criterion for parameter swaps between adjacent replicas
  - [ ] Cold replica (T_meta → 0) = production parameters
- [ ] Implement `HPCSimulationManager`
  - [ ] Launch parameter sweep jobs across full universe
  - [ ] Aggregate results, rank by Sharpe / free energy alpha ratio
  - [ ] Feed best parameters back to paper trading engine
- [ ] Validate: compare parallel tempering vs random search vs grid search on held-out test period

**Deliverables:**
- [ ] Ray-based HPC cluster (local or cloud)
- [ ] `ParallelTemperingOptimizer` with demonstrated parameter quality improvement
- [ ] Parameter sweep results for full pair universe

---

### Week 10 — Integration, Testing & Hardening

**Goal:** Integrate all modules into a single coherent system, stress test everything.

**Tasks:**
- [ ] End-to-end integration test: data → physics → signals → backtest/paper/HPC, all in one run
- [ ] Failure mode testing:
  - [ ] Data feed interruption recovery
  - [ ] Exchange API downtime handling
  - [ ] Pair cointegration breakdown mid-trade
  - [ ] Extreme volatility event (simulate flash crash in backtest)
- [ ] Configuration management: YAML configs for all physical parameters, no hardcoded values
- [ ] Logging: structured JSON logs for every physical state update, signal, and order
- [ ] Documentation: docstrings for all classes, architecture README, physical model appendix

**Deliverables:**
- [ ] Full integration test suite: all modes (backtest, paper, HPC) pass end-to-end
- [ ] Runbook: how to launch each mode, what to monitor, how to respond to alerts
- [ ] Config file: `config.yaml` with all physical and execution parameters

---

### Week 11 — Performance Analysis & Physical Attribution

**Goal:** Deeply analyze backtest + paper trading results through the physical lens.

**Tasks:**
- [ ] Physical attribution analysis:
  - [ ] Separate P&L by source: free energy gradient, T_eff timing, detailed balance boost
  - [ ] Plot phase diagram: all active pairs in (γ, σ) space, color-coded by profitability
  - [ ] Time series of T_eff: show relationship to market VIX / crypto realized vol
- [ ] Regime analysis:
  - [ ] Identify ergodicity breaking events, compare to known market stress periods
  - [ ] Show detailed balance spikes preceding price corrections
- [ ] Parameter stability analysis: how stable are (γ, σ) estimates, how often do they change
- [ ] Overfitting analysis: walk-forward validation, out-of-sample performance

**Deliverables:**
- [ ] Physical attribution report (PDF/notebook)
- [ ] Phase diagram visualization for full pair universe
- [ ] Walk-forward performance vs in-sample performance comparison

---

### Week 12 — Production Readiness & Documentation

**Goal:** Make the system deployable and reproducible.

**Tasks:**
- [ ] Containerize: Dockerfiles for each component (data, physics, execution, HPC)
- [ ] Deployment: docker-compose for local, Kubernetes config for cloud
- [ ] CI/CD: GitHub Actions pipeline — lint, test, build on every commit
- [ ] Monitoring: Prometheus + Grafana dashboards for physical state metrics
- [ ] Final documentation:
  - [ ] System architecture document
  - [ ] Physical model derivations appendix
  - [ ] API reference for all modules
  - [ ] User guide: how to add new pairs, tune parameters, interpret physical signals
- [ ] Security: API key management, secret rotation, network isolation

**Deliverables:**
- [ ] Docker-compose deployment running all three modes simultaneously
- [ ] Complete documentation suite
- [ ] Production checklist: everything needed to run with real capital

---

## Appendix: Physical Model Reference Map

| Pipeline Step | Physical Model | F&S Chapter | Key Equation |
|---|---|---|---|
| Pair Screening | Effective pair potential, bonding criterion | Ch. 2, 4 | V(r) minimum → cointegration |
| Parameter Estimation | Langevin / OU MLE | Ch. 9 | dx = -γx dt + σ dW |
| Temperature | Fluctuation-Dissipation Theorem | Ch. 3 | T_eff = σ²/2γ |
| Signal Generation | Helmholtz Free Energy | Ch. 7, 8 | F(x) = ½γx² |
| Regime Detection | Ergodic sampling theory | Ch. 3, 13 | E = <x²>_t / (σ²/2γ) |
| Arbitrage Detection | Detailed Balance / NESS | Ch. 3 | J = π_i P_ij - π_j P_ji |
| HPC Optimization | Parallel Tempering | Ch. 13 | P(swap) = min(1, exp(-Δβ·ΔU)) |
| Tail Risk | Umbrella Sampling + WHAM | Ch. 7 | F(x) = -kT ln p(x) |
| Position Sizing | Boltzmann Factor | Ch. 3 | size ∝ exp(-ΔF/T_eff) |

---

*All physical models derived from: Frenkel, D. & Smit, B. "Understanding Molecular Simulation: From Algorithms to Applications." 2nd Ed. Academic Press, 2002.*
