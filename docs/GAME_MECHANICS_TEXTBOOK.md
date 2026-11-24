# Territorial.io — Game Mechanics & Visual Grammar Textbook

> **Zone A Knowledge Document** — Extracted strictly via offline static inspection of [text.html](../text.html) (`research/string_table_decoder.py`, `research/mechanics_extractor.py`, `research/ui_palette_extractor.py`).
> The autonomous AI **never** accesses game memory, DOM state, or internal functions at runtime; it uses only external screenshots and OS mouse/keyboard input.

---

## 1. Core Simulation Timing & Economy Laws

| Parameter | Exact Value / Formula (from `text.html`) | Strategic Significance |
| :--- | :--- | :--- |
| **Tick Interval (`bf.aCD`)** | `56 ms` (`~17.86 ticks/sec`) | Sets reaction tempo and attack wave resolution speed. |
| **Interest Payout Cadence** | Every `10 ticks` (`~0.56 s`, when `tick % 10 == 9`) | Compound exponential growth engine; keeping reserves high early maximizes exponential compounding. |
| **Full Income Cycle (Land Income)** | Every `100 ticks` (`~5.60 s`, when `tick % 100 == 99`) | Adds `+1` troop per owned pixel (`floor(32 * land / 32)`). Visible on the Top-Center Balance Box's left/right vertical white bars (`ea = 9..0`). |
| **Opening Interest Decay** | `t < 1920` ticks (`~107.5 s`): decays linearly from **`7.00%`** (`700/10000`) down to **`1.00%`** (`100/10000`) | First ~100 seconds determine the entire match. Expanding efficiently *without* zeroing out troops exploits high compounding. |
| **Soft Troop Cap (`k2`)** | `min(100 * owned_pixels, 1,000,000,000)` | Reaching `100 × land` triggers **Red Interest**: interest rate drops linearly `e9 -= floor(2 * e9 * (troops - k1) / k1)`. |
| **Hard Troop Cap** | `150 * owned_pixels` | At `150 × land`, interest hits **`0.00%`**. Sitting in red interest forfeits free compounding. |

---

## 2. Expansion, Combat & Naval Mechanics

### 2.1 Neutral Territory Expansion (`gs()`)
- Advancing across `fx` contiguous neutral border pixels costs `fx * gL` troops per wave (flat low border tax, zero defender resistance).
- **Strategic Corollary**: Neutral expansion has massively higher land-per-troop yield than fighting players. Securing maximum neutral perimeter early while avoiding over-extension is paramount.

### 2.2 Player-vs-Player Combat (`gt()`, `gw()`, `h1()`)
For each border advancement wave contesting `fx` shared pixels against defender `fw`:
- **Base capture tax**: `gu = fx * gL`
- **Defender resistance**: `gv = floor(fx * defender_troops / (1 + floor(10 * defender_land / 16))) ≈ 1.6 * fx * (defender_troops / defender_land)`
- **Counter-attack clash**: `gx = active_attack(defender -> attacker)` (cancels 1-for-1!)
- **Total attacker cost**: `gz = gu + 2 * gv + gx`
- **Defender troop loss**: `gv` (reserve) + `gx` (counter-attack)

> **The 2:1 Defender Advantage Law**: Attacking a healthy defender incurs **twice the troop casualties** (`2 * gv + fx`) compared to what the defender loses (`gv`), plus attack launch taxes.
> Therefore, winning Territorial.io requires:
> 1. **High Density Defense**: Maintaining strong `troops / land` ratio deters neighbors and punishes attackers 2-to-1.
> 2. **Opportunistic Vulturing**: Attacking immediately when a neighbor suffers a sharp troop drop (`ΔT/Δt << 0`, e.g., after they full-send or get double-teamed).
> 3. **Counter-Attack Cancellation**: Sending a controlled defense/counter wave cancels incoming attack pressure 1-for-1 (`gx`).

### 2.3 Naval Boats & Interception
- Coastal territories can launch naval boats across water towards distant shores (`KeyB` or radial boat icon on water/coast).
- Enemy boats approaching owned coastline can be intercepted (`KeyI` or clicking near boat).
- Boats appear visually as distinct moving icon markers traversing blue water regions toward a landing pixel.

---

## 3. Native Bot Difficulty Personas (`c5` / `aE`)

| Difficulty | Passivity `kH` | Aggression `kF` | Boat/Mtn Thresh `kY` | Search Budget `kZ` | Action Tier `ka` | Boat Intercept `kc/1000` |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Very Easy** | 97 | 0 | 500 | 60 | 1 | 10.0% |
| **Easy** | 94 | 0 | 450 | 74 | 2 | 15.0% |
| **Normal** | 70 | 5 | 400 | 112 | 3 | 25.0% |
| **Hard** | 40 | 25 | 300 | 200 | 4 | 40.0% |
| **Very Hard** | 20 | 50 | 80 | 256 | 6 | 60.0% |
| **Impossible** | 0 | 100 | 50 | 512 | 8 | 100.0% |

---

## 4. Visual Interface & Scale-Invariant Layout Grammar

Territorial.io renders everything onto `<canvas id="canvasA">` scaled relative to `min(W, H)` (`h.iB`). All CV detectors MUST localize elements dynamically without hardcoded coordinates:

1. **Main Menu (`cP`)**:
   - Centered play bar around `y ≈ 0.54 * H`.
   - Left (`60%` width): `"Multiplayer"` green button `rgba(22, 88, 22, 0.8)`.
   - Right (`40%` width): `"Single Player"` teal button `rgba(22, 88, 88, 0.8)`.
2. **Top-Center Troop Balance Box (`cL`)**:
   - Width `≈ 0.24 * min(W, H)`, height `≈ width / 6`, anchored top-center.
   - Green background `rgba(0, 100, 0, 0.8)` when gaining troops; red `rgba(150, 0, 0, 0.8)` when losing troops.
   - **Left & Right white vertical bars**: drain across the 10-sub-cycle (`100 ticks`) land-income period; `+income` flashes yellow `rgb(255, 255, 100)` on cycle completion.
   - **Red-Interest Visual Cue**: Balance text switches from white `rgb(255,255,255)` to pink/red `rgb(255,160,160)` when `troops >= 100 * land`.
3. **Top-Right Stats Panel (`cN`)**:
   - Semi-transparent dark box `rgba(0,0,0,0.75)` with `1 px` white border and bright green progress bar `rgba(0,255,0,0.6)` along its bottom edge.
   - Displays Bots/Players, Interest `%` (turns coral-red `rgb(255,120,100)` during red interest), Land count, and Elapsed Time `M:SS`.
4. **Bottom-Center Troop Allocation Slider (`cI`)**:
   - Width `≈ 0.389 * min(W, H)`, height `= width / 12`, centered near bottom margin.
   - Left `[-]` cap (`1.5 × height`), Right `[+]` cap (`1.5 × height`), central track (`ib ∈ [1/1024, 1.0]`).
   - Color gradient encodes allocation percentage: **Green (`0%`) → Yellow (`33%`) → Red (`66%`) → Magenta (`100%`)**.
   - Renders `"{troops} ({pct}%)"` directly centered inside the bar.
5. **Top-Left Leaderboard (`cM`)**:
   - Dark semi-transparent ranking box listing leading kingdoms, icons/crowns, and territory shares.
6. **Territory & Borders**:
   - Every owned territory renders a crisp darkened inner border (`2–4 px` perimeter) separating it cleanly from adjacent players even when fills share similar hues.
   - Player names and troop numbers (`850`, `14.2K`, `1.35M`) render dynamically centered inside the largest inscribed rectangle of each territory (`cb` / `aL8`).

---

## 5. Legitimate Human Input Controls (`d4`, `dg`)

| Control | Physical Input | Effect |
| :--- | :--- | :--- |
| **Spawn / Attack / Expand** | Left-click on map pixel (or `Space` at mouse cursor) | Places spawn during spawn phase; initiates land attack/expansion on target territory at current slider `%`. |
| **Adjust Troop Slider** | Click/drag bottom slider track, click `-`/`+`, mouse wheel on slider, or `W`/`S` (`±1/8`), `D`/`A` (`±1/32`) | Sets troop commitment percentage `ib`. |
| **Launch Naval Boat** | `KeyB` at target water/coast cursor (or radial menu boat button) | Dispatches ship from nearest owned coast toward cursor. |
| **Intercept Boat** | `KeyI` at incoming boat cursor | Launches defensive interceptor ship. |
| **Camera Zoom & Pan** | Mouse wheel on map, click-drag map, or `NumpadAdd`/`NumpadSubtract` & `Arrow` keys | Zooms centered on cursor / pans viewport. |
| **Peace Vote** | `KeyP` (or menu button) | Proposes/accepts peace when winning or stalemated. |
| **Reinforce Weakest Attack** | `KeyM` | Adds slider troops to currently weakest active attack. |
