# spin-glass-qml

2D Edwards–Anderson spin glass, `H = -Σ_<ij> J_ij Z_i Z_j - h Σ_i X_i`, open
boundaries, square lattice. A parametrized circuit is trained **classically**
with Backpropagating Pauli Propagation (BP-PPS, arXiv:2607.15184) and then run
on quantum hardware (IBM Nighthawk). Target scale: 10×10 = 100 qubits.

## The goal (owner's decision, 2026-09-04)

> **A bond-parameterised HVA, trained classically with BP-PPS, prepares the
> ground state in a quantum register using a SHALLOWER circuit than the
> state-of-the-art quantum ground-state-preparation algorithms.**

One claim, measured in **2Q gate count at matched accuracy**. Everything else in
this repo is either evidence for it, a competitor to beat, or parked.

**This replaces the "three goals" structure.** Time evolution (old goals 1 and
2) is **parked**, not cancelled — its results stand and are reusable as a second
instance of the same mechanism, but no new work goes there until the claim above
is settled. See `docs/issues/01-scale-plan.md` and `docs/benchmark_plan.md`.

### What "state of the art" means here — this is the whole difficulty

The competitors measured so far (linear-schedule adiabatic Trotter, single-angle
VQE) are **honest textbook baselines, not SOTA**. The 15.0× / 1.8× numbers in
`docs/status_4x4.md` were measured against those. **The new goal is not
established until the SOTA versions are measured**, and two of them are known
threats:

| Family | SOTA version to beat | Why it threatens us |
|:--|:--|:--|
| Adiabatic | **Counterdiabatic (CD) / DCQO, BF-DCQO** | The leading CD term for TFIM is `Σ_i α_i Y_i` — a **1-qubit** rotation, so it costs **zero** in our 2Q currency. CD adiabatic gets strictly better at the same 2Q count. |
| Variational | **ADAPT-VQE / AVQITE** | AVQITE reports CNOT count linear in `N` with a coefficient **half** of HVA's on finite-field TFIM — aimed straight at our ansatz. |
| Imaginary time | QITE | |
| Single-angle VQE | — | Keep as a baseline, but **label it as a baseline**, never as SOTA. |

Do not write "N× shallower than competitors" in any final text before those
rows exist. `docs/issues/02-comparison-models.md` owns this table.

**Fairness rules for the comparison.** Same `H`, same qubit connectivity, same
4-colouring, same accuracy target (`dE` or `F`), cost reported in 2Q gates.
ADAPT/AVQITE normally choose their operator pool from hardware measurements; at
4×4 we simulate everything classically anyway, so **run them classically and
report the number — do not exclude them on a definitional technicality.**

## What we may and may not claim

Three sentences are true and usable:

1. **The circuit is shallower** than the alternatives that put the same state in
   a register — the claim above.
2. **The state is in the register**, available as the input to further quantum
   computation. This is a *capability* claim. QMC returns bitstrings from
   `|psi_0(x)|^2` and an energy; it never hands you `|psi_GS>`. Write it
   narrowly: **"QMC alone does not give you the register state."**
3. **QMC fails on the model** — only if we switch to a non-stoquastic model
   (see below), and only once the average-sign collapse is *measured*.

### "Classically impossible" is unavailable — in any form

The circuit is produced by BP-PPS, a classical simulator. To optimise `θ` the
trainer must evaluate `E(θ) = <+|U(θ)† H U(θ)|+>` on a **classical CPU**,
thousands of times. If that succeeds, a classical algorithm has already computed
both the energy and a full description of the state — before any quantum
computer is involved.

The root cause is shallowness, and it is not escapable by changing the model:

```
deep enough to be classically hard  ->  too deep for the noisy device,
                                        and BP-PPS cannot train it either
shallow enough to run on the device ->  classically describable by the very
                                        method that trained it
```

**The shallowness hardware requires and the shallowness classical simulation
requires are the same condition.** So: never write "classically impossible",
and never try to buy it by changing the Hamiltonian. "QMC fails" is a true and
narrower statement — QMC failing is not all classical methods failing, since
BP-PPS is a classical method that does not fail.

(Caveat kept for completeness: PPS yields *expectation values*, not samples;
sampling shallow 2D circuits is still believed classically hard. That does not
help this goal, whose deliverable is the energy and the state.)

### Frustration is not the sign problem

The current `H` is **stoquastic for every `J`** — `J` sits on the diagonal and
every off-diagonal element is `-h < 0`. Frustration is always present; a sign
problem never is. Never write "sign problem" about this Hamiltonian without
reading `docs/issues/02-comparison-models.md` first.

### The model may change — decision is deferred, on purpose

A non-stoquastic variant (**model A**: same square lattice, each bond carries one
of `XX`/`YY`/`ZZ` with Gaussian random `J_ij`, plus a transverse field) would
kill QMC and supply the motivation for *why anyone needs a quantum circuit for
this state at all*. Its cost is that QMC stops being the 100-qubit reference
value, leaving DMRG alone.

**Sequencing decision: settle the depth claim on the current model first.** The
depth claim is model-independent, and the SOTA competitor work (above) is where
it can die. Porting to a new model before that risks wasting the port. The full
model-A analysis, the sign-problem/frustration check, and the staged plan are in
`docs/issues/06-novelty.md` §10.

If the model does change, note that `|+...+>` loses its justification — the
`Pi_i X_i` parity argument that makes `|0...0>` cap fidelity at 0.5 (see
`configs/optimizer.yaml`, `optimizer.ground_state.initial_state`) does not
survive `XX`/`YY` bonds, and the initial state must be re-chosen.

**Naming (owner's decision, 2026-09-03).** The 2-angles-per-layer competitor is
called **VQE**, never QAOA: it has QAOA's circuit shape but minimises the energy
of the transverse-field `H`, whose ground state is entangled, where textbook
QAOA minimises a diagonal cost whose ground state is a bitstring. The key in
`results/4x4/gs_competitors.json` is still `qaoa` — data files are not edited
after the fact — so **key `qaoa` == the VQE row**. Our own ansatz keeps the name
**HVA** even though its per-bond parameterisation is the literature's
multi-angle QAOA (Herrman et al. 2022); say so in Related Work rather than
letting a reviewer find it.

## Working on this repo

Issues are split into topic files under **`docs/issues/`**. Read the index
(`docs/issues/README.md`) and then only the one file for the topic at hand —
each file carries its own state, decisions already made, and open questions.
Update that file at the end of a session so the next one starts warm.

- `docs/RUNBOOK.md` — how to actually run things (written for the desktop).
- `docs/benchmark_plan.md` — the overall thesis and tier structure.
- `docs/manual.md` — code and config reference.

## Hard rules

1. **Two machines.** Code is written on the laptop; simulations run on the
   desktop (AMD Ryzen 5 5600, RTX 2060). Do not launch a full-size run from the notebook session.
2. **Python owns the couplings.** `scripts/00_build_model.py` is the only place
   `J` is generated. Julia loads `model_config.json` and validates the bond
   order; it must never call its own `build_bonds` to regenerate them.
3. **Gate ordering.** Sequences are stored in circuit order, `U = g_{T-1}···g_0`.
   `propagate_forward` walks them in **reverse** (observable at the output,
   pushed toward the input); `propagate_backward` walks them forward. This was
   a real bug once — see `e9c3b50`.
4. **The string-dict engine is the oracle, not the default.**
   `src/bppps/propagation.py` is what every other engine is checked against;
   it is never chosen for speed. Any engine may be used at any lattice size
   **once it has been proven term-for-term equal to the oracle at 4×4**
   (TESTs 17–22). Bitpacking is *required* from 7×7 up, because the string
   representation does not fit; below that it is simply allowed, and 4×4
   production now trains on the sorted-array engine
   (`truncation.engine: sorted`) as well as sweeping with the numba one.
   Do not reintroduce a rule that pins an engine to a lattice size.
   **The sorted key packs x into the low 32 bits and z into the high 32, so
   it holds at most 32 qubits** — 5×5 fits, 7×7 does not and needs the key
   widened first. **The numba engine shares that key and that ceiling.**
   Above 32 qubits the packing is not injective (`Z_0` and `X_32` both give
   `2**32`) and both engines write `d[key] = value`, so one term silently
   overwrites the other. This is not hypothetical: it invalidated the `L=6`
   row of `results/lightcone_production_delta.json`, verified by re-running
   that propagation on the string engine and finding support on qubit 33.
   All three packed entry points now call
   `propagation_packed.check_gate_sequence_packable`, which reads the
   circuit's own qubit indices — the only quantity available before the
   `@njit` loop starts building keys with raw bit ops that never reach
   `make_key`. TEST 22 is the regression test.
   Target generation runs on this engine too now
   (`target.engine` / `time_sweep.engine`, default `sorted`).
   ⚠️ **A second, independent guard exists uncommitted on the laptop**
   (`assert_key_width` / `pack_key` / `to_sorted_arrays`, with its own
   `test_22_...` of the same number). It is not on `main`. The two are
   complementary — one checks the circuit before the `@njit` loop, the other
   checks key construction — but they must be reconciled into one TEST 22
   before either is quoted. See the merge commit that recorded this.
5. **Choose the engine by measurement, and re-measure on an idle machine.**
   GPU is *not* the default anywhere, but the reason is now a crossover, not
   a verdict. Measured 2026-09-03 on an idle desktop
   (`results/4x4/gpu_benchmark.json`): cupy loses to numpy at 30K terms
   (0.23×), wins at 100K (2.0×), 300K (12.3×) and 1.2M (17.3×). The earlier
   flat "GPU loses" reading came from a run that was aborted at 42 minutes
   under CPU contention. **Training is on the losing side of that crossover**
   — its gradient evaluations sit near 400K terms where sorted-numpy 0.50 s
   ties sorted-cupy 0.53 s — which is why the CPU engine is what training
   uses. Target generation at 1.2M terms is on the winning side and is the
   open GPU question. The comparison models' statevector is also measured
   now: 3.5–6.6× for the HVA/VQE gradient, 5.4–5.8× for the batched
   adiabatic scan. **Read those propagation multipliers as an upper bound:**
   they come from applying one gate repeatedly to a fixed-size array, and on
   the real target-generation walk — where the term count moves every gate and
   truncation intervenes — the same ~300K-term point measures 3.6×, not 12.3×.
   Full reasoning: `docs/issues/03-engine-performance.md`.
6. `scripts/00_validate_small.py` must print `ALL 22 TESTS PASSED` before any
   result from a run is trusted. TEST 20 skipping (no CUDA device) is a pass.
