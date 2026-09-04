# spin-glass-qml

2D Edwards–Anderson spin glass, `H = -Σ_<ij> J_ij Z_i Z_j - h Σ_i X_i`, open
boundaries, square lattice. A parametrized circuit is trained **classically**
with Backpropagating Pauli Propagation (BP-PPS, arXiv:2607.15184) and then run
on quantum hardware (IBM Nighthawk). Target scale: 10×10 = 100 qubits.

## The three goals

| # | Claim | Status |
|:--|:--|:--|
| 1 | Time evolution from an arbitrary product state, more accurate per unit depth than Trotter | primary |
| 2 | Shallow-circuit sampling of the evolved state | primary |
| 3 | Ground-state preparation **in a quantum register** | in parallel (see `docs/issues/02-comparison-models.md`) |

Goal 3's claim is a **capability** one, not a complexity one: the deliverable is
the state itself, held in the register and available as the input to further
quantum computation. QMC returns classical bitstrings drawn from
`|psi_0(x)|^2` and an energy; it never hands you `|psi_GS>`, and there is no
"QMC circuit". So QMC is goal 3's **reference value**, not its competitor — it
supplies the energy the device is checked against. The competitors are the
other ways to get that state into a register (Trotterised adiabatic, QITE,
VQE), and against those the claim is **circuit depth**.

**Naming (owner's decision, 2026-09-03).** The 2-angles-per-layer competitor is
called **VQE**, never QAOA: it has QAOA's circuit shape but minimises the energy
of the transverse-field `H`, whose ground state is entangled, where textbook
QAOA minimises a diagonal cost whose ground state is a bitstring. The key in
`results/4x4/gs_competitors.json` is still `qaoa` — data files are not edited
after the fact — so **key `qaoa` == the VQE row**. Our own ansatz keeps the name
**HVA** even though its per-bond parameterisation is the literature's
multi-angle QAOA; the same name is already used on the time-evolution side and
splitting it per goal would confuse more than it fixes.

This is deliberately *not* a classical-intractability claim. The model is
stoquastic for every `J`, so QMC samples the ground state without a sign
problem. Frustration and the sign problem are different things here. Never
write "sign problem" into a document about this Hamiltonian without checking
`docs/issues/02-comparison-models.md` first, and never upgrade goal 3 into a
"classically impossible" claim.

The ground-state circuit starts from `|+...+>`, not `|0...0>` — see
`configs/optimizer.yaml` (`optimizer.ground_state.initial_state`) for why the
parity of `Pi_i X_i` makes `|0...0>` cap the fidelity at 0.5.

## Working on this repo

Issues are split into topic files under **`docs/issues/`**. Read the index
(`docs/issues/README.md`) and then only the one file for the topic at hand —
each file carries its own state, decisions already made, and open questions.
Update that file at the end of a session so the next one starts warm.

- `docs/RUNBOOK.md` — how to actually run T1 and T1-T (written for the desktop).
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
