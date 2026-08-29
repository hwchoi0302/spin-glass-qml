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
QAOA), and against those the claim is **circuit depth**.

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
4. **Bitpacking from 7×7 up.** 4×4 stays on string-keyed dicts.
5. `scripts/00_validate_small.py` must print `ALL 15 TESTS PASSED` before any
   result from a run is trusted.
