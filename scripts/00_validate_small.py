"""Phase 0-3 integration test (OOM-safe version).

모든 검증을 sparse 연산으로 수행합니다.
SparsePauliOp 검증도 dense 변환 없이 sparse matrix 비교로 진행합니다.
"""
import sys
import os
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

import numpy as np
from hamiltonians import SpinGlass2D, frustration_ratio
from ansatz import HVA, TrotterCircuit
from classical_bench import ExactDiag


def test_1_ferromagnetic():
    """2x2 강자성 바닥 에너지 검증."""
    print("=" * 60)
    print("TEST 1: 2x2 lattice, all J=+1, h=0 (ferromagnetic)")
    print("=" * 60)

    model = SpinGlass2D(Lx=2, Ly=2, h=0.0, coupling_type='ea_bimodal', seed=0)
    model.J = np.ones(model.num_bonds)

    H = model.build_sparse_matrix()
    ed = ExactDiag(H, model.num_qubits)

    E0 = ed.ground_energy()
    print(f"  Ground energy: {E0:.6f} (expected: -4.0)")
    assert abs(E0 - (-4.0)) < 1e-10

    energies, states = ed.ground_state(k=2)
    psi_gs = states[:, 0]
    probs = ed.bitstring_distribution(psi_gs)
    assert probs[0] + probs[15] > 0.99
    print("  ✅ PASSED\n")


def test_2_pauli_op_consistency():
    """SparsePauliOp ↔ sparse matrix 일치 검증 (sparse 비교, dense 변환 없음)."""
    print("=" * 60)
    print("TEST 2: SparsePauliOp vs sparse matrix (4x4, sparse 비교)")
    print("=" * 60)

    model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
    H_sparse = model.build_sparse_matrix()
    pauli_op = model.get_pauli_terms()

    # SparsePauliOp → sparse matrix (to_matrix는 dense라 위험!)
    # 대신: 랜덤 상태벡터로 H|ψ⟩와 pauli_op|ψ⟩ 비교
    rng = np.random.default_rng(123)
    for trial in range(5):
        psi = rng.normal(size=2**16) + 1j * rng.normal(size=2**16)
        psi /= np.linalg.norm(psi)

        H_psi = H_sparse @ psi
        # SparsePauliOp의 행렬-벡터 곱은 evolve 또는 직접 연산
        # Qiskit SparsePauliOp에서 sparse matrix 추출
        pauli_sparse = pauli_op.to_matrix(sparse=True)
        P_psi = pauli_sparse @ psi

        diff = np.max(np.abs(H_psi - P_psi))
        assert diff < 1e-12, f"Trial {trial}: diff = {diff}"

    print(f"  5회 랜덤 상태 검증 완료 (max diff < 1e-12)")
    print("  ✅ PASSED\n")


def test_3_energy_consistency():
    """4x4 EA bimodal 에너지 일관성."""
    print("=" * 60)
    print("TEST 3: 4x4 lattice, energy consistency")
    print("=" * 60)

    model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
    print(f"  Qubits: {model.num_qubits}, Bonds: {model.num_bonds}")

    H = model.build_sparse_matrix()
    ed = ExactDiag(H, model.num_qubits)

    E0 = ed.ground_energy()
    energies, states = ed.ground_state(k=1)
    psi_gs = states[:, 0]
    E_computed = ed.compute_energy(psi_gs, model.bonds, model.J, model.h)

    print(f"  Ground energy (eigsh):         {E0:.6f}")
    print(f"  Energy (compute_energy):       {E_computed:.6f}")
    assert abs(E0 - E_computed) < 1e-8
    print("  ✅ PASSED\n")


def test_4_frustration():
    """Villain, EA, Gaussian frustration 비율."""
    print("=" * 60)
    print("TEST 4: Frustration ratios")
    print("=" * 60)

    for coupling, expected_label in [
        ('villain', '100%'),
        ('ea_bimodal', 'random'),
        ('gaussian', 'random'),
    ]:
        model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type=coupling, seed=42)
        fr = frustration_ratio(model.J, model.bonds, 4, 4)
        print(f"  {coupling:12s}: frustration = {fr:.4f} ({expected_label})")
        if coupling == 'villain':
            assert abs(fr - 1.0) < 1e-10, "Villain must be 100% frustrated"

    print("  ✅ PASSED\n")


def test_5_time_evolution():
    """시간 진화 norm 보존 및 관측량."""
    print("=" * 60)
    print("TEST 5: Time evolution (4x4)")
    print("=" * 60)

    model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
    H = model.build_sparse_matrix()
    ed = ExactDiag(H, model.num_qubits)

    psi0 = np.zeros(2**16)
    psi0[0] = 1.0

    for t in [0.1, 0.5, 1.0]:
        psi_t = ed.time_evolve(psi0, t)
        norm = np.linalg.norm(psi_t)
        assert abs(norm - 1.0) < 1e-10, f"Norm not preserved at t={t}"

        obs = ed.local_observables(psi_t, model.bonds)
        E = ed.compute_energy(psi_t, model.bonds, model.J, model.h)
        print(f"  t={t:.1f}: norm={norm:.10f}, E={E:.6f}, "
              f"<X_0>={obs['X'][0]:.4f}, <Z_0>={obs['Z'][0]:.4f}")

    print("  ✅ PASSED\n")


def test_6_hva():
    """HVA 회로 구성."""
    print("=" * 60)
    print("TEST 6: HVA circuit (4x4, 3 layers)")
    print("=" * 60)

    model = SpinGlass2D(Lx=4, Ly=4, h=1.0, coupling_type='ea_bimodal', seed=42)
    hva = HVA(
        num_qubits=model.num_qubits,
        bonds=model.bonds,
        n_layers=3, Lx=4, Ly=4,
    )

    assert hva.count_params() == 3 * (16 + 24), f"params: {hva.count_params()}"
    assert hva.circuit_depth() == 15, f"depth: {hva.circuit_depth()}"
    assert hva.count_2q_gates() == 72, f"2q gates: {hva.count_2q_gates()}"

    params = np.random.default_rng(42).uniform(-np.pi, np.pi, hva.count_params())
    qc = hva.build_circuit(params)
    print(f"  Params: {hva.count_params()}, Depth: {hva.circuit_depth()}, "
          f"2Q gates: {hva.count_2q_gates()}")
    print(f"  Circuit: {qc.num_qubits} qubits, {len(qc.data)} gates")
    print("  ✅ PASSED\n")


def test_7_trotter_vs_ed():
    """Trotter vs ED fidelity (2x2 — statevector 비교 가능)."""
    print("=" * 60)
    print("TEST 7: Trotter vs ED fidelity (2x2)")
    print("=" * 60)

    model = SpinGlass2D(Lx=2, Ly=2, h=1.0, coupling_type='ea_bimodal', seed=42)
    H = model.build_sparse_matrix()
    ed = ExactDiag(H, model.num_qubits)

    trotter = TrotterCircuit(
        hamiltonian_op=model.get_pauli_terms(),
        num_qubits=model.num_qubits,
    )

    psi0 = np.zeros(2**4)
    psi0[0] = 1.0

    from qiskit.quantum_info import Statevector
    sv_init = Statevector.from_label('0' * model.num_qubits)

    for t, dt in [(0.3, 0.05), (0.5, 0.1), (1.0, 0.1)]:
        qc = trotter.build_circuit(t, dt, order=2)
        psi_trotter = np.array(sv_init.evolve(qc))
        psi_exact = ed.time_evolve(psi0, t)
        fid = ed.state_fidelity(psi_trotter, psi_exact)
        depth = trotter.circuit_depth(t, dt, order=2)
        print(f"  t={t}, dt={dt}: fidelity={fid:.8f}, depth={depth}")
        assert fid > 0.999, f"Fidelity too low: {fid}"

    print("  ✅ PASSED\n")


def test_8_trotter_dt_comparison():
    """Trotter dt=0.1 vs dt=0.2 비교."""
    print("=" * 60)
    print("TEST 8: Trotter dt=0.1 vs dt=0.2 (2x2)")
    print("=" * 60)

    model = SpinGlass2D(Lx=2, Ly=2, h=1.0, coupling_type='ea_bimodal', seed=42)
    H = model.build_sparse_matrix()
    ed = ExactDiag(H, model.num_qubits)

    trotter = TrotterCircuit(
        hamiltonian_op=model.get_pauli_terms(),
        num_qubits=model.num_qubits,
    )

    psi0 = np.zeros(2**4)
    psi0[0] = 1.0

    from qiskit.quantum_info import Statevector
    sv_init = Statevector.from_label('0' * model.num_qubits)

    print(f"  {'t':>4s} | {'dt=0.1 fidelity':>16s} {'depth':>6s} | "
          f"{'dt=0.2 fidelity':>16s} {'depth':>6s}")
    print(f"  {'-'*4}-+-{'-'*16}-{'-'*6}-+-{'-'*16}-{'-'*6}")

    for t in [0.5, 1.0, 1.5]:
        psi_exact = ed.time_evolve(psi0, t)
        row = f"  {t:4.1f} |"
        for dt in [0.1, 0.2]:
            qc = trotter.build_circuit(t, dt, order=2)
            psi_t = np.array(sv_init.evolve(qc))
            fid = ed.state_fidelity(psi_t, psi_exact)
            depth = trotter.circuit_depth(t, dt, order=2)
            row += f" {fid:16.6f} {depth:6d} |"
        print(row)

    print("  ✅ PASSED\n")


def test_9_memory_safety():
    """메모리 안전장치 동작 확인."""
    print("=" * 60)
    print("TEST 9: Memory safety checks")
    print("=" * 60)

    # 100큐비트 SpinGlass2D 생성은 OK (파울리 항만 저장)
    model100 = SpinGlass2D(Lx=10, Ly=10, h=1.0, coupling_type='ea_bimodal', seed=42)
    print(f"  10x10 모델 생성: OK (qubits={model100.num_qubits}, bonds={model100.num_bonds})")

    # get_pauli_terms()는 100큐비트에서도 안전 (sparse 표현)
    pauli_op = model100.get_pauli_terms()
    print(f"  SparsePauliOp 생성: OK ({len(pauli_op)} terms)")

    # build_sparse_matrix()는 차단되어야 함
    try:
        model100.build_sparse_matrix()
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  build_sparse_matrix() 차단: ✅ ({e})")

    # ExactDiag는 MAX_QUBITS_ED(=26)까지 허용, 그 위는 차단
    import scipy.sparse as sp
    import warnings as _warnings
    from classical_bench import MAX_QUBITS_ED, statevector_gb

    dummy_H = sp.eye(4, format='csr')
    with _warnings.catch_warnings():
        _warnings.simplefilter('ignore')
        ExactDiag(dummy_H, num_qubits=MAX_QUBITS_ED)
    print(f"  ExactDiag({MAX_QUBITS_ED}q) 허용: ✅ "
          f"(상태벡터 {statevector_gb(MAX_QUBITS_ED):.2f} GiB)")

    try:
        ExactDiag(dummy_H, num_qubits=MAX_QUBITS_ED + 1)
        assert False, "Should have raised ValueError"
    except ValueError:
        print(f"  ExactDiag({MAX_QUBITS_ED + 1}q) 차단: ✅")

    # 22큐비트 초과에서는 matrix-free 연산자가 sparse 행렬과 일치해야 한다
    model = SpinGlass2D(Lx=3, Ly=3, h=1.0, coupling_type='ea_bimodal', seed=7)
    H_sparse = model.build_sparse_matrix()
    H_free = model.build_linear_operator()
    rng = np.random.default_rng(0)
    v = rng.standard_normal(2 ** model.num_qubits)
    dev = np.max(np.abs(H_free @ v - H_sparse @ v))
    print(f"  matrix-free H == sparse H: max dev = {dev:.3e}")
    assert dev < 1e-10, f"matrix-free Hamiltonian disagrees: {dev}"

    print("  ✅ PASSED\n")


def _pauli_matrix(label):
    """Dense matrix of a Pauli label, with label[k] acting on qubit k.

    Qubit 0 is the least significant factor, matching Qiskit's ordering.
    """
    import numpy as np
    single = {
        'I': np.eye(2, dtype=complex),
        'X': np.array([[0, 1], [1, 0]], dtype=complex),
        'Y': np.array([[0, -1j], [1j, 0]], dtype=complex),
        'Z': np.diag([1, -1]).astype(complex),
    }
    M = np.array([[1]], dtype=complex)
    for k in range(len(label) - 1, -1, -1):
        M = np.kron(M, single[label[k]])
    return M


def test_10_spd_matches_circuit():
    """SPD forward pass가 실제 Qiskit 회로의 U^dag O U 와 일치하는지 검증.

    게이트 순서 관례(circuit order로 저장, Heisenberg는 역순 순회)가
    깨지면 이 테스트가 즉시 실패한다.
    """
    from qiskit.quantum_info import Operator
    from bppps.propagation import build_hva_gate_sequence, propagate_forward
    from bppps.pauli_utils import make_observable_label

    print("=" * 60)
    print("TEST 10: SPD forward == U^dag O U (2x2, 2 layers)")
    print("=" * 60)

    model = SpinGlass2D(Lx=2, Ly=2, h=1.0, coupling_type='ea_bimodal', seed=7)
    N = model.num_qubits
    hva = HVA(num_qubits=N, bonds=model.bonds, n_layers=2, Lx=2, Ly=2, J=model.J)
    theta = np.random.default_rng(0).uniform(-1.0, 1.0, hva.count_params())

    U = Operator(hva.build_circuit(theta)).data
    gate_seq = build_hva_gate_sequence(N, model.bonds, hva.substep_bonds, 2, theta)

    for pauli, qubit in (('X', 1), ('Z', 0)):
        label = make_observable_label(N, pauli, qubit)
        spo = propagate_forward({label: 1.0}, gate_seq, 0.0)

        lhs = np.zeros((2**N, 2**N), dtype=complex)
        for P, a in spo.items():
            lhs += a * _pauli_matrix(P)
        rhs = U.conj().T @ _pauli_matrix(label) @ U

        err = np.linalg.norm(lhs - rhs)
        print(f"  {pauli}_{qubit}: ||SPD - U^dag O U||_F = {err:.3e} "
              f"({len(spo)} terms)")
        assert err < 1e-10, f"{pauli}_{qubit}: SPD/circuit mismatch, err={err}"

    print("  ✅ PASSED\n")


def test_11_bppps_gradient():
    """BP-PPS backward pass (Eq. 20-21) vs 중앙차분 검증."""
    from bppps.propagation import (
        build_hva_gate_sequence, propagate_forward, propagate_backward,
    )
    from bppps.pauli_utils import make_observable_label, is_iz_only

    print("=" * 60)
    print("TEST 11: BP-PPS gradient vs central difference (2x2, 2 layers)")
    print("=" * 60)

    model = SpinGlass2D(Lx=2, Ly=2, h=1.0, coupling_type='ea_bimodal', seed=7)
    N = model.num_qubits
    hva = HVA(num_qubits=N, bonds=model.bonds, n_layers=2, Lx=2, Ly=2, J=model.J)
    n_params = hva.count_params()
    rng = np.random.default_rng(0)
    theta = rng.uniform(-1.0, 1.0, n_params)

    # Hamiltonian SPO (ground-state mode)
    ham = {}
    for idx, (i, j) in enumerate(model.bonds):
        chars = ['I'] * N
        chars[i] = 'Z'
        chars[j] = 'Z'
        key = ''.join(chars)
        ham[key] = ham.get(key, 0.0) - model.J[idx]
    for q in range(N):
        key = make_observable_label(N, 'X', q)
        ham[key] = ham.get(key, 0.0) - model.h

    def energy_and_grad(th):
        seq = build_hva_gate_sequence(N, model.bonds, hva.substep_bonds, 2, th)
        evolved = propagate_forward(ham, seq, 0.0)
        E = sum(a for P, a in evolved.items() if is_iz_only(P))
        seed = {P: 1.0 for P in evolved if is_iz_only(P)}
        return E, propagate_backward(evolved, seed, seq, len(th), 0.0)

    E, grad = energy_and_grad(theta)

    eps = 1e-6
    grad_fd = np.zeros(n_params)
    for i in range(n_params):
        tp, tm = theta.copy(), theta.copy()
        tp[i] += eps
        tm[i] -= eps
        grad_fd[i] = (energy_and_grad(tp)[0] - energy_and_grad(tm)[0]) / (2 * eps)

    rel = np.linalg.norm(grad - grad_fd) / np.linalg.norm(grad_fd)
    print(f"  E = {E:.8f}, |grad| = {np.linalg.norm(grad):.6f}")
    print(f"  relative L2 error (BP-PPS vs CD) = {rel:.3e}")
    assert rel < 1e-6, f"gradient mismatch: rel err = {rel}"

    # 에너지가 실제로 배포될 상태 U|0>의 에너지인지 확인
    from qiskit.quantum_info import Statevector
    psi = np.array(Statevector.from_label('0' * N).evolve(hva.build_circuit(theta)))
    H = model.build_sparse_matrix()
    E_sv = float(np.real(np.conj(psi) @ (H @ psi)))
    print(f"  E_SPD = {E:.8f} vs E_statevector[U|0>] = {E_sv:.8f}")
    assert abs(E - E_sv) < 1e-10, (
        f"SPD energy is not the energy of U|0>: {E} vs {E_sv}"
    )

    print("  ✅ PASSED\n")

def test_12_truncation_error_estimate():
    """BP-PPS Appendix B (Eq. B16) 오차 추정이 실제 오차를 상한하는지 검증.

    논문 Fig. 8 의 축소판: 회로를 고정하고 절단 임계값만 바꿔가며
    (i) delta=0 대비 실제 에너지 오차와 (ii) 버려진 l2 무게의 quadrature 합을
    비교한다. 추정치가 실제 오차 아래로 내려가면 진단 도구로 못 쓴다.
    """
    from bppps.propagation import (
        TruncationStats, build_hva_gate_sequence, propagate_forward,
    )
    from bppps.pauli_utils import is_iz_only, make_observable_label

    print("=" * 60)
    print("TEST 12: truncation error estimate bounds the observed error")
    print("=" * 60)

    model = SpinGlass2D(Lx=3, Ly=3, h=1.0, coupling_type='ea_bimodal', seed=11)
    N = model.num_qubits
    hva = HVA(num_qubits=N, bonds=model.bonds, n_layers=4, Lx=3, Ly=3, J=model.J)
    theta = np.random.default_rng(1).uniform(-0.8, 0.8, hva.count_params())
    gate_seq = build_hva_gate_sequence(N, model.bonds, hva.substep_bonds, 4, theta)

    ham = {}
    for idx, (i, j) in enumerate(model.bonds):
        chars = ['I'] * N
        chars[i] = 'Z'
        chars[j] = 'Z'
        key = ''.join(chars)
        ham[key] = ham.get(key, 0.0) - model.J[idx]
    for q in range(N):
        key = make_observable_label(N, 'X', q)
        ham[key] = ham.get(key, 0.0) - model.h

    def energy(delta):
        stats = TruncationStats()
        evolved = propagate_forward(ham, gate_seq, delta, stats)
        E = sum(a for P, a in evolved.items() if is_iz_only(P))
        return E, stats.error_estimate

    E_exact, _ = energy(0.0)
    print(f"  exact (delta=0): E = {E_exact:.10f}")

    for delta in (1e-2, 1e-3, 1e-4, 1e-5):
        E, eps = energy(delta)
        err = abs(E - E_exact)
        print(f"  delta={delta:.0e}: |dE|={err:.3e}, eps_emp={eps:.3e}")
        assert eps >= err, (
            f"error estimate {eps:.3e} below observed error {err:.3e} "
            f"at delta={delta}"
        )

    print("  ✅ PASSED\n")


def test_13_two_stage_optimizer():
    """Trotter warm start -> Adam -> L-BFGS-B 전 구간이 도는지, 그리고
    BP-PPS 에너지가 실제로 배포될 상태 U|0> 의 에너지인지 확인."""
    from qiskit.quantum_info import Statevector
    from bppps import BPPPSTrainer, trotter_warm_start
    from bppps.pauli_utils import make_observable_label

    print("=" * 60)
    print("TEST 13: warm start + Adam + L-BFGS-B (2x2, 3 layers)")
    print("=" * 60)

    model = SpinGlass2D(Lx=2, Ly=2, h=1.0, coupling_type='ea_bimodal', seed=7)
    N = model.num_qubits
    hva = HVA(num_qubits=N, bonds=model.bonds, n_layers=3, Lx=2, Ly=2, J=model.J)

    ham = {}
    for idx, (i, j) in enumerate(model.bonds):
        chars = ['I'] * N
        chars[i] = 'Z'
        chars[j] = 'Z'
        key = ''.join(chars)
        ham[key] = ham.get(key, 0.0) - model.J[idx]
    for q in range(N):
        key = make_observable_label(N, 'X', q)
        ham[key] = ham.get(key, 0.0) - model.h

    trainer = BPPPSTrainer(
        num_qubits=N, bonds=model.bonds, substep_bonds=hva.substep_bonds,
        n_layers=3, delta=0.0, mode='ground_state', hamiltonian_spo=ham,
    )
    params_init = trotter_warm_start(N, model.bonds, model.J, model.h, 0.5, 3)
    config = {
        'stage1': {'name': 'Adam', 'learning_rate': 0.05, 'epochs': 40},
        'stage2': {'enabled': True, 'name': 'L-BFGS-B',
                   'max_iter': 100, 'tolerance_grad': 1e-9},
    }
    params, record = trainer.optimize(config, params_init=params_init,
                                      verbose=False)

    psi = np.array(Statevector.from_label('0' * N).evolve(
        hva.build_circuit(params)))
    H = model.build_sparse_matrix()
    E_sv = float(np.real(np.conj(psi) @ (H @ psi)))

    print(f"  Adam    : {record['adam_losses'][0]:.6f} -> "
          f"{record['adam_losses'][-1]:.6f}")
    print(f"  L-BFGS-B: -> {record['final_loss']:.6f} "
          f"({len(record['lbfgsb_losses'])} evaluations)")
    print(f"  E_SPD = {record['final_loss']:.10f} vs "
          f"E_statevector[U|0>] = {E_sv:.10f}")

    assert record['lbfgsb_losses'], "L-BFGS-B stage did not run"
    assert record['final_loss'] <= record['adam_losses'][-1] + 1e-9, \
        "L-BFGS-B did not improve on Adam"
    assert abs(record['final_loss'] - E_sv) < 1e-9, \
        f"SPD energy {record['final_loss']} != energy of U|0> {E_sv}"

    print("  ✅ PASSED\n")

def test_14_trotter_sequence_reversal_invariant():
    """타겟 Trotter 시퀀스가 역순 불변인지 검증.

    게이트 순서 버그를 고치기 전에 생성된 targets_*.json 을 계속 쓸 수 있는지가
    여기에 달려 있다. S2 스텝은 [RX(all), RZZ(all), RX(all)] 로 palindromic 이고,
    한 블록 안의 RX 끼리·RZZ 끼리는 서로 교환하며, S4 = S2(p)S2(p)S2(q)S2(p)S2(p)
    역시 palindromic 이다. 따라서 시퀀스를 뒤집어도 같은 유니터리이고,
    타겟은 순서 버그의 영향을 받지 않는다.

    구조적 성질이므로 격자 크기와 무관하다. 여기서는 2x2 에서 절단 없이
    (delta=0) 두 방향의 전파 결과가 일치하는지 확인한다.
    """
    from bppps.propagation import (
        build_trotter_gate_sequence, propagate_forward,
    )
    from bppps.pauli_utils import make_observable_label

    print("=" * 60)
    print("TEST 14: target Trotter sequence is reversal-invariant")
    print("=" * 60)

    model = SpinGlass2D(Lx=2, Ly=2, h=1.0, coupling_type='ea_bimodal', seed=7)
    N = model.num_qubits
    hva = HVA(num_qubits=N, bonds=model.bonds, n_layers=1, Lx=2, Ly=2, J=model.J)

    for order in (2, 4):
        seq = build_trotter_gate_sequence(
            N, hva.substep_bonds, model.J, model.h,
            dt=0.05, n_steps=4, order=order)
        rev = list(reversed(seq))

        max_dev = 0.0
        for pauli, q in (('X', 0), ('Z', 1)):
            label = make_observable_label(N, pauli, q)
            a = propagate_forward({label: 1.0}, seq, 0.0)
            b = propagate_forward({label: 1.0}, rev, 0.0)
            for P in set(a) | set(b):
                max_dev = max(max_dev, abs(a.get(P, 0.0) - b.get(P, 0.0)))

        print(f"  S{order}: max |forward - reversed| = {max_dev:.3e} "
              f"({len(seq)} gates)")
        assert max_dev < 1e-12, (
            f"S{order} Trotter sequence is NOT reversal-invariant "
            f"(dev={max_dev}); cached targets would need regenerating."
        )

    # 1차 Trotter 는 palindromic 이 아니므로 반대로 불변이 아니어야 한다.
    seq1 = build_trotter_gate_sequence(
        N, hva.substep_bonds, model.J, model.h, dt=0.3, n_steps=3, order=1)
    label = make_observable_label(N, 'X', 0)
    a = propagate_forward({label: 1.0}, seq1, 0.0)
    b = propagate_forward({label: 1.0}, list(reversed(seq1)), 0.0)
    dev1 = max(abs(a.get(P, 0.0) - b.get(P, 0.0)) for P in set(a) | set(b))
    print(f"  S1: max |forward - reversed| = {dev1:.3e} (expected non-zero)")
    assert dev1 > 1e-9, "S1 unexpectedly reversal-invariant; test is not sensitive"

    print("  ✅ PASSED\n")

if __name__ == '__main__':
    test_1_ferromagnetic()
    test_2_pauli_op_consistency()
    test_3_energy_consistency()
    test_4_frustration()
    test_5_time_evolution()
    test_6_hva()
    test_7_trotter_vs_ed()
    test_8_trotter_dt_comparison()
    test_9_memory_safety()
    test_10_spd_matches_circuit()
    test_11_bppps_gradient()
    test_12_truncation_error_estimate()
    test_13_two_stage_optimizer()
    test_14_trotter_sequence_reversal_invariant()

    print("=" * 60)
    print("ALL 14 TESTS PASSED ✅")
    print("=" * 60)
