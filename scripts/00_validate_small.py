"""Phase 0-3 integration test (OOM-safe version).

모든 검증을 sparse 연산으로 수행합니다.
SparsePauliOp 검증도 dense 변환 없이 sparse matrix 비교로 진행합니다.
"""
import sys
sys.path.insert(0, '/home/hyunwoo/workspace/spin-glass-qml/src')

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
        num_qubits=model.num_qubits, bonds=model.bonds,
        J=model.J, h=model.h, Lx=2, Ly=2,
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
        num_qubits=model.num_qubits, bonds=model.bonds,
        J=model.J, h=model.h, Lx=2, Ly=2,
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

    # ExactDiag도 차단되어야 함
    try:
        import scipy.sparse as sp
        dummy_H = sp.eye(4, format='csr')
        ExactDiag(dummy_H, num_qubits=25)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"  ExactDiag(25q) 차단: ✅")

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

    print("=" * 60)
    print("ALL 9 TESTS PASSED ✅")
    print("=" * 60)
