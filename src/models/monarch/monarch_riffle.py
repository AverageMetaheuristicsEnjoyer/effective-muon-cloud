"""Fast riffle for the butterfly forward: out1 (B,k,q) -> t (l,B,r).

The eager riffle is a stride-4 gather (~1 TB/s). Two faster variants:

A. Triton kernel: load contiguous (BB, Q) rows, de-interleave in registers
   via reshape+split, store four contiguous l-planes. Coalesced both ways.
B. Zero-kernel: permute w1's rows to l-major per call (tiny), so bmm1's
   output is already l-grouped and the riffle becomes a permute whose
   innermost runs are 160B -> aten copies it at near-full bandwidth.

Both are bitwise-identical to the eager riffle (same dot products, only the
order in which finished values are moved changes).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _riffle_kernel(O1, T, B, Q: tl.constexpr, QPAD: tl.constexpr,
                   BB: tl.constexpr):
    """O1 (B, k, Q) contiguous storage; T (4, B, Q). One CTA: (BB rows, one k)."""
    pid_b = tl.program_id(0)
    k = tl.program_id(1)
    NBK: tl.constexpr = 4
    QSUB: tl.constexpr = Q // NBK
    JPAD: tl.constexpr = QPAD // NBK
    K: tl.constexpr = 4  # nb blocks in stage 1

    offs_b = pid_b * BB + tl.arange(0, BB)
    offs_q = tl.arange(0, QPAD)
    qm = offs_q < Q
    src = O1 + offs_b[:, None] * (K * Q) + k * Q + offs_q[None, :]
    x = tl.load(src, mask=qm[None, :], other=0.0)      # (BB, QPAD) contiguous

    x = tl.reshape(x, (BB, JPAD, 2, 2))                # q = j*4 + (hi*2+lo)
    lo0, lo1 = tl.split(x)                             # (BB, JPAD, 2) each
    p0, p2 = tl.split(lo0)                             # l = 0, 2
    p1, p3 = tl.split(lo1)                             # l = 1, 3

    offs_j = tl.arange(0, JPAD)
    jm = offs_j < QSUB
    base = offs_b[:, None] * Q + k * QSUB + offs_j[None, :]
    tl.store(T + 0 * (B * Q) + base, p0, mask=jm[None, :])
    tl.store(T + 1 * (B * Q) + base, p1, mask=jm[None, :])
    tl.store(T + 2 * (B * Q) + base, p2, mask=jm[None, :])
    tl.store(T + 3 * (B * Q) + base, p3, mask=jm[None, :])


def riffle_triton(out1_storage, B, nb, q):
    """out1_storage: (B, nb, q) contiguous. Returns t (nb, B, q) blocked."""
    t = torch.empty(nb, B, q, device=out1_storage.device, dtype=out1_storage.dtype)
    BB = 32
    qpad = triton.next_power_of_2(q)
    _riffle_kernel[(B // BB, nb)](out1_storage, t, B, q, qpad, BB,
                                  num_warps=4, num_stages=2)
    return t


@triton.jit
def _unriffle_kernel(DT, DO1, B, Q: tl.constexpr, QPAD: tl.constexpr,
                     BB: tl.constexpr):
    """Inverse riffle: dout1[b, k, j*4+l] = dt[l, b, k*QSUB+j].

    DT (4, B, Q) blocked planes -> DO1 (B, K, Q) contiguous storage.
    Loads four contiguous plane segments, joins in registers, one
    contiguous store."""
    pid_b = tl.program_id(0)
    k = tl.program_id(1)
    NBK: tl.constexpr = 4
    QSUB: tl.constexpr = Q // NBK
    JPAD: tl.constexpr = QPAD // NBK
    K: tl.constexpr = 4

    offs_b = pid_b * BB + tl.arange(0, BB)
    offs_j = tl.arange(0, JPAD)
    jm = offs_j < QSUB
    src = offs_b[:, None] * Q + k * QSUB + offs_j[None, :]
    p0 = tl.load(DT + 0 * (B * Q) + src, mask=jm[None, :], other=0.0)
    p1 = tl.load(DT + 1 * (B * Q) + src, mask=jm[None, :], other=0.0)
    p2 = tl.load(DT + 2 * (B * Q) + src, mask=jm[None, :], other=0.0)
    p3 = tl.load(DT + 3 * (B * Q) + src, mask=jm[None, :], other=0.0)

    j01 = tl.join(p0, p1)                 # (BB, JPAD, 2)  last axis c1 = l&1
    j23 = tl.join(p2, p3)
    jj = tl.join(j01, j23)                # (BB, JPAD, 2, 2) [.., c1, c2] = p_{2*c2+c1}
    jj = tl.permute(jj, (0, 1, 3, 2))     # -> [.., c2, c1]: flat l = 2*c2+c1 holds p_l
    x = tl.reshape(jj, (BB, JPAD * 4))    # q = j*4 + l

    offs_q = tl.arange(0, QPAD)
    qm = offs_q < Q
    dst = DO1 + offs_b[:, None] * (K * Q) + k * Q + offs_q[None, :]
    tl.store(dst, x, mask=qm[None, :])


def unriffle_triton(dt, B, nb, q):
    """dt (nb, B, q) blocked -> dout1 storage (B, nb, q) contiguous."""
    do1 = torch.empty(B, nb, q, device=dt.device, dtype=dt.dtype)
    BB = 32
    qpad = triton.next_power_of_2(q)
    _unriffle_kernel[(B // BB, nb)](dt, do1, B, q, qpad, BB,
                                    num_warps=4, num_stages=2)
    return do1


def w1_l_major(w1):
    """Permute w1 rows q = j*nb+l -> q' = l*qsub+j (tiny tensor, ~5us)."""
    k, q, p = w1.shape
    nb = k
    return (w1.view(k, q // nb, nb, p).permute(0, 2, 1, 3)
            .reshape(k, q, p).contiguous())


def riffle_after_lmajor(out1_lm_storage, B, nb, q):
    """out1 from l-major w1: storage (B, k, l, qsub) -> t (l, B, k*qsub+j).

    permute keeps 160B-contiguous innermost runs; aten copies near-BW.
    """
    qsub = q // nb
    return (out1_lm_storage.view(B, nb, nb, qsub)
            .permute(2, 0, 1, 3).reshape(nb, B, q).contiguous())


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    B = 16 * 1024
    for (k, q, p) in [(4, 320, 320), (4, 320, 896)]:
        w1 = (torch.randn(k, q, p, device="cuda") * 0.02).bfloat16()
        x = torch.randn(B, k * p, device="cuda", dtype=torch.bfloat16)
        x_r = x.reshape(B, k, p).transpose(0, 1)

        def bmm1(w):
            o = torch.empty(B, k, q, device="cuda", dtype=torch.bfloat16).transpose(0, 1)
            return torch.bmm(x_r, w.transpose(-1, -2), out=o)

        # eager reference
        o1 = bmm1(w1)
        ref_t = (o1.transpose(0, 1).reshape(B, q, k)
                 .permute(2, 0, 1).contiguous())

        # A: triton riffle on natural out1 storage
        tA = riffle_triton(o1.transpose(0, 1).contiguous(), B, k, q)
        okA = torch.equal(tA, ref_t)

        # B: l-major w1 + aten permute
        w1lm = w1_l_major(w1)
        o1lm = bmm1(w1lm)
        tB = riffle_after_lmajor(o1lm.transpose(0, 1).contiguous(), B, k, q)
        okB = torch.equal(tB, ref_t)

        def bench(fn, it=100):
            for _ in range(10):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(it):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) / it * 1e3

        o1s = o1.transpose(0, 1).contiguous()
        o1lms = o1lm.transpose(0, 1).contiguous()
        ms_eager = bench(lambda: (o1s.reshape(B, q, k).permute(2, 0, 1).contiguous()))
        ms_A = bench(lambda: riffle_triton(o1s, B, k, q))
        ms_B = bench(lambda: riffle_after_lmajor(o1lms, B, k, q))
        gb = 2 * B * k * q * 2 / 1e9
        print(f"k{k} q{q} p{p}: eq A={okA} B={okB} | eager {ms_eager:.4f} ms "
              f"({gb/ms_eager:.2f} TB/s) | triton {ms_A:.4f} ({gb/ms_A:.2f} TB/s) "
              f"| lmajor {ms_B:.4f} ({gb/ms_B:.2f} TB/s)", flush=True)
