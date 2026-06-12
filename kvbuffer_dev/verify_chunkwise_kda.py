"""Verify chunkwise (KVBuffer) gated-delta-rule == recurrent KDA, pure python.

Convention matches cula/ops/kda_decode_mtp.py vk kernel:
  state S has shape [V, K], S[v][k].
  per token t (decay-first):
    g_t[k] = decay (per channel)         # given directly here
    S[v][k] *= g_t[k]                     # step1 decay
    s[v] = sum_k S[v][k]*k_t[k]           # step2
    v_new[v] = beta_t*(v_t[v]-s[v])       # step3
    S[v][k] += v_new[v]*k_t[k]            # step4 rank-1
    o_t[v] = sum_k S[v][k]*q_t[k]         # step5 (post-update state)
"""
import math, random

random.seed(0)
V, K, T = 4, 4, 3

def randmat(r, c): return [[random.uniform(-1,1) for _ in range(c)] for _ in range(r)]
def randvec(n):    return [random.uniform(-1,1) for _ in range(n)]

S0   = randmat(V, K)
q    = [randvec(K) for _ in range(T)]
k    = [randvec(K) for _ in range(T)]
vv   = [randvec(V) for _ in range(T)]
g    = [[random.uniform(0.3,1.0) for _ in range(K)] for _ in range(T)]  # decay in (0,1]
beta = [random.uniform(0.0,1.0) for _ in range(T)]

# ---------------- RECURRENT (ground truth) ----------------
def recurrent():
    S = [row[:] for row in S0]
    outs = []
    for t in range(T):
        for v in range(V):
            for c in range(K):
                S[v][c] *= g[t][c]                      # decay
        s = [sum(S[v][c]*k[t][c] for c in range(K)) for v in range(V)]
        vnew = [beta[t]*(vv[t][v]-s[v]) for v in range(V)]
        for v in range(V):
            for c in range(K):
                S[v][c] += vnew[v]*k[t][c]              # rank-1
        o = [sum(S[v][c]*q[t][c] for c in range(K)) for v in range(V)]
        outs.append(o)
    return outs, S

# ---------------- CHUNKWISE (KVBuffer) ----------------
def chunkwise():
    # cumulative decay b_t[k] = prod_{i<=t} g_i[k], b indexed 0..T-1 = after token t
    b = [[0.0]*K for _ in range(T)]
    for c in range(K):
        acc = 1.0
        for t in range(T):
            acc *= g[t][c]
            b[t][c] = acc
    # decayed / inv-decayed feature maps
    kdec = [[k[t][c]*b[t][c]       for c in range(K)] for t in range(T)]  # k_t * b_t
    kinv = [[k[t][c]/b[t][c]       for c in range(K)] for t in range(T)]  # k_t / b_t
    qdec = [[q[t][c]*b[t][c]       for c in range(K)] for t in range(T)]  # q_t * b_t

    # A[t][i] = <kdec_t, kinv_i>  (strict lower, i<t); P[t][i] = <qdec_t, kinv_i> (i<=t)
    A = [[sum(kdec[t][c]*kinv[i][c] for c in range(K)) for i in range(T)] for t in range(T)]
    P = [[sum(qdec[t][c]*kinv[i][c] for c in range(K)) for i in range(T)] for t in range(T)]

    # S0 @ kdec_t  and  S0 @ qdec_t  (V-vectors per t)
    Skdec = [[sum(S0[v][c]*kdec[t][c] for c in range(K)) for v in range(V)] for t in range(T)]
    Sqdec = [[sum(S0[v][c]*qdec[t][c] for c in range(K)) for v in range(V)] for t in range(T)]

    # triangular solve for u_t[v]: u_t = beta_t (v_t - S0@kdec_t - sum_{i<t} A[t][i] u_i)
    u = [[0.0]*V for _ in range(T)]
    for t in range(T):
        for v in range(V):
            acc = vv[t][v] - Skdec[t][v]
            for i in range(t):
                acc -= A[t][i]*u[i][v]
            u[t][v] = beta[t]*acc

    # outputs o_t[v] = S0@qdec_t + sum_{i<=t} P[t][i] u_i[v]
    outs = []
    for t in range(T):
        o = []
        for v in range(V):
            acc = Sqdec[t][v]
            for i in range(t+1):
                acc += P[t][i]*u[i][v]
            o.append(acc)
        outs.append(o)

    # final state S_T[v][k] = b_{T-1}[k]*(S0[v][k] + sum_i u_i[v]*kinv_i[k])
    ST = [[b[T-1][c]*(S0[v][c] + sum(u[i][v]*kinv[i][c] for i in range(T)))
           for c in range(K)] for v in range(V)]
    return outs, ST

oR, SR = recurrent()
oC, SC = chunkwise()

def maxerr(a, b):
    return max(abs(x-y) for ra,rb in zip(a,b) for x,y in zip(ra,rb))

print("output max abs err :", maxerr(oR, oC))
print("state  max abs err :", maxerr(SR, SC))
ok = maxerr(oR,oC) < 1e-9 and maxerr(SR,SC) < 1e-9
print("MATCH" if ok else "MISMATCH")
