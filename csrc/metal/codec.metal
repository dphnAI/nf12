// SPDX-License-Identifier: Apache-2.0
// Encode, decode, bf16t packing and embedding lookup (SPEC.md sections 2-4, 10).
// Unless noted, one simdgroup handles one super-atom and a threadgroup holds
// kSimdgroups simdgroups; sa is uniform per simdgroup, so early returns never
// split a simdgroup before a simd collective.
#include "nf12_metal.h"

using namespace nf12;

constant constexpr uint kSimdgroups = 8;
constant constexpr uint kHistBins = 128;
constant constexpr uint kAnalyzeSuperAtomsPerSimdgroup = 16;

inline ulong simdgroup_super_atom(uint tg, uint sg) { return ulong(tg) * kSimdgroups + sg; }

// ---------------------------------------------------------------------------
// Encode, pass 1: per-group exponent intervals. hist[0][b] counts groups whose
// fitting interval of bases starts at b, hist[1][b] those ending at b, and
// hist[2][b] groups that collide with the sentinel at base b (see
// reference.escape_counts_all_bases).
kernel void nf12_encode_analyze(device const ushort* w [[buffer(0)]],
                                constant ulong& ldw [[buffer(1)]],
                                constant ulong& num_sa [[buffer(2)]],
                                constant uint& k_blocks [[buffer(3)]],
                                device atomic_uint* hist [[buffer(4)]],
                                uint tg [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
  threadgroup atomic_uint sh[3 * kHistBins];
  for (uint i = tid; i < 3 * kHistBins; i += kSimdgroups * 32) atomic_store_explicit(&sh[i], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ulong sa0 = ulong(tg) * kAnalyzeSuperAtomsPerSimdgroup * kSimdgroups + sg;
  for (uint i = 0; i < kAnalyzeSuperAtomsPerSimdgroup; ++i) {
    const ulong sa = sa0 + ulong(i) * kSimdgroups;
    if (sa >= num_sa) break;
    const ulong off = lane_offset(super_atom(sa, k_blocks), ldw, lane);
    for (uint j = 0; j < 4; ++j) {
      uint r[4];
      load_atom(w, off, ldw, j, r);
      uint hmin = 127, hmax = 0;
      bool positive = true;
      for (uint q = 0; q < 4; ++q) {
        const uint ha = (r[q] >> 8) & 0x7fu, hb = (r[q] >> 24) & 0x7fu;
        hmin = min(hmin, min(ha, hb));
        hmax = max(hmax, max(ha, hb));
        positive = positive && (r[q] & 0x80008000u) == 0u;
      }
      const uint lo = hmax > 7u ? hmax - 7u : 0u;
      const uint hi = min(hmin, kMaxExpBase);
      if (lo <= hi) {
        atomic_fetch_add_explicit(&sh[lo], 1u, memory_order_relaxed);
        atomic_fetch_add_explicit(&sh[kHistBins + hi], 1u, memory_order_relaxed);
      }
      if (positive && hmin == hmax && hmin <= kMaxExpBase)
        atomic_fetch_add_explicit(&sh[2 * kHistBins + hmin], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i = tid; i < 3 * kHistBins; i += kSimdgroups * 32) {
    const uint v = atomic_load_explicit(&sh[i], memory_order_relaxed);
    if (v != 0u) atomic_fetch_add_explicit(&hist[i], v, memory_order_relaxed);
  }
}

// Encode, pass 2: escaped groups per super-atom.
kernel void nf12_encode_count(device const ushort* w [[buffer(0)]],
                              constant ulong& ldw [[buffer(1)]],
                              constant ulong& num_sa [[buffer(2)]],
                              constant uint& k_blocks [[buffer(3)]],
                              constant uint& exp_base [[buffer(4)]],
                              device uint* counts [[buffer(5)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const ulong sa = simdgroup_super_atom(tg, sg);
  if (sa >= num_sa) return;
  const ulong off = lane_offset(super_atom(sa, k_blocks), ldw, lane);
  uint escaped = 0;
  for (uint j = 0; j < 4; ++j) {
    uint r[4];
    load_atom(w, off, ldw, j, r);
    Group g;
    escaped += encode_group(r, exp_base, g) ? 0u : 1u;
  }
  escaped = simd_sum(escaped);
  if (lane == 0) counts[sa] = escaped;
}

// Encode, pass 3: packed bytes and escapes. `offsets` is the exclusive prefix
// sum of the per-super-atom counts, so escape indices follow flat group id
// order (atom-major, then lane) independent of the launch grid.
kernel void nf12_encode_write(device const ushort* w [[buffer(0)]],
                              constant ulong& ldw [[buffer(1)]],
                              constant ulong& num_sa [[buffer(2)]],
                              constant uint& k_blocks [[buffer(3)]],
                              constant uint& exp_base [[buffer(4)]],
                              device const long* offsets [[buffer(5)]],
                              device uint4* packed [[buffer(6)]],
                              device uint4* escapes [[buffer(7)]],
                              uint tg [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
  const ulong sa = simdgroup_super_atom(tg, sg);
  if (sa >= num_sa) return;
  const ulong off = lane_offset(super_atom(sa, k_blocks), ldw, lane);
  uint next = uint(offsets[sa]);
  Group g[4];
  for (uint j = 0; j < 4; ++j) {
    uint r[4];
    load_atom(w, off, ldw, j, r);
    const bool esc = !encode_group(r, exp_base, g[j]);
    const uint idx = next + simd_prefix_exclusive_sum(esc ? 1u : 0u);
    if (esc) {
      escapes[idx] = uint4(r[0], r[1], r[2], r[3]);
      g[j] = Group{0u, idx, 0u};
    }
    next += simd_sum(esc ? 1u : 0u);
  }
  device uint4* dst = packed + sa * (kSuperAtomBytes / 16) + lane;
  dst[0] = uint4(g[0].meta, g[1].meta, g[2].meta, g[3].meta);
  dst[kPlaneBytes / 16] = uint4(g[0].lo0, g[0].lo1, g[1].lo0, g[1].lo1);
  dst[2 * kPlaneBytes / 16] = uint4(g[2].lo0, g[2].lo1, g[3].lo0, g[3].lo1);
}

// ---------------------------------------------------------------------------
// Decode to a row-major [N, K] bf16 matrix with leading dimension ldo.
// grid.y selects the expert for stacked MoE weights (expert-local escapes and
// exp_base, SPEC.md 9.1); dense weights use one "expert".
kernel void nf12_decode(device const uint4* packed [[buffer(0)]],
                        device const uint4* escapes [[buffer(1)]],
                        constant long* escape_offsets [[buffer(2)]],
                        device const uchar* exp_base [[buffer(3)]],
                        constant ulong& num_sa [[buffer(4)]],
                        constant uint& k_blocks [[buffer(5)]],
                        constant ulong& ldo [[buffer(6)]],
                        constant ulong& expert_rows [[buffer(7)]],
                        device ushort* out [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
  const ulong sa = simdgroup_super_atom(tg.x, sg);
  if (sa >= num_sa) return;
  const uint e = tg.y;
  const NF12Source src{packed + ulong(e) * num_sa * (kSuperAtomBytes / 16), escapes + escape_offsets[e],
                       exp_base_x4(exp_base[e])};
  uint r[4][4];
  src.load(sa, lane, r);
  const ulong off = lane_offset(super_atom(sa, k_blocks), ldo, lane);
  device ushort* dst = out + ulong(e) * expert_rows * ldo;
  for (uint j = 0; j < 4; ++j) store_atom(dst, off, ldo, j, r[j]);
}

// Row-major bf16 -> bf16t twin (SPEC.md 4.4): atom j's registers at plane j.
kernel void nf12_pack_bf16t(device const ushort* w [[buffer(0)]],
                            constant ulong& ldw [[buffer(1)]],
                            constant ulong& num_sa [[buffer(2)]],
                            constant uint& k_blocks [[buffer(3)]],
                            device uint4* out [[buffer(4)]],
                            uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
  const ulong sa = simdgroup_super_atom(tg, sg);
  if (sa >= num_sa) return;
  const ulong off = lane_offset(super_atom(sa, k_blocks), ldw, lane);
  device uint4* dst = out + sa * (kSuperAtomBytesBf16t / 16) + lane;
  for (uint j = 0; j < 4; ++j) {
    uint r[4];
    load_atom(w, off, ldw, j, r);
    dst[j * (kPlaneBytes / 16)] = uint4(r[0], r[1], r[2], r[3]);
  }
}

// ---------------------------------------------------------------------------
// Embedding lookup (SPEC.md section 10). Thread (x, y) = (16 * kb + 4 * j + t, id):
// it decodes group (atom j, lane 4g + t) of the id's row block, which holds 4 of
// the row's values (columns 64kb + 16j + {2t, 2t+1, 2t+8, 2t+9}).
template <typename Id>
inline void embedding_impl(device const uint* packed, device const uint4* escapes, uint base_x4,
                           device const Id* ids, ulong vocab, uint k_blocks, device uint* out, ulong ldo,
                           uint2 gid) {
  const uint kb = gid.x >> 4, j = (gid.x >> 2) & 3u, t = gid.x & 3u;
  if (kb >= k_blocks) return;
  const long id = long(ids[gid.y]);
  device uint* dst = out + (ulong(gid.y) * ldo + ulong(kb) * kSuperAtomCols + 16 * j + 2 * t) / 2;
  if (id < 0 || ulong(id) >= vocab) {
    dst[0] = 0u;
    dst[4] = 0u;
    return;
  }
  const ulong row = ulong(id);
  const uint ro = uint(row & 15u), lane = 4 * (ro & 7u) + t;
  const ulong base = ((row >> 4) * k_blocks + kb) * (kSuperAtomBytes / 4) + 4 * lane;  // in uints
  // Atoms 0-1 keep L0/L1 in plane 1, atoms 2-3 in plane 2 (SPEC.md 4.2).
  const ulong lo = base + (1 + (j >> 1)) * (kPlaneBytes / 4) + 2 * (j & 1u);
  const Group g{packed[base + j], packed[lo], packed[lo + 1]};
  uint r[4];
  decode_group(g, base_x4, escapes, r);
  const uint h = ro >> 3;  // rows g (R0, R2) or g + 8 (R1, R3)
  dst[0] = r[h];
  dst[4] = r[2 + h];
}

kernel void nf12_embedding_i32(device const uint* packed [[buffer(0)]],
                               device const uint4* escapes [[buffer(1)]],
                               device const uchar* exp_base [[buffer(2)]],
                               device const int* ids [[buffer(3)]],
                               constant ulong& vocab [[buffer(4)]],
                               constant uint& k_blocks [[buffer(5)]],
                               device uint* out [[buffer(6)]],
                               constant ulong& ldo [[buffer(7)]],
                               uint2 gid [[thread_position_in_grid]]) {
  embedding_impl(packed, escapes, exp_base_x4(exp_base[0]), ids, vocab, k_blocks, out, ldo, gid);
}

kernel void nf12_embedding_i64(device const uint* packed [[buffer(0)]],
                               device const uint4* escapes [[buffer(1)]],
                               device const uchar* exp_base [[buffer(2)]],
                               device const long* ids [[buffer(3)]],
                               constant ulong& vocab [[buffer(4)]],
                               constant uint& k_blocks [[buffer(5)]],
                               device uint* out [[buffer(6)]],
                               constant ulong& ldo [[buffer(7)]],
                               uint2 gid [[thread_position_in_grid]]) {
  embedding_impl(packed, escapes, exp_base_x4(exp_base[0]), ids, vocab, k_blocks, out, ldo, gid);
}

// Fills `count` 32-bit words with `value`. Backends whose encoders do not zero
// new buffers (MLX) use it for the encode histograms and zero-filled outputs.
// 256 threads per threadgroup; the index is 64-bit, so outputs may exceed 2^32 words.
kernel void nf12_fill_u32(device uint* out [[buffer(0)]],
                          constant ulong& count [[buffer(1)]],
                          constant uint& value [[buffer(2)]],
                          uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]]) {
  const ulong i = ulong(tg) * 256 + tid;
  if (i < count) out[i] = value;
}
