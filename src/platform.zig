//! Target-dependent facts the SIMD build is parameterized by (RFC §3.2).
//!
//! Nothing here hard-codes a lane count: `lane_count` is whatever the compile
//! target's widest natural f32 vector is (8 on AVX2, 16 on AVX-512, 4 on
//! NEON). Kernels take `L` as a comptime parameter so the harness can also
//! instantiate them at other widths as a labeled experiment.

const std = @import("std");

/// The widest natural f32 vector width for the compile target.
pub const lane_count: usize = std.simd.suggestVectorLength(f32) orelse 8;

/// `L` f32 values packed into one hardware vector register.
pub fn Vec(comptime L: usize) type {
    return @Vector(L, f32);
}

/// One cache line — at least as wide as any vector in play. SoA arrays are
/// allocated to this alignment: hygiene rather than correctness (modern cores
/// tolerate unaligned vector loads), but free, and it removes a variable from
/// benchmark interpretation (RFC §3.2 "Alignment").
pub const cache_line: std.mem.Alignment = .@"64";

/// Rounds `n` up to a multiple of `m`.
pub fn alignUp(n: usize, m: usize) usize {
    std.debug.assert(m != 0);
    return (n + m - 1) / m * m;
}

/// Allocates a cache-line-aligned f32 array.
pub fn allocLanes(allocator: std.mem.Allocator, n: usize) ![]align(cache_line.toByteUnits()) f32 {
    return allocator.alignedAlloc(f32, cache_line, n);
}

test "lane count is a sane power of two" {
    try std.testing.expect(lane_count >= 1);
    try std.testing.expect(std.math.isPowerOfTwo(lane_count));
}

test "alignUp rounds to the next multiple" {
    try std.testing.expectEqual(@as(usize, 0), alignUp(0, 4));
    try std.testing.expectEqual(@as(usize, 4), alignUp(1, 4));
    try std.testing.expectEqual(@as(usize, 4), alignUp(4, 4));
    try std.testing.expectEqual(@as(usize, 8), alignUp(5, 4));
}

test "allocLanes returns cache-line-aligned memory" {
    const buf = try allocLanes(std.testing.allocator, 32);
    defer std.testing.allocator.free(buf);
    try std.testing.expectEqual(@as(usize, 0), @intFromPtr(buf.ptr) % cache_line.toByteUnits());
}
