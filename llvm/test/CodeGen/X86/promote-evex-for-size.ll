; Tests for X86PromoteEVEXForSize: promote VEX vector memory ops to EVEX when the
; compressed displacement (disp8*N) makes the instruction strictly smaller.
;
; ON  = AVX512VL + the prefer-evex-for-size tuning feature: promotion enabled.
; OFF = AVX512VL only: the pass is gated off, everything stays VEX.
;
; RUN: llc -mtriple=x86_64-- -mattr=+avx512vl,+avx512dq,+prefer-evex-for-size \
; RUN:   -show-mc-encoding < %s | FileCheck %s --check-prefix=ON
; RUN: llc -mtriple=x86_64-- -mattr=+avx512vl,+avx512dq \
; RUN:   -show-mc-encoding < %s | FileCheck %s --check-prefix=OFF

; Move family (the original pass scope) -- still promoted after generalization.
; ON:  vmovaps {{[0-9]+}}(%rdi), %xmm{{[0-9]+}} # {{.*}}[0x62,
; OFF: vmovaps {{[0-9]+}}(%rdi), %xmm{{[0-9]+}} # {{.*}}[0xc5,
define <4 x float> @move_bigdisp(ptr %p) {
  %g = getelementptr i8, ptr %p, i64 2032
  %l = load <4 x float>, ptr %g, align 16
  ret <4 x float> %l
}

; Binary arithmetic with a memory source -- newly promotable family.
; 2032 / 16 = 127 fits int8, so EVEX disp8 wins.
; ON:  vaddps {{[0-9]+}}(%rdi), %xmm{{[0-9]+}}, %xmm{{[0-9]+}} # {{.*}}[0x62,
; OFF: vaddps {{[0-9]+}}(%rdi), %xmm{{[0-9]+}}, %xmm{{[0-9]+}} # {{.*}}[0xc5,
define <4 x float> @addps_bigdisp(ptr %p, <4 x float> %x) {
  %g = getelementptr i8, ptr %p, i64 2032
  %l = load <4 x float>, ptr %g, align 16
  %a = fadd <4 x float> %x, %l
  ret <4 x float> %a
}

; Negative: |disp| <= 127 already uses VEX disp8; EVEX would only grow it.
; ON:  vaddps 64(%rdi), %xmm{{[0-9]+}}, %xmm{{[0-9]+}} # {{.*}}[0xc5,
define <4 x float> @addps_smalldisp(ptr %p, <4 x float> %x) {
  %g = getelementptr i8, ptr %p, i64 64
  %l = load <4 x float>, ptr %g, align 16
  %a = fadd <4 x float> %x, %l
  ret <4 x float> %a
}

; Negative: disp not a multiple of N (=16), so CDisp8 cannot compress it.
; ON:  vaddps 2040(%rdi), %xmm{{[0-9]+}}, %xmm{{[0-9]+}} # {{.*}}[0xc5,
define <4 x float> @addps_nonmultiple(ptr %p, <4 x float> %x) {
  %g = getelementptr i8, ptr %p, i64 2040
  %l = load <4 x float>, ptr %g, align 1
  %a = fadd <4 x float> %x, %l
  ret <4 x float> %a
}
