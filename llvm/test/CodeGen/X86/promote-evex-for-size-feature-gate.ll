; Correctness test for X86PromoteEVEXForSize's feature gate.
;
; Promotion (VEX->EVEX) is uphill in features: an EVEX twin can require a feature
; that hasVLX() does not imply. vandps/vandpd's EVEX form (VANDPSZ128rm) needs
; AVX512DQ. Promoting it on a VLX-but-no-DQ target would emit an instruction the
; CPU cannot decode -- a miscompile. The generated featuresAvailableForPromote()
; gate must refuse promotion unless AVX512DQ is present.
;
; When AVX512DQ is present, promote to EVEX (0x62 prefix).
; RUN: llc -mtriple=x86_64-- -mattr=+avx512vl,+avx512dq,+prefer-evex-for-size \
; RUN:   -show-mc-encoding < %s | FileCheck %s --check-prefix=DQ
; When AVX512DQ is absent, the instruction must stay VEX (0xc5/0xc4 prefix) even
; though the displacement is otherwise compressible.
; RUN: llc -mtriple=x86_64-- -mattr=+avx512vl,-avx512dq,+prefer-evex-for-size \
; RUN:   -show-mc-encoding < %s | FileCheck %s --check-prefix=NODQ

; DQ:   vandps {{[0-9]+}}(%rdi), %xmm{{[0-9]+}}, %xmm{{[0-9]+}} # {{.*}}[0x62,
; NODQ: vandps {{[0-9]+}}(%rdi), %xmm{{[0-9]+}}, %xmm{{[0-9]+}} # {{.*}}[0xc5,
define <2 x double> @andpd_bigdisp(ptr %p, <2 x double> %x) {
  %g = getelementptr i8, ptr %p, i64 2032
  %l = load <2 x double>, ptr %g, align 16
  %ax = bitcast <2 x double> %x to <2 x i64>
  %al = bitcast <2 x double> %l to <2 x i64>
  %r = and <2 x i64> %ax, %al
  %rr = bitcast <2 x i64> %r to <2 x double>
  ret <2 x double> %rr
}
