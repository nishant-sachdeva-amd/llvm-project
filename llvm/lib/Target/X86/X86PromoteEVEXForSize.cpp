//===- X86PromoteEVEXForSize.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass promotes VEX-encoded vector memory moves to their EVEX equivalent
// when doing so makes the instruction strictly smaller. It is the inverse of
// X86CompressEVEX (which only ever shrinks EVEX->VEX/legacy) and runs after it
// so the two can never oscillate.
//
// The win comes from EVEX's compressed displacement (disp8*N / CDisp8): when a
// memory operand uses a displacement that does not fit VEX's signed 8-bit form
// (|disp| > 127) but is an exact multiple of the access size N whose quotient
// disp/N does fit a signed byte, EVEX encodes the displacement in one byte where
// VEX needs four. That 3-byte saving outweighs EVEX's 2-byte-larger prefix, so
// the EVEX instruction is 1-2 bytes shorter.
//
// Promotion is a pure opcode substitution between pairs LLVM already treats as
// equivalent (the entries of X86CompressEVEXTable, inverted here). It is gated
// behind the TuningPreferEVEXForSize subtarget feature (on for znver5) and
// AVX512VL, so it is off by default for every other target.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionAnalysisManager.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOperand.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/IR/Analysis.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/Pass.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>

using namespace llvm;

#define PROMOTE_EVEX_DESC "Promoting VEX vector moves to EVEX to reduce code size"
#define PROMOTE_EVEX_NAME "x86-promote-evex-for-size"

#define DEBUG_TYPE PROMOTE_EVEX_NAME

namespace {
// Reuse the generated EVEX<->VEX equivalence table (EVEX->VEX direction).
#define GET_X86_COMPRESS_EVEX_TABLE
#include "X86GenInstrMapping.inc"

static unsigned getEncoding(uint64_t TSFlags) {
  return TSFlags & X86II::EncodingMask;
}

// True if \p EVEXDesc is a plain vector-move-shaped EVEX form whose VEX twin we
// may promote to. We deliberately scope to the move family: exactly one data
// register plus the 5-operand memory address, CDisp8-capable. This excludes
// - binary/FMA mem ops (vaddsd, vfmadd... mem) -- extra source operands,
// - immediate-rewrite forms (VRNDSCALE/VSHUF/VALIGN) -- extra immediate,
// - gather/scatter -- EVEX uses a k-mask where VEX uses a vector mask, a
//   different operand layout that a bare setDesc would corrupt,
// - masked moves -- extra k operand (and EVEX-only anyway),
// - moffs _alt forms -- CD8 scale is 0.
// Broadening to immediate-free non-move mem ops is possible but is a separate,
// separately-reviewed change.
static bool isPromotableEVEXMemForm(const MCInstrDesc &EVEXDesc) {
  if (getEncoding(EVEXDesc.TSFlags) != X86II::EVEX)
    return false;
  if (X86II::getMemoryOperandNo(EVEXDesc.TSFlags) < 0)
    return false;

  unsigned Scale =
      (EVEXDesc.TSFlags & X86II::CD8_Scale_Mask) >> X86II::CD8_Scale_Shift;
  if (Scale == 0) // not CDisp8-capable (e.g. moffs _alt forms)
    return false;

  // A load/store move has exactly: one data register + the 5 address operands.
  return EVEXDesc.getNumOperands() == X86::AddrNumOperands + 1;
}

// Build VEX-opcode -> EVEX-opcode by inverting X86CompressEVEXTable, keeping only
// promotable plain memory move pairs. The table maps {OldOpc(EVEX or legacy) ->
// NewOpc(smaller)}; we key on NewOpc and require it be VEX-encoded, which drops
// the legacy-SSE->VEX rows that share a VEX key. A VEX opcode can map from
// several byte-equivalent EVEX forms (e.g. VMOVDQUrm <- VMOVDQU{8,16,32,64}); we
// pick deterministically by lowest EVEX opcode value.
static DenseMap<unsigned, unsigned>
buildInverseMap(const X86InstrInfo *TII) {
  DenseMap<unsigned, unsigned> Map;
  for (const X86TableEntry &E : X86CompressEVEXTable) {
    unsigned EVEXOpc = E.OldOpc;
    unsigned VEXOpc = E.NewOpc;
    const MCInstrDesc &VEXDesc = TII->get(VEXOpc);
    if (getEncoding(VEXDesc.TSFlags) != X86II::VEX)
      continue;
    const MCInstrDesc &EVEXDesc = TII->get(EVEXOpc);
    if (!isPromotableEVEXMemForm(EVEXDesc))
      continue;
    auto It = Map.find(VEXOpc);
    if (It == Map.end() || EVEXOpc < It->second)
      Map[VEXOpc] = EVEXOpc;
  }
  return Map;
}

class X86PromoteEVEXForSizeLegacy : public MachineFunctionPass {
public:
  static char ID;
  X86PromoteEVEXForSizeLegacy() : MachineFunctionPass(ID) {}
  StringRef getPassName() const override { return PROMOTE_EVEX_DESC; }

  bool runOnMachineFunction(MachineFunction &MF) override;

  // Runs after regalloc; no virtual register operands.
  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties().setNoVRegs();
  }
};

} // end anonymous namespace

char X86PromoteEVEXForSizeLegacy::ID = 0;

// Returns true if MI was promoted in place.
static bool tryPromote(MachineInstr &MI, const X86Subtarget &ST,
                       const DenseMap<unsigned, unsigned> &InverseMap) {
  const MCInstrDesc &Desc = MI.getDesc();
  if (getEncoding(Desc.TSFlags) != X86II::VEX)
    return false;

  // A VEX-encoded instruction can only reference xmm/ymm0-15 and GPR0-15, so no
  // explicit extended-register check is needed: the EVEX twin is byte-equivalent.
  auto It = InverseMap.find(MI.getOpcode());
  if (It == InverseMap.end())
    return false;
  unsigned EVEXOpc = It->second;
  const MCInstrDesc &EVEXDesc = ST.getInstrInfo()->get(EVEXOpc);

  unsigned f =
      (EVEXDesc.TSFlags & X86II::CD8_Scale_Mask) >> X86II::CD8_Scale_Shift;
  if (f == 0)
    return false;
  int64_t N = 1LL << (f - 1);

  int MemNo = X86II::getMemoryOperandNo(Desc.TSFlags);
  if (MemNo < 0)
    return false;
  MemNo += X86II::getOperandBias(Desc);

  const MachineOperand &Disp = MI.getOperand(MemNo + X86::AddrDisp);
  if (!Disp.isImm()) // RIP-relative / constant-pool / reloc never CDisp8-compress
    return false;
  int64_t D = Disp.getImm();

  // The three-part win predicate (must match X86MCCodeEmitter::isDispOrCDisp8).
  if (D >= -128 && D <= 127)
    return false; // VEX already uses its 1-byte disp8; EVEX would only grow it
  if (D % N != 0)
    return false; // CDisp8 only compresses exact multiples of N
  int64_t Q = D / N;
  if (!isInt<8>(Q))
    return false; // compressed byte must fit a signed byte

  MI.setDesc(EVEXDesc);
  // CompressEVEX may have tagged this instruction when it shrank it EVEX->VEX
  // earlier; clear that stale comment flag so it doesn't mask ours.
  MI.clearAsmPrinterFlag(X86::AC_EVEX_2_LEGACY);
  MI.clearAsmPrinterFlag(X86::AC_EVEX_2_VEX);
  MI.setAsmPrinterFlag(X86::AC_VEX_2_EVEX);
  return true;
}

static bool runOnMF(MachineFunction &MF) {
  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  if (!ST.hasVLX() || !ST.preferEVEXForSize())
    return false;

  LLVM_DEBUG(dbgs() << "Start X86PromoteEVEXForSizePass\n";);
  DenseMap<unsigned, unsigned> InverseMap = buildInverseMap(ST.getInstrInfo());
  LLVM_DEBUG(dbgs() << "VEX->EVEX inverse map has " << InverseMap.size()
                    << " entries\n";);

  bool Changed = false;
  for (MachineBasicBlock &MBB : MF)
    for (MachineInstr &MI : MBB)
      Changed |= tryPromote(MI, ST, InverseMap);

  LLVM_DEBUG(dbgs() << "End X86PromoteEVEXForSizePass\n";);
  return Changed;
}

bool X86PromoteEVEXForSizeLegacy::runOnMachineFunction(MachineFunction &MF) {
  return runOnMF(MF);
}

INITIALIZE_PASS(X86PromoteEVEXForSizeLegacy, PROMOTE_EVEX_NAME,
                PROMOTE_EVEX_DESC, false, false)

FunctionPass *llvm::createX86PromoteEVEXForSizeLegacyPass() {
  return new X86PromoteEVEXForSizeLegacy();
}

PreservedAnalyses
X86PromoteEVEXForSizePass::run(MachineFunction &MF,
                               MachineFunctionAnalysisManager &MFAM) {
  bool Changed = runOnMF(MF);
  if (!Changed)
    return PreservedAnalyses::all();
  PreservedAnalyses PA = getMachineFunctionPassPreservedAnalyses();
  PA.preserveSet<CFGAnalyses>();
  return PA;
}
