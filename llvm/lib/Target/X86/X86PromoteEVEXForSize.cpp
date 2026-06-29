//===- X86PromoteEVEXForSize.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass promotes VEX-encoded vector memory instructions to their EVEX
// equivalent when doing so makes the instruction strictly smaller. It is the
// inverse of X86CompressEVEX (which only ever shrinks EVEX->VEX/legacy) and runs
// after it so the two can never oscillate.
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
// Safety of the bare setDesc substitution rests on two invariants:
//  1. Operand-layout identity. CompressEVEX performs the identical bare setDesc
//     in the EVEX->VEX direction and is Release-correct; promotion is its exact
//     inverse, so any table pair that is not immediate-value-divergent is
//     layout-symmetric. The only immediate-divergent pairs are exactly those
//     X86CompressEVEX::performCustomAdjustments rewrites; we blocklist them via
//     needsImmAdjustment.
//  2. Feature availability. Compression is downhill in features (the VEX target
//     is always a subset), but promotion is uphill: an EVEX twin may need
//     AVX512DQ/BW/VNNI/IFMA that hasVLX() does not imply. The generated
//     featuresAvailableForPromote() gates each EVEX target against the subtarget.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/X86BaseInfo.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
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

STATISTIC(NumPromoted, "Number of VEX instructions promoted to EVEX for size");

namespace {
// Reuse the generated EVEX<->VEX equivalence table (EVEX->VEX direction) and the
// generated featuresAvailableForPromote() feature gate.
#define GET_X86_COMPRESS_EVEX_TABLE
#include "X86GenInstrMapping.inc"

static unsigned getEncoding(uint64_t TSFlags) {
  return TSFlags & X86II::EncodingMask;
}

// The CD8 scale field (0 if the form is not CDisp8-capable). The access size is
// N = 1 << (field - 1).
static unsigned cd8ScaleField(const MCInstrDesc &Desc) {
  return (Desc.TSFlags & X86II::CD8_Scale_Mask) >> X86II::CD8_Scale_Shift;
}

// EVEX opcodes whose immediate is rewritten when converting to/from VEX (the
// exact set X86CompressEVEX::performCustomAdjustments handles). Their immediate
// has different units/bits between the two encodings, so a bare setDesc would
// miscompile; they must never be promotion targets.
static bool needsImmAdjustment(unsigned EVEXOpc) {
  switch (EVEXOpc) {
  case X86::VALIGNDZ128rri:
  case X86::VALIGNDZ128rmi:
  case X86::VALIGNQZ128rri:
  case X86::VALIGNQZ128rmi:
  case X86::VSHUFF32X4Z256rmi:
  case X86::VSHUFF32X4Z256rri:
  case X86::VSHUFF64X2Z256rmi:
  case X86::VSHUFF64X2Z256rri:
  case X86::VSHUFI32X4Z256rmi:
  case X86::VSHUFI32X4Z256rri:
  case X86::VSHUFI64X2Z256rmi:
  case X86::VSHUFI64X2Z256rri:
  case X86::VRNDSCALEPDZ128rri:
  case X86::VRNDSCALEPDZ128rmi:
  case X86::VRNDSCALEPSZ128rri:
  case X86::VRNDSCALEPSZ128rmi:
  case X86::VRNDSCALEPDZ256rri:
  case X86::VRNDSCALEPDZ256rmi:
  case X86::VRNDSCALEPSZ256rri:
  case X86::VRNDSCALEPSZ256rmi:
  case X86::VRNDSCALESDZrri:
  case X86::VRNDSCALESDZrmi:
  case X86::VRNDSCALESSZrri:
  case X86::VRNDSCALESSZrmi:
  case X86::VRNDSCALESDZrri_Int:
  case X86::VRNDSCALESDZrmi_Int:
  case X86::VRNDSCALESSZrri_Int:
  case X86::VRNDSCALESSZrmi_Int:
    return true;
  default:
    return false;
  }
}

// True if \p EVEXDesc is a memory-operand EVEX form whose VEX twin we may
// promote to. The structural requirements are: EVEX-encoded, has a memory
// operand, and is CDisp8-capable (nonzero CD8 scale -- only then is there a
// displacement-size win). This admits the full family of memory ops in the
// table (moves, binary arith, FMA, scalar, imm8-trailing forms), not just
// moves: operand-layout identity with the VEX twin is guaranteed by the
// symmetry theorem (this is the exact inverse of CompressEVEX's Release-correct
// setDesc), so no per-form operand-count check is needed here.
//
// Two correctness gates are applied elsewhere, not here: immediate-rewrite
// opcodes are blocklisted by needsImmAdjustment, and feature availability of the
// EVEX target by featuresAvailableForPromote -- both in buildInverseMap.
//
// Excluded structurally:
// - non-CDisp8 forms (e.g. moffs _alt) -- CD8 scale is 0, no win,
// - EVEX_B (broadcast/rounding/SAE) forms -- defensive: the table emitter
//   already keeps these out, so this never fires, but it makes the filter
//   self-evidently safe without relying on a generated-file property.
// (gather/scatter and masked/512-bit forms carry EVEX_K/EVEX_L2 and are absent
// from the table entirely.)
static bool isPromotableEVEXMemForm(const MCInstrDesc &EVEXDesc) {
  if (getEncoding(EVEXDesc.TSFlags) != X86II::EVEX)
    return false;
  if (X86II::getMemoryOperandNo(EVEXDesc.TSFlags) < 0)
    return false;
  if (cd8ScaleField(EVEXDesc) == 0) // not CDisp8-capable (e.g. moffs _alt forms)
    return false;
  if (EVEXDesc.TSFlags & X86II::EVEX_B) // broadcast/rounding/SAE -- never a twin
    return false;
  return true;
}

// Build VEX-opcode -> EVEX-opcode by inverting X86CompressEVEXTable, keeping only
// promotable memory-operand pairs that are correct on this subtarget. The table
// maps {OldOpc(EVEX or legacy) -> NewOpc(smaller)}; we key on NewOpc and require
// it be VEX-encoded, which drops the legacy-SSE->VEX rows that share a VEX key.
//
// Three filters reject unsafe targets: isPromotableEVEXMemForm (structural),
// needsImmAdjustment (immediate-value divergence), and featuresAvailableForPromote
// (the EVEX twin's feature requirements vs. this subtarget). The feature gate is
// applied *here*, before tiebreaking, so that a feature-unavailable twin can
// never shadow a legal one (see the collision handling below).
//
// A VEX opcode can map from several byte-equivalent EVEX forms (e.g. VMOVDQUrm
// <- VMOVDQU{8,16,32,64}Z128rm). Among the feature-available survivors we pick
// deterministically by lowest EVEX opcode value -- but only if they agree on the
// CD8 scale N (they must, since N drives the disp8 math); if any two disagree we
// drop the key rather than guess.
static DenseMap<unsigned, unsigned>
buildInverseMap(const X86Subtarget &ST) {
  const X86InstrInfo *TII = ST.getInstrInfo();
  DenseMap<unsigned, SmallVector<unsigned, 2>> Candidates;
  for (const X86TableEntry &E : X86CompressEVEXTable) {
    unsigned EVEXOpc = E.OldOpc;
    unsigned VEXOpc = E.NewOpc;
    if (getEncoding(TII->get(VEXOpc).TSFlags) != X86II::VEX)
      continue;
    if (!isPromotableEVEXMemForm(TII->get(EVEXOpc)))
      continue;
    if (needsImmAdjustment(EVEXOpc)) // immediate units differ EVEX vs VEX
      continue;
    if (!featuresAvailableForPromote(EVEXOpc, &ST)) // EVEX twin needs absent feat
      continue;
    Candidates[VEXOpc].push_back(EVEXOpc);
  }

  DenseMap<unsigned, unsigned> Map;
  for (const auto &KV : Candidates) {
    ArrayRef<unsigned> Twins = KV.second;
    unsigned ScaleField = cd8ScaleField(TII->get(Twins[0]));
    unsigned Best = Twins[0];
    bool Ambiguous = false;
    for (unsigned Opc : Twins) {
      if (cd8ScaleField(TII->get(Opc)) != ScaleField) {
        Ambiguous = true; // twins disagree on N -- refuse to guess
        break;
      }
      Best = std::min(Best, Opc);
    }
    if (!Ambiguous)
      Map[KV.first] = Best;
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

  unsigned f = cd8ScaleField(EVEXDesc);
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

  // Operand-layout identity (the symmetry theorem). Belt-and-suspenders: the
  // proof is that this is the inverse of CompressEVEX's Release-correct setDesc,
  // but assert it cheaply in debug builds before mutating.
  assert(EVEXDesc.getNumOperands() == Desc.getNumOperands() &&
         EVEXDesc.getNumDefs() == Desc.getNumDefs() &&
         "VEX/EVEX promotion pair disagrees on operand layout");

  MI.setDesc(EVEXDesc);
  // CompressEVEX may have tagged this instruction when it shrank it EVEX->VEX
  // earlier; clear that stale comment flag so it doesn't mask ours.
  MI.clearAsmPrinterFlag(X86::AC_EVEX_2_LEGACY);
  MI.clearAsmPrinterFlag(X86::AC_EVEX_2_VEX);
  MI.setAsmPrinterFlag(X86::AC_VEX_2_EVEX);
  ++NumPromoted;
  return true;
}

static bool runOnMF(MachineFunction &MF) {
  const X86Subtarget &ST = MF.getSubtarget<X86Subtarget>();
  if (!ST.hasVLX() || !ST.preferEVEXForSize())
    return false;

  LLVM_DEBUG(dbgs() << "Start X86PromoteEVEXForSizePass\n";);
  DenseMap<unsigned, unsigned> InverseMap = buildInverseMap(ST);
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
