#include "llvm/Pass.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/Debug.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"


#include "llvm/PassRegistry.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/Transforms/IPO/PassManagerBuilder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/IR/InlineAsm.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <string>
#include <stack>
#include <set>

// Required for DEBUG macro (and used for filtering logs)
// DEBUG info is only present in llvm builds with enabled assertions
#define DEBUG_TYPE "memtracepass"

constexpr int SHARED_MEMORY {3};

using namespace llvm;

unsigned int getTypeSizeInBytes(Type*);
unsigned int getStructSizeInBytes(StructType*);

typedef std::vector<Value*> GMemOps;
typedef std::map<Function*, GMemOps  > GMemTable;
static std::map<PHINode*, unsigned int> phiMap;
static std::map<SelectInst*, unsigned int> selectMap;
static std::map<CallInst*, std::set<unsigned int>> fCallMap;
static std::map<Function*, std::map<std::string, Function*> > sFuncTable;

enum TraceType {
    Load    = 0,        // 0000
    Store   = 1  << 28, // 0001
    AAdd    = 2  << 28, // 0010
    ASub    = 3  << 28, // 0011
    AExch   = 4  << 28, // 0100
    AMin    = 5  << 28, // 0101
    AMax    = 6  << 28, // 0110
    AInc    = 7  << 28, // 0111
    ADec    = 8  << 28, // 1000
    ACAS    = 9  << 28, // 1001
    AAnd    = 10 << 28, // 1010
    AOr     = 11 << 28, // 1011
    AXor    = 12 << 28, // 1100
    Unknown = 13 << 28, // 1101
};



GlobalVariable *addGlobalString(Module &M, Twine Name, StringRef Data) {
    // Create a constant array that contains our data
    Constant *StrConstant = ConstantDataArray::getString(M.getContext(), Data);
    // Add a named global variable definition to the module
    GlobalVariable *GV = new GlobalVariable(M, StrConstant->getType(),
                                            true, GlobalValue::ExternalLinkage,
                                            StrConstant, Name
                                            );
    return GV;
}

GlobalVariable *addGlobalInt(Module &M, Twine Name, int Data, unsigned int AddrSpace=0) {

    // Create a constant integer
    IntegerType* it = IntegerType::get(M.getContext(), 32);
    ConstantInt *IntConstant = ConstantInt::get(it, Data, false);

    // Add a named global variable definition to the module
    GlobalVariable *GV = new GlobalVariable(M, IntConstant->getType(),
                                            false, GlobalValue::ExternalLinkage,
                                            IntConstant, Name, nullptr, 
                                            GlobalValue::NotThreadLocal,
                                            AddrSpace
                                            );
    return GV;
}

unsigned int getTypeSizeInBytes(Type *Ty) {
    if ( ArrayType::classof(Ty) || VectorType::classof(Ty) || PointerType::classof(Ty) ) {
        return Ty->getScalarSizeInBits()/8;
    } else if ( StructType::classof(Ty) )    {
        return getStructSizeInBytes( dyn_cast<StructType>(Ty) );
    } else {
        return Ty->getPrimitiveSizeInBits()/8;
    }
}

unsigned int getStructSizeInBytes(StructType *S) {
    unsigned int Size = 0;
    for( auto Element : S->elements()) {
        Size += getTypeSizeInBytes(Element);
    }
    return Size;
}


TraceType getAtomicType(CallInst* ca) {
    std::string FuncName = ca->getName();

    //errs() << FuncName << "\n";
    if (FuncName.find("atomicAdd", 0) != std::string::npos) {
        return TraceType::AAdd;
    }
    else if (FuncName.find("atomicSub", 0) != std::string::npos ) {
        return TraceType::ASub;
    }
    else if (FuncName.find("atomicExch", 0) != std::string::npos ) {
        return TraceType::AExch;
    }
    else if (FuncName.find("atomicMin", 0) != std::string::npos ) {
        return TraceType::AMin;
    }
    else if (FuncName.find("atomicMax", 0) != std::string::npos ) {
        return TraceType::AMax;
    }
    else if (FuncName.find("atomicInc", 0) != std::string::npos ) {
        return TraceType::AInc;
    }
    else if (FuncName.find("atomicDec", 0) != std::string::npos ) {
        return TraceType::ADec;
    }
    else if (FuncName.find("atomicCAS", 0) != std::string::npos ) {
        return TraceType::ACAS;
    }
    else if (FuncName.find("atomicAnd", 0) != std::string::npos ) {
        return TraceType::AAnd;
    }
    else if (FuncName.find("atomicOr", 0) != std::string::npos ) {
        return TraceType::AOr;
    }
    else if (FuncName.find("atomicXor", 0) != std::string::npos ) {
        return TraceType::AXor;
    } else {
        errs() << "Unknown Atomic Atomic: " << FuncName << "\n";
        return TraceType::Unknown;
    }
}

Function* getOrCreateSpecialization(Module& M, Function* original, std::set<unsigned int> argInx) {
    std::string postfix;
    std::string newName;
    SmallVector<ReturnInst*, 8> returns;

    for (auto e : argInx )
        postfix += std::to_string(e);

    if (sFuncTable[original][postfix])
        return sFuncTable[original][postfix];

    newName = original->getName();
    newName += postfix;

    ValueToValueMapTy VMap;

    auto clone = CloneFunction(original, VMap);
    clone->setName(newName);
    sFuncTable[original][postfix] = clone;
    errs() << original->getName() << " => " << clone->getName() << "\n";
     
    return clone;
}

bool isMemOp(Value* v) {
    bool rval = false;
    if (dyn_cast<LoadInst>(v)) {
        rval = true;
    } else if (dyn_cast<StoreInst>(v)) {
        rval = true;
    } else if (auto FuncCall = dyn_cast<CallInst>(v)) {
        if ( FuncCall->hasName() ) {
            auto FuncName = FuncCall->getCalledFunction()->getName();
            if ( FuncName.find("atomic", 0) != std::string::npos ) {
                rval = true;
            }
        }
    }
    return rval;
}

bool isParent(Value* v) {
    // Check if  this can generate values that can be used for GMemOp
    // PtrLoad
    bool rval = false;
    if (auto li = dyn_cast<LoadInst>(v)) {
        if (PointerType::classof(li->getType())) rval = true;
    } else if ( dyn_cast<GetElementPtrInst>(v)) {
        rval = true;
    } else if ( auto pn =  dyn_cast<PHINode>(v) ){
        if (PointerType::classof(v->getType())) {
            phiMap[pn] += 1; // count
        }
    } else if ( auto s = dyn_cast<SelectInst>(v) ) {
        if (PointerType::classof(v->getType())) {
            selectMap[s] += 1; // count
        }
    } else if ( auto cast = dyn_cast<CastInst>(v)) {
        if (   PointerType::classof(cast->getDestTy()) 
            || PointerType::classof(cast->getDestTy()) ) rval = true;
    } else if ( auto cast = dyn_cast<CallInst>(v) ) {
        if (auto func = cast->getCalledFunction() ) {
            if (PointerType::classof(func->getReturnType()))
                errs() << "Function Retruning Ptr found, not resolvable!\n";
        }
    }

    return rval;
}

Value* isFCall(Value* v, Value* def) {
    // Check if this is a function call and return origin pointer inside
    // the function
    Value* rval = nullptr; 
    if (auto call = dyn_cast<CallInst>(v) ) {
        auto callee = call->getCalledFunction();
        if ( callee->doesNotAccessMemory() ) // Easy check, but can this even happen???
            return rval;
        if (callee->arg_empty() || callee->arg_size() < 4) {
            errs() << "Not enough args for Tracing in Kernel: " 
                <<  callee->getName() << " trying next kernel\n";
        } else {
            int cnt = 0;
            for(auto& arg : call->arg_operands()) {
                if (arg == def) break;
                ++cnt;
                //arg->dump();
            }
            fCallMap[call].insert(cnt); // remember for each call which arguments were used
            auto arg_it = callee->arg_begin();
            for (int i = 0; i < cnt; i++) *arg_it++;
            rval = &*arg_it;
        }
    }
    return rval;
}

std::vector<Instruction*> getDescendants(Value* origin) {
    std::vector<Instruction*> descendants;
    std::stack<Value*> workStack;    
    workStack.push(origin);
    while( workStack.size() > 0) {
        auto element = workStack.top();
        workStack.pop();
        for (auto e : element->users() ) {
            // Apply Transfer Function that classifies Op in Lattice
            // => if we need to trace
            if ( isMemOp(e) ) {  
                descendants.push_back(dyn_cast<Instruction>(e)); // we know this is an instruction
            } 
            if ( isParent(e) ) { // Check if this creates new descendants
                workStack.push(e);
            }
            // go into device Functions that are Traceble and add them to the origin stack
            if ( auto v = isFCall(e, element) ) {
                // TODO Handle Recursing Functions...
                workStack.push(v);
            }
        } 
    }  
    return descendants;
}

std::vector<Value*> getPointers(Function* F) {
    std::vector<Value*> Ptrs;
    for(auto& arg : F->args() ) {
        if (PointerType::classof(arg.getType())) 
            Ptrs.push_back(&arg);
    }
    for (auto& arg : M.get_globals()) {
        auto ty = arg.getType();
        if (PointerType::classof(ty)) 
            ty.dump();
            if ( ty.getAddressSpace == 0) {
                Ptrs.push_back(&arg);
            }
    }
    // Minus the number of pointers introduced by the tracing augmentation
    std::vector<Value*> rval(Ptrs.begin(), Ptrs.begin() + Ptrs.size() - 3);
    return rval;
}

GMemTable generateGlobalMemoryOperationTable(Module& M, std::vector<Function*> kernels)  {
    std::vector<Value*> memOpList;
    GMemTable table;

    for(auto func : kernels) {
        // get cmd parameters for tracing
        // errs() << f->getName() << "\n";
        auto originalPointers = getPointers(func); 
        
        // get all descendants of the pointers    
        for (auto p : originalPointers) {
            auto descendants = getDescendants(p);
            for (auto d : descendants) {
                auto homeFunction = d->getFunction();
                if (std::count(table[homeFunction].begin(), table[homeFunction].end(), d) == 0)
                    table[homeFunction].push_back(d); 

            }
        }
    }
    // Now that we have possible candidates, we have to make sure that there
    // are no ambigiouities in the address spaces at certain points
    for ( auto e : phiMap ) {
        if (e.first->getNumIncomingValues() != e.second) {
            errs() << "Ambigious Address Space in PhiNode found, aborting!\n";
            table.clear();
        }
    }
    for ( auto e : selectMap ) {
        if (e.second != 2) {
            errs() << "Ambigious Address Space in SelectInst found, aborting!\n";
            table.clear();
        }
    }
    
    // create specialized functions and replace all function calls
    std::set<Value*> newPointers;
    errs() << "------------------------------------------\nSpecialization:\n------------------------------------------\n";
    for ( auto call : fCallMap ) {
        // remove old home functions from  tables
        if (table.count(call.first->getCalledFunction()))
            table.erase(call.first->getCalledFunction());

        auto sfunc = getOrCreateSpecialization(M, call.first->getCalledFunction(), call.second);
        call.first->setCalledFunction(sfunc);

        for (auto inx : call.second )
            newPointers.insert(call.first->getArgOperand(inx));
    }
    
    // re-analyse the specialized functions
    for (auto p : newPointers ) {
        auto descendants = getDescendants(p);
        for (auto d : descendants) {
            auto homeFunction = d->getFunction();
            if (std::count(table[homeFunction].begin(), table[homeFunction].end(), d) == 0)
                table[homeFunction].push_back(d); 

        }
    }

    

    return table;
}

Value* insertSmidAsm(Module &M, Function* F) {
    IntegerType* I32Type = IntegerType::get(M.getContext(), 32);
    IntegerType* I64Type = IntegerType::get(M.getContext(), 64);
    StringRef asm_op("mov.u32 $0, %smid;");
    StringRef asm_constraint("=r");
    FunctionType *asm_ftype = FunctionType::get(I32Type, false);
    InlineAsm *smid_asm = InlineAsm::get(asm_ftype,
        asm_op,
        asm_constraint,
        false,
        InlineAsm::AsmDialect::AD_ATT
    );
    IRBuilder<> SmidBuilder(&(F->front().front()));
    return SmidBuilder.CreateCall(smid_asm);
}

Value* insertLaneAsm(Module &M, Function* F) {
    IntegerType* I32Type = IntegerType::get(M.getContext(), 32);
    IntegerType* I64Type = IntegerType::get(M.getContext(), 64);
    StringRef asm_op("mov.u32 $0, %laneid;");
    StringRef asm_constraint("=r");
    FunctionType *asm_ftype = FunctionType::get(I32Type, false);
    InlineAsm *lane_asm = InlineAsm::get(asm_ftype,
        asm_op,
        asm_constraint,
        false,
        InlineAsm::AsmDialect::AD_ATT
    );
    IRBuilder<> LaneBuilder(&(F->front().front()));
    return LaneBuilder.CreateCall(lane_asm);
}

std::vector<Function*> getKernelFunctions(Module &M) {
    std::vector<Function*> Kernels;
    NamedMDNode * kernel_md = M.getNamedMetadata("nvvm.annotations");
    if (kernel_md) {
        // MDNodes in NamedMDNode
        for (const MDNode *node : kernel_md->operands()) {
            //node->dump();
            // MDOperands in MDNode
            for (const MDOperand &op : node->operands()) {
                Metadata * md = op.get();
                if(ValueAsMetadata* v = dyn_cast_or_null<ValueAsMetadata>(md)) {
                    //v->dump();
                    if (Function* f = dyn_cast<Function>(v->getValue())) {
                        //f->dump();
                        Kernels.push_back(f);
                    }
                }
            }
        }
    }
    return Kernels;
}

bool insertTracing (Module &M) { 
    IntegerType* I64Type   = IntegerType::get(M.getContext(), 64);
    IntegerType* I32Type   = IntegerType::get(M.getContext(), 32);
    ConstantInt *Increment = ConstantInt::get(I32Type, 1, false);
    ConstantInt *TwoInc    = ConstantInt::get(I64Type, 2, false);

    auto Functions = getKernelFunctions(M);
    if ( Functions.size() == 0 ) {
        report_fatal_error("Error: Module does not have any kernel metadata");
        return false;
    }

    Function* TraceCall;
    for (auto& F : M.functions()) {
        if (F.getName().find("__mem_trace", 0) == std::string::npos) continue;
        TraceCall = &F;
        break;
    }
    if (!TraceCall) {
        errs() << "Critical Error: Missing trace function!\n";
        return false;
    }
    

    auto MemOpsTable = generateGlobalMemoryOperationTable(M,Functions);
    if (MemOpsTable.empty()) { // abort in case of empty table
        errs() << "Aborting tracing Instrumentation\n";
        return false;
    } 
    errs() << "------------------------------------------\nInstrumentation:\n------------------------------------------\n";
    for ( auto Entry : MemOpsTable) {
        auto F = Entry.first;
        std::vector<Instruction*> Insts;
        /*for (BasicBlock& bb: F->getBasicBlockList() ) {
            for (Instruction& inst : bb) {
                Insts.push_back(&inst);
            }
        }*/
        // local info counters
        int LoadCounter = 0;
        int AtomicCounter = 0;
        int StoreCounter = 0;
        errs() << F->getName() << " ";

        // get cmd parameters for tracing
        if (F->arg_empty() || F->arg_size() < 4) {
            errs() << "Not enough args for Tracing in Kernel: " 
                <<  F->getName() << " trying next kernel\n";
            continue;
        } 

        Function::arg_iterator KernelArg = F->arg_end();

        KernelArg--;
        auto SlotPow     = &*KernelArg--;
        auto MaxInx      = &*KernelArg--;
        auto DataBuff    = &*KernelArg--;
        auto IndexArray2 = &*KernelArg--;
        auto IndexArray1 = &*KernelArg;
        
        auto lan = SlotPow->getName();

        // check if this a function we augmented
        if(lan.find("__ns") == std::string::npos) continue;
        IRBuilder<> FStartBuilder(&(F->front().front()));

        auto Blacklist = generateShmemBlacklist(F);
        auto Smid = insertSmidAsm(M,F);
        auto Lane = insertLaneAsm(M,F);

    
        // shift descriptor
        auto WideSmid = FStartBuilder.CreateZExtOrBitCast(Smid, I64Type, "desc");
        auto Desc = FStartBuilder.CreateShl(WideSmid, 32);
        

        // Get Buffer Segment based on SMID and Load the Pointer
        auto NSlots    = FStartBuilder.CreateShl(Increment, SlotPow, "n_slots");
        auto SlotMask  = FStartBuilder.CreateSub(NSlots, Increment, "slot_mask");
        auto IndexSlot = FStartBuilder.CreateAnd(Smid, SlotMask);

        for (auto v : Entry.second) {
            Instruction* inst = dyn_cast<Instruction>(v);
            Value *PtrOperand, *LDesc;
            uint64_t ValueSize;
            IRBuilder<> builder(inst);
            if (auto li = dyn_cast<LoadInst>(inst)) {
                PtrOperand = li->getPointerOperand();
                if (Blacklist[li] == 1) {     
                 //   errs() << "L hit smem array: " << PtrOperand->getName() << "\n";
                    continue;
                }
                if ( auto ConstPtrOp = dyn_cast<ConstantExpr>(PtrOperand)) {
                    if (ConstPtrOp->isCast()) {    
                        //errs() << "L hit non-array\n";     
                        ConstPtrOp->dump();
                        continue;
                    }
                }
                LDesc = builder.CreateOr(Desc, (uint64_t)TraceType::Load );
                ValueSize = getTypeSizeInBytes(li->getType());
                 
                LoadCounter++;
             } else if (auto si = dyn_cast<StoreInst>(inst)) {
                PtrOperand = si->getPointerOperand();
                if (Blacklist[PtrOperand] == 1 || Blacklist[si] == 1 ) {     
                  //  errs() << "S hit array: " << PtrOperand->getName() << "\n";
                    continue;
                }
                if ( auto ConstPtrOp = dyn_cast<ConstantExpr>(PtrOperand)) {
                    if (ConstPtrOp->isCast()) {     
                       // errs() << "S hit non-array\n"; 
                        ConstPtrOp->dump(); 
                        continue;    
                    }
                }
                    LDesc = builder.CreateOr(Desc, (uint64_t)TraceType::Store );
                    ValueSize = getTypeSizeInBytes(si->getValueOperand()->getType());

                // Build Descriptor
                StoreCounter++;
            } else if (auto FuncCall = dyn_cast<CallInst>(inst)) {
                // functions without name cannot be atomics
                if ( !FuncCall->hasName() ) continue; 
                Value* FirstArg;
                auto FuncName = FuncCall->getCalledFunction()->getName();
                if ( FuncName.find("atomic", 0) != std::string::npos ) {
                    FirstArg = FuncCall->getArgOperand(0);
                    if (Blacklist[FirstArg] == 1) {
                       // errs() << "Atomic hit array\n";
                        continue;
                    }
                    if ( auto ConstOpPtr = dyn_cast<ConstantExpr>(FirstArg)) {
                        if (ConstOpPtr->isCast()) { 
                      //      errs() << "Atomic hit non-array\n"; 
                            ConstOpPtr->dump();
                            continue;
                        }
                    }
                    PtrOperand = FuncCall->getArgOperand(0);
                    LDesc      = builder.CreateOr(Desc, (uint64_t)getAtomicType(FuncCall));
                    ValueSize  = getTypeSizeInBytes(FuncCall->getArgOperand(1)->getType());
                    AtomicCounter++;
                } else {
                    continue;
                }
            } else { continue;}
            // Add tracing
            LDesc = builder.CreateOr(LDesc, (uint64_t) ValueSize);
            auto PtrToStore = builder.CreatePtrToInt(PtrOperand, I64Type);
            builder.CreateCall(TraceCall, {DataBuff, IndexArray1,IndexArray2 , MaxInx, LDesc, PtrToStore, Lane, IndexSlot});
        }
        errs() << "\tL: " << LoadCounter << " S: " << StoreCounter << " A: " << AtomicCounter << "\n";
    }
    return true;
}

// Needs to be a ModulePass because we modify the global variables.
// Implemented as struct because its members are public by default.
// Read the LLVM Coding Standards to learn why that is sane in this case.
struct TracePass : public ModulePass {
    // Static variable containing the pass id, strictly necessary
    static char ID;
    // Constructor only delegates to ModulePass constructor
    TracePass() : ModulePass(ID) {}


    bool runOnModule(Module &M) override {
        // no CUDA module 
        if(M.getTargetTriple().find("nvptx64") == std::string::npos) {
            return false;
        }

        return insertTracing(M);
    }

};

// Initialize pass id, needs to be outside of class because c++
char TracePass::ID = 0;
// Register pass, so that it can be invoked by adding -memtracepass, also give a short description
// for opt -help and tell the pass manager that this is not an analysis pass
static RegisterPass<TracePass> X("memtrace-pass", "includes static and dynamic load/store counting", false, false);

// This enables Autoregistration of the Pass
static void registerTracePass(const PassManagerBuilder &,legacy::PassManagerBase &PM) {
    PM.add(new TracePass);
}
static RegisterStandardPasses RegisterTracePass(PassManagerBuilder::EP_OptimizerLast, registerTracePass);
static RegisterStandardPasses RegisterTracePass0(PassManagerBuilder::EP_EnabledOnOptLevel0, registerTracePass);
