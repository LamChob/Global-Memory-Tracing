#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include "clang/Rewrite/Core/Rewriter.h"

#include <iostream>
#include <fstream>

using namespace clang;

namespace {

  class KernelVisitor : public RecursiveASTVisitor<KernelVisitor> {
  private:
      ASTContext *astContext; // used for getting additional AST info
		  Rewriter &rewriter;
		  std::string augmentedSrcFile = "augmentedSource.cu";

  public:
      explicit KernelVisitor(CompilerInstance *CI, Rewriter &rewriter, std::string saveTo)
        : astContext(&(CI->getASTContext())), rewriter(rewriter), augmentedSrcFile(saveTo) // initialize private members
      {
          //rewriter.setSourceMgr(astContext->getSourceManager(), astContext->getLangOpts());
      }

      virtual ~KernelVisitor() {
	      /*if(rewriter.overwriteChangedFiles())
			      std::cerr <<"success!";
	      else {
			      std::cerr << rewriter.getSourceMgr().getDiagnostics().getFlagValue().str();
	      }*/
	      std::cerr << "rewrite:\n";
	      std::string output;
	      llvm::raw_string_ostream result(output);
	      rewriter.getEditBuffer(rewriter.getSourceMgr().getMainFileID()).write(result);
	      result.flush();
	      //std::cerr << output << "\n";

	      // Write new source file to disk
	      std::ofstream out(augmentedSrcFile);
          out << "#include \"TraceUtils.h\"\n";
          out << "#include \"DeviceUtils.h\"\n";
	      out << output;
	      out.close();
      }

      virtual bool VisitFunctionDecl(FunctionDecl *func) {
        auto& SM = astContext->getSourceManager();
        
		    std::string funcName = func->getNameInfo().getName().getAsString();

            if (funcName == "main") {
                  //llvm::errs() << "** Found Main:\n";
            }
            if( SM.isInMainFile(func->getLocStart()) ) {
                for( auto attr : func->attrs()) {
                    if (attr == nullptr)
                            break;
                    if(std::strcmp( attr->getSpelling(),"global")==0 || std::strcmp(attr->getSpelling(), "device") == 0) {
                //        llvm::errs() << "** Found Func: " << funcName << "\n";
                        if (func->param_size() == 0) continue;
                        auto param = *(func->param_begin()+func->param_size()-1);

                        // TODO check if name blk_offset is already used
                        rewriter.InsertTextAfterToken(param->getLocation(), 
                            ", uint32_t* __inx1, uint32_t* __inx2, uint64_t* __dbuff, uint32_t __max_n, int __ns"
                        );
                    }
            }
        }

        return true;
      }

      virtual bool VisitStmt(Stmt *st) {
        auto& SM = astContext->getSourceManager();

		      if (CUDAKernelCallExpr *ce = dyn_cast_or_null<CUDAKernelCallExpr>(st)) {
				      //llvm::errs() << "** Found kernel call:\n";
                    auto launch = ce->getConfig();

                    std::string streamName("");
                    std::string kName = "";
                    auto callee = ce->getDirectCallee();
                    if (callee) {
                        kName = callee->getNameInfo().getName().getAsString();
                    }
                    else { // templates different Name resolve
                        auto ule = dyn_cast<UnresolvedLookupExpr>(ce->getCallee());
                        kName = ule->getName().getAsString();
                    }

                    if (dyn_cast<CXXDefaultArgExpr>(launch->getArg(3))) {
                        streamName = "NULL";
                    } else if ( dyn_cast<ImplicitCastExpr>(launch->getArg(3) )) {
                        if (auto ND = dyn_cast<DeclRefExpr>(launch->getArg(3)->IgnoreImpCasts()) ) 
                            streamName = ND->getFoundDecl()->getNameAsString();
                        else 
                            std::cerr << "Error: Missing Stream Declaration!\n";
                    } else {
                            std::cerr << "Error: Unknow Stream Declaration!\n";
                    }
                    if (streamName.compare("") == 0) {
                        return false;
                    }

                    std::string preLaunch;
                    preLaunch.append("\t__t->createStream(");
                    preLaunch.append(streamName);
                    preLaunch.append(");\n\tcudaStreamAddCallback(");
                    preLaunch.append(streamName);
                    preLaunch.append(", trace_start, (void*) \"");
                    preLaunch.append(kName);
                    preLaunch.append("\", 0);\n");
                    rewriter.InsertTextBefore(ce->getLocStart(), preLaunch);

                    auto param = *(ce->arg_begin()+ce->getNumArgs()-1);
                    rewriter.InsertTextAfterToken(param->getLocEnd(), 
                        ", __t->getFrontInxArg(), __t->getBackInxArg(), __t->getTraceBuffArg(), __t->getSlotSizeArg(), __t->getSlotPow2Arg()"
                    );

                    std::string postLaunch(";\ncudaStreamAddCallback(");
                    postLaunch.append(streamName);
                    postLaunch.append(", trace_stop, (void*) __t, 0);\n");
                    rewriter.InsertTextAfterToken(ce->getLocEnd(), postLaunch);
		      } else if ( auto call = dyn_cast_or_null<CallExpr>(st) ) { // is device function
                    if ( auto func = dyn_cast_or_null<FunctionDecl>(call->getDirectCallee()) ) {
                        if( SM.isInMainFile(func->getLocStart()) ) {
                        for( auto attr : func->attrs()) {
                            if (attr == nullptr)
                                    break;
                            std::cerr << "Found FCALL\n";
                            if(std::strcmp(attr->getSpelling(), "device") == 0) {
                                auto param = *(call->arg_begin()+call->getNumArgs()-1);
                                rewriter.InsertTextAfterToken(param->getLocEnd(), 
                                    ", __inx1, __inx2, __dbuff, __max_n, __ns"
                                );
                            }
                        }
                    }
                }
              }
          return true;
      }

  };

  class AugmentKernelASTConsumer : public ASTConsumer {
    CompilerInstance &Instance;
    std::set<std::string> ParsedTemplates;
	Rewriter rewriter;
    KernelVisitor visitor;

  public:
    AugmentKernelASTConsumer(CompilerInstance &Instance, std::set<std::string> ParsedTemplates, std::string saveTo)
        : Instance(Instance), ParsedTemplates(ParsedTemplates), rewriter(Instance.getSourceManager(), Instance.getLangOpts()), visitor(&Instance, rewriter, saveTo) {}

    // override this to call our KernelVisitor on the entire source file
    virtual void HandleTranslationUnit(ASTContext &Context) {
        /* we can use ASTContext to get the TranslationUnitDecl, which is
             a single Decl that collectively represents the entire source file */
		    //Context.PrintStats();
        visitor.TraverseDecl(Context.getTranslationUnitDecl());
    }
  };

  class AugmentCudaKernelAction : public PluginASTAction {
    std::set<std::string> ParsedTemplates;
	  std::string augmentedSrcFile = "augmentedSource.cu";

  public:
		  ~AugmentCudaKernelAction() {

		  }

  protected:
    std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI, llvm::StringRef) override {
      return llvm::make_unique<AugmentKernelASTConsumer>(CI, ParsedTemplates, augmentedSrcFile);
    }

    bool ParseArgs(const CompilerInstance &CI, const std::vector<std::string> &args) override {
	    for(const std::string & arg : args)
		    std::cerr << "argument: " << arg << "\n";

	    auto itr = std::find(args.begin(), args.end(), "-f");
	    if (itr != args.end() && ++itr != args.end())
			    augmentedSrcFile = *itr;

      if (!args.empty() && args[0] == "help")
        PrintHelp(llvm::errs());

      return true;
    }

    void PrintHelp(llvm::raw_ostream& ros) {
      ros << "Help for plugin goes here\n";
    }

  };

}

static FrontendPluginRegistry::Add<AugmentCudaKernelAction>
X("cuda-aug", "find and augment cuda kernel");
