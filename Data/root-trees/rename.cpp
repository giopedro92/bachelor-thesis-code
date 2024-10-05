#if !defined(__CINT__) || defined(__MAKECINT__)

#include <Riostream.h>
#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>

#endif

void Rename() {
    TFile *file = TFile::Open("dataNew_1_2.root", "UPDATE");

    //TFile f("myfile.root","update");
    //TTree *T = (TTree*)f.Get("oldname");
    //T->SetName("newname");
    //T->Write();

    if (file->IsOpen()) {
        TTree *oldTree = (TTree*)file->Get("treeList_0_24_0_24_Sgn");
        if (oldTree) {
            oldTree->SetName("TreeD");
            oldTree->Write();
            file->Write();
        } else {
            std::cerr << "Tree not found!" << std::endl;
        }
        file->Close();
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}