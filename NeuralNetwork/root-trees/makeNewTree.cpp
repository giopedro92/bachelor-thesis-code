#if !defined(__CINT__) || defined(__MAKECINT__)

#include <Riostream.h>
#include <TFile.h>
#include <TROOT.h>
#include <TTree.h>

#endif

void makeNewTree() {
  // importa dai 2 file .root
  TFile *fileS            = new TFile("./signalNew_1_2.root");
  TFile *fileB            = new TFile("./background_1_2.root");
  
  // crea 2 TREE
  TTree *oldtreeS            = (TTree *)fileS->Get("treeList_0_24_0_24_Sgn");
  TTree *oldtreeB            = (TTree *)fileB->Get("treeList_0_24_0_24_Sgn");

  // crea nuovo file .root
  TFile *newfile          = new TFile("sgn_bkg_simul.root", "recreate");
  
  // per crare nuovi TREE
  // TTree *newtreeS      = new TTree("newtreeS", "newtreeS");
  // TTree *newtreeB      = new TTree("newtreeB", "newtreeB");

  // const char *newnameS = "newtreeS";
  // const char *newnameB = "newtreeB";

  // clonare la struttra dei TREE nel nuovo file .root
  TTree *treeS         = oldtreeS->CloneTree(0);
  TTree *treeB         = oldtreeB->CloneTree(0);
  treeS->SetName("treeSgn");
  treeB->SetName("treeBkg");

  /* Float_t massK0S_sig, tImpParBach_sig; */
  /* Float_t massK0S_bkg, tImpParBach_bkg; */
  // ...

  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("tImpParBach", &tImpParBach_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree1->SetBranchAddress("massK0S", &massK0S_sig); */

  /* tree2->SetBranchAddress("massK0S", &massK0S_bkg); */
  /* tree2->SetBranchAddress("tImpParBach", &tImpParBach_bkg); */
  /* tree->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */
  /* tree2->SetBranchAddress("massK0S", &massK0S_sig); */

  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "tImpParBach", &tImpParBach, "tImpParBach/F"); */
  /* newtreeS->Branch( "tImpParV0", &tImpParV0, "tImpParV0/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */
  /* newtreeS->Branch( "massK0S", &massK0S, "massK0S/F"); */

  /* newtreeB->Branch( "massK0S", &massK0S, "massK0S/F"); */
  // ...

  // riempio il treeS
  for (Long64_t i = 0; i < oldtreeS->GetEntries(); i++) {
    if (i % 100000 == 0) Printf("processing event = %d", i);
    oldtreeS->GetEntry(i);

    treeS->Fill();
  }

  treeS->Write();

  // riempio il treeB
  for (Long64_t i = 0; i < oldtreeB->GetEntries(); i++) {
    if (i % 100000 == 0) Printf("processing event = %d", i);
    oldtreeB->GetEntry(i);

    treeB->Fill();
  }

  treeB->Write();

  delete newfile;
}