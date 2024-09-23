int main() {
    TFile *file = TFile::Open("your_file.root", "UPDATE");
    if (file->IsOpen()) {
        TTree *oldTree = (TTree*)file->Get("old_tree_name");
        if (oldTree) {
            oldTree->SetName("new_tree_name");
            file->Write();
        } else {
            std::cerr << "Tree not found!" << std::endl;
        }
        file->Close();
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}