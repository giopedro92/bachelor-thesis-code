int main() {
    TFile *file = TFile::Open("", "UPDATE");
    if (file->IsOpen()) {
        file->Delete("tree_name;*");
        file->Write();
        file->Close();
    } else {
        std::cerr << "Error opening file!" << std::endl;
    }
}