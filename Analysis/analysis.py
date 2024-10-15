import uproot
import numpy             as np
import matplotlib.pyplot as plt
import os


'''---------------------DATA LOADING-------------------------------------------'''
# Creates the directory if it does not exist
output_folder = "plots"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Uploading the root file and trees
dataS = uproot.open('./root-trees/signalNew_1_2.root')
dataB = uproot.open('./root-trees/background_1_2.root')
treeS = dataS["treeList_0_24_0_24_Sgn"]
treeB = dataB["treeList_0_24_0_24_Sgn"]


# Extracting the varibales from the signal and background trees
massK0S_sig     = treeS["massK0S"].array()
tImpParBach_sig = treeS["tImpParBach"].array()
tImpParV0_sig   = treeS["tImpParV0"].array()
CtK0S_sig       = treeS["CtK0S"].array()
cosPAK0S_sig    = treeS["cosPAK0S"].array()
nSigmapr_sig    = treeS["nSigmapr"].array()
dcaV0_sig       = treeS["dcaV0"].array()

massK0S_bkg     = treeB["massK0S"].array()
tImpParBach_bkg = treeB["tImpParBach"].array()
tImpParV0_bkg   = treeB["tImpParV0"].array()
CtK0S_bkg       = treeB["CtK0S"].array()
cosPAK0S_bkg    = treeB["cosPAK0S"].array()
nSigmapr_bkg    = treeB["nSigmapr"].array()
dcaV0_bkg       = treeB["dcaV0"].array()


# Converting the tree arrays to numpy arrays
massK0S_sig_np     = massK0S_sig.to_numpy()
tImpParBach_sig_np = tImpParBach_sig.to_numpy()
tImpParV0_sig_np   = tImpParV0_sig.to_numpy()
CtK0S_sig_np       = CtK0S_sig.to_numpy()
cosPAK0S_sig_np    = cosPAK0S_sig.to_numpy()
nSigmapr_sig_np    = nSigmapr_sig.to_numpy()
dcaV0_sig_np       = dcaV0_sig.to_numpy()

massK0S_bkg_np     = massK0S_bkg.to_numpy()
tImpParBach_bkg_np = tImpParBach_bkg.to_numpy()
tImpParV0_bkg_np   = tImpParV0_bkg.to_numpy()
CtK0S_bkg_np       = CtK0S_bkg.to_numpy()
cosPAK0S_bkg_np    = cosPAK0S_bkg.to_numpy()
nSigmapr_bkg_np    = nSigmapr_bkg.to_numpy()
dcaV0_bkg_np       = dcaV0_bkg.to_numpy()


'''-------------------------------HISTOGRAMS--------------------------------------------'''
# Plotting the histogram for all the variables
plt.figure(figsize=(15, 10))

plt.subplot(2, 4, 1)
plt.hist(massK0S_sig, bins=50, alpha=0.5, density=True, label='Signal')
plt.hist(massK0S_bkg, bins=50, alpha=0.5, density=True, label='Background')
plt.xlabel('massK0S')
plt.ylabel('Counts')
#plt.legend()

plt.subplot(2, 4, 2)
plt.hist(tImpParBach_sig, bins=50, alpha=0.5, density=True, label='Signal')
plt.hist(tImpParBach_bkg, bins=50, alpha=0.5, density=True, label='Background')
plt.xlabel('var2tImpParBach')
plt.ylabel('Counts')
plt.yscale('log')
#plt.legend()

plt.subplot(2, 4, 3)
plt.hist(tImpParV0_sig, bins=50, alpha=0.5, density=True, label='Signal')
plt.hist(tImpParV0_bkg, bins=50, alpha=0.5, density=True, label='Background')
plt.xlabel('tImpParV0')
plt.ylabel('Counts')
plt.yscale('log')
#plt.legend()

plt.subplot(2, 4, 4)
plt.hist(CtK0S_sig, bins=50, alpha=0.5, density=True, label='Signal')
plt.hist(CtK0S_bkg, bins=50, alpha=0.5, density=True, label='Background')
plt.xlabel('CtK0S')
plt.ylabel('Counts')
plt.yscale('log')
plt.legend()

plt.subplot(2, 4, 5)
plt.hist(cosPAK0S_sig, bins=50, alpha=0.5, density=True, label='Signal')
plt.hist(cosPAK0S_bkg, bins=50, alpha=0.5, density=True, label='Background')
plt.xlabel('cosPAK0S')
plt.ylabel('Counts')
plt.yscale('log')
#plt.legend()

plt.subplot(2, 4, 6)
plt.hist(nSigmapr_sig, bins=50, alpha=0.5, density=True, label='Signal')
plt.hist(nSigmapr_bkg, bins=50, alpha=0.5, density=True, label='Background')
plt.xlabel('nSigmapr')
plt.ylabel('Counts')
#plt.legend()

plt.subplot(2, 4, 7)
plt.hist(dcaV0_sig, bins=50, alpha=0.5, density=True, label='Signal')
plt.hist(dcaV0_bkg, bins=50, alpha=0.5, density=True, label='Background')
plt.xlabel('dcaV0')
plt.ylabel('Counts')
plt.yscale('log')
#plt.legend()

# Saves figure on the file
plt.savefig(os.path.join(output_folder, "vars_histogram.svg"))
plt.tight_layout()
#plt.show()


'''-----------------------------CORRELATION MATRIXES-----------------------------------'''
# Put the numpy arrays of each variable into columns and then calculate the transposed
X_signal = np.vstack((massK0S_sig_np,
                      tImpParBach_sig_np,
                      tImpParV0_sig_np,
                      CtK0S_sig_np,
                      cosPAK0S_sig_np,
                      nSigmapr_sig_np,
                      dcaV0_sig_np)).T
X_background = np.vstack((massK0S_bkg_np,
                          tImpParBach_bkg_np,
                          tImpParV0_bkg_np,
                          CtK0S_bkg_np,
                          cosPAK0S_bkg_np,
                          nSigmapr_bkg_np,
                          dcaV0_bkg_np)).T

# Computing correlation matrix
# rowvar=False means that the columns represent the variables and the rows the observations
signal_corr_matrix = np.corrcoef(X_signal,
                                 rowvar=False)
background_corr_matrix = np.corrcoef(X_background,
                                     rowvar=False)

# Definition of the variables name
variable_names = ['massK0S',
                  'tImpParBach',
                  'tImpParV0',
                  'CtK0S',
                  'cosPAK0S',
                  'nSigmapr',
                  'dcaV0']

# Plot of the correlation matrix
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(signal_corr_matrix, cmap='coolwarm', aspect='auto')
plt.title('Signal Correlation Matrix')
plt.xticks(ticks=np.arange(len(variable_names)), labels=variable_names, rotation=45)
plt.yticks(ticks=np.arange(len(variable_names)), labels=variable_names)
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(background_corr_matrix, cmap='coolwarm', aspect='auto')
plt.title('Background Correlation Matrix')
plt.xticks(ticks=np.arange(len(variable_names)), labels=variable_names, rotation=45)
plt.yticks(ticks=np.arange(len(variable_names)), labels=variable_names)
plt.colorbar()

# Saves figure con the file
plt.savefig(os.path.join(output_folder, "correlation_matrixes.svg"))
plt.tight_layout()
#plt.show()