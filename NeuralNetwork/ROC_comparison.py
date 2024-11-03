############################### Start import ###############################
print("--------------------START IMPORT--------------------")

import sys
import os
import matplotlib.pyplot as plt

############################### End import ###############################

############################### Start of code ###############################
print("--------------------START OF THE CODE--------------------")

# Get model names from the command line arguments
model_names = sys.argv[1:]

# List to store ROC curves of all models
all_fpr = []
all_tpr = []

# Loop through all models
for model_name in model_names:
    file_name = model_name + ".txt"

    file_path = os.path.join("evaluation_results/", file_name)

    with open(file_path, 'r') as f:
        for _ in range(5):
            next(f)
        fpr = []
        tpr = []
        for line in f:
            values = line.strip().split("\t")
            fpr.append(float(values[0]))
            tpr.append(float(values[1]))
        all_fpr.append(fpr)
        all_tpr.append(tpr)

# Plot all overlaid ROC curves
for fpr, tpr, model_name in zip(all_fpr, all_tpr, model_names):
    plt.plot(tpr, [1 - x for x in fpr], label=model_name)

plt.xlabel("Signal Efficiency")
plt.ylabel("Background Rejection")
plt.title("ROC Curve")
plt.legend()

# Save figure
save_path = "evaluation_results/ROC_comparison/ROC_comparison.svg"
plt.savefig(save_path)

#plt.show()

############################### End ###############################
print("--------------------END--------------------")