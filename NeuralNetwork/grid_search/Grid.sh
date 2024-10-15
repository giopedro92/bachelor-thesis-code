#!/bin/bash

# Delete directory of previous attemps
rm -rf output_files
# Create a directory to save output files
mkdir -p output_files

# Array to track already selected scripts
selected_scripts=()

############################## Python code ##############################

# Function to execute the selected script and save the output
execute_script() {
    case "$1" in
        "1")
            echo "Output will be saved in a txt file..."
            python3 BDT_Grid.py > output_files/output_BDT.txt
            echo "########################## Done! ##########################"
            ;;
        "2")
            echo "Output will be saved in a txt file..."
            python3 KNN_Grid.py > output_files/output_KNN.txt
            echo "########################## Done! ##########################"
            ;;
        "3")
            echo "Output will be saved in a txt file..."
            python3 NN_Grid.py > output_files/output_NN.txt
            echo "########################## Done! ##########################"
            ;;
        "4")
            echo "Output will be saved in a txt file..."
            python3 Random_Forest_Grid.py > output_files/output_RandomForest.txt
            echo "########################## Done! ##########################"
            ;;
    esac
}

####################################################################################



############################## Setting User Interface ##############################

# Loop to prompt the user to choose which script to execute
while true; do
    echo "########################## Grid Search ##########################"
    echo "Select the script to analyze the dataset:"
    
    # Displaying remaining options
    for option in "1. BDT" "2. KNN" "3. NN" "4. RandomForest"; do
        if ! [[ " ${selected_scripts[@]} " =~ " ${option%%.*} " ]]; then
            echo " - $option"
        fi
    done
    
    # Option to quit
    echo " - 5. Thank you, I'm good"
    
    read -p "Your Choice: " choice
    
    # Check if the choice is valid
    case "$choice" in
        1|2|3|4)
            # Check if the script is already selected
            if [[ " ${selected_scripts[@]} " =~ " $choice " ]]; then
                echo "########################## Invalid choice ##########################"
                echo "Script already selected. Please try again."
            else
                echo "########################## Executing the code... ##########################"
                execute_script "$choice"
                selected_scripts+=("$choice")
                
            fi
            ;;
        5)
            echo "Exiting..."
            break
            ;;
        *)
            echo "########################## Invalid choice ##########################"
            echo "Invalid choice. Please try again."
            ;;
    esac
    
    # Check if all scripts have been selected
    if [[ ${#selected_scripts[@]} -eq 4 ]]; then
        echo "Exiting..."
        break
    fi
done
##########################################################################################