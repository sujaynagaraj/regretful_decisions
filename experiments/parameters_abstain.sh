#!/usr/bin/env bash

declare -a noise_types=("class_conditional" "class_independent" "group")
declare -a model_types=("LR" "NN" "SVM")
declare -a datasets=("cshock_eicu" "cshock_mimic" "support" "saps" "lungcancer" "lungcancer_imbalanced" "saps_imbalanced" "support_imbalanced" "cshock_eicu_imbalanced" "cshock_mimic_imbalanced")

# Define misspecification parameters for class_independent and class_conditional
declare -a misspecify_params_class_independent=("under" "over")
declare -a misspecify_params_class_conditional=("flipped")

for i in {0..0}
do
    for j in {1..1}
    do
        for k in {0..4}
        do
            noise_type=${noise_types[$i]}
            model_type=${model_types[$j]}
            dataset=${datasets[$k]}

            # # Select the appropriate misspecification parameters based on noise_type
            # if [ "$noise_type" == "class_independent" ]; then
            #     misspecify_params=("${misspecify_params_class_independent[@]}")
            # elif [ "$noise_type" == "class_conditional" ]; then
            #     misspecify_params=("${misspecify_params_class_conditional[@]}")
            # else
            #     misspecify_params=("")  # No misspecification parameters for other noise types
            # fi

            # for misspecify_param in "${misspecify_params[@]}"
            # do
                #sbatch launch_abstain.sh $noise_type $model_type $dataset $misspecify_param
            sbatch launch_abstain.sh $noise_type $model_type $dataset "correct"
            # done
        done
    done
done
