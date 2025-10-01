import matplotlib.pyplot as plt
import numpy as np
import json

def display_average_scores_among_search(json_file, saving_title='average_scores_among_search.png'):
    search_idx = []
    after_mean_dice = []
    after_mean_instance_dice = []
    before_mean_dice = -1
    before_mean_instance_dice = -1
    with open(json_file, 'r') as f:
        search_results = json.load(f)
        for search in search_results.keys():
            search_idx.append(search.split('_')[-1])
            after_mean_dice.append(search_results[search]['mean_dice'])
            after_mean_instance_dice.append(search_results[search]['mean_instance_dice'])

    max_after_mean_dice = max(after_mean_dice)
    max_after_mean_instance_dice = max(after_mean_instance_dice)
    max_after_mean_dice_idx = [idx+1 for idx, val in enumerate(after_mean_dice) if val == max_after_mean_dice]
    max_after_mean_instance_dice_idx = [idx+1 for idx, val in enumerate(after_mean_instance_dice) if val == max_after_mean_instance_dice]
    print(f'Max after mean Dice: {max_after_mean_dice} at search index {max_after_mean_dice_idx}')
    print(f'Max after mean Instance Dice: {max_after_mean_instance_dice} at search index {max_after_mean_instance_dice_idx}')

    before_mean_dice = search_results[search]['mean_dice_before_processing']
    before_mean_instance_dice = search_results[search]['mean_instance_dice_before_processing']

    w = 0.35

    f, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [1, 4]})

    # --- Panel A ---
    # place the bars at the middle of the total span of panel B
    center_pos = (len(search_idx)/4 - 1) / 2.0
    ax0.bar(center_pos - w/2, before_mean_instance_dice, width=w, label='Mean Instance Dice')
    ax0.bar(center_pos + w/2, before_mean_dice, width=w, label='Mean Dice')

    ax0.legend()
    ax0.set_xlabel('(A) Before Post-processing', fontsize=14)
    ax0.set_ylabel('Scores', fontsize=14)
    ax0.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    # 👇 match panel B’s x-span, but center the two bars
    ax0.set_xlim(-0.1, len(search_idx)/4 - 0.1)
    ax0.set_xticks([center_pos])
    ax0.set_xticklabels([''])

    ax0.set_yticks(np.arange(0, 1.1, 0.2))

    # --- Panel B ---
    indB = np.arange(len(search_idx))
    ax1.bar(indB - w/2, after_mean_instance_dice, width=w, label='Mean Instance Dice')
    ax1.bar(indB + w/2, after_mean_dice, width=w, label='Mean Dice')
    ax1.legend()
    ax1.set_xlabel('(B) Search Index', fontsize=14)
    ax1.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    ax1.set_xticks(indB)
    ax1.set_xticklabels(search_idx)
    ax1.set_yticks(np.arange(0, 1.1, 0.2))

    plt.suptitle(
        'Average mean Dice and mean Instance Dice scores on all training cubes (A) before post-processing \n'
        'and (B) after post-processing by searched parameters',
        fontsize=16
    )

    plt.tight_layout()
    plt.savefig(saving_title, dpi=300)

    return set(max_after_mean_instance_dice_idx) & set(max_after_mean_dice_idx)

def display_variance_range_vs_roundness(json_file, select, saving_title='variance_range_vs_roundness.png'):
    search_selected = ['search_'+str(i) for i in select]
    low_variance = []
    high_variance = []
    roundness = []
    for search in search_selected:
        with open(json_file, 'r') as f:
            search_results = json.load(f)
            low_variance.append(search_results[search]['var_low'])
            high_variance.append(search_results[search]['var_high'])
            roundness.append(search_results[search]['roundness'])
        variance_range = np.asarray(high_variance) - np.asarray(low_variance)
        print(f'Search Index: {search} Variance Range: {variance_range} Roundness: {roundness}')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.scatter(variance_range, roundness)
    for i in range(len(search_selected)):
        ax.annotate(search_selected[i], (variance_range[i], roundness[i]))
    ax.set_xlabel('Variance Range', fontsize=14)
    ax.set_ylabel('Roundness', fontsize=14)
    ax.set_title('Variance Range vs Roundness', fontsize=16)
    x_range = max(variance_range) - min(variance_range)
    y_range = max(roundness) - min(roundness)
    ax.set_xlim(min(variance_range) - 0.1*x_range, max(variance_range) + 0.1*x_range)
    ax.set_ylim(min(roundness) - 0.1*y_range, max(roundness) + 0.1*y_range)
    plt.tight_layout()
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.savefig(saving_title, dpi=300)
    plt.show()

def display_variance_low_vs_roundness(json_file, select=[1, 5, 7, 11, 12, 13, 14], saving_title='variance_low_vs_roundness.png'):
    search_selected = ['search_'+str(i) for i in select]
    low_variance = []
    roundness = []
    for search in search_selected:
        with open(json_file, 'r') as f:
            search_results = json.load(f)
            low_variance.append(search_results[search]['var_low'])
            roundness.append(search_results[search]['roundness'])
        print(f'Search Index: {search} Variance Low: {low_variance} Roundness: {roundness}')
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.scatter(low_variance, roundness)
    for i in range(len(search_selected)):
        ax.annotate(search_selected[i], (low_variance[i], roundness[i]))
    ax.set_xlabel('Variance Low', fontsize=14)
    ax.set_ylabel('Roundness', fontsize=14)
    ax.set_title('Variance Low vs Roundness', fontsize=16)
    x_range = max(low_variance) - min(low_variance)
    y_range = max(roundness) - min(roundness)
    ax.set_xlim(min(low_variance) - 0.1*x_range, max(low_variance) + 0.1*x_range)
    ax.set_ylim(min(roundness) - 0.1*y_range, max(roundness) + 0.1*y_range)
    plt.tight_layout()
    plt.grid(True, linestyle='--', which='major', color='grey', alpha=.25)
    plt.savefig(saving_title, dpi=300)
    plt.show()

if __name__ == '__main__':
    json_file = '/hdd2/yang/projects/glomeruli_segmentation/2025-zhou-hipct-hierarchical-segmentation/data/parameter_search/LADAF-2020-27_Left_cube_12-1/seed_1_padding_32/search_result.json'
    saving_title = 'LADAF-2020-27_Left_cube_12-1_average_scores_among_search.png'
    search_list = display_average_scores_among_search(json_file, saving_title=saving_title)
    search_list = list(search_list)
    print(f'Best search indices: {search_list}')
    #saving_title = 'LADAF-2020-27_Left_cube_12-1_variance_range_vs_roundness.png'
    #display_variance_range_vs_roundness(json_file, search_list, saving_title=saving_title)
    saving_title = 'LADAF-2020-27_Left_cube_12-1_variance_low_vs_roundness.png'
    display_variance_low_vs_roundness(json_file, search_list, saving_title=saving_title)


    # search_list = [3,5,7,11,14,15]
    #display_range_roundness(json_file, search_list)

    # im_dir = 'D:\\Yang\\data\\kidney_seg\\2.58um_data\\extraction\\param_search_images'
    # pred_dir = 'D:\\Yang\\data\\kidney_seg\\2.58um_data\\extraction\\param_search_preds'
    # save_dir = 'D:\\Yang\\results\\glomeruli_segmentation\\param_search\\2.58um_search_5_seed_0'
    # var_low = 98597.96758243884
    # var_high = 426400.55614048045
    # roundness = 0.7608542063635457
    # generate_results(im_dir, pred_dir, save_dir, var_low, var_high, roundness)