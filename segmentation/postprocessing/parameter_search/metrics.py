import numpy as np


def compute_tp_fp_fn_tn(mask_ref, mask_pred, ignore_mask=None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


def generate_dice_scores(groundtruth, prediction):

    prediction = np.expand_dims(prediction, axis=0)
    prediction = np.expand_dims(prediction, axis=0)
    groundtruth = np.expand_dims(groundtruth, axis=0)
    groundtruth = np.expand_dims(groundtruth, axis=0)

    # nnUNet
    tp, fp, fn, tn = compute_tp_fp_fn_tn(groundtruth, prediction)
    if tp + fp + fn == 0:
        dice_score = np.nan
    else:
        dominator = np.sum(groundtruth) + np.sum(prediction)
        dice_score = 2 * tp / dominator

    return dice_score

def instance_dice(groundtruth, filtered, gt_props_table):
    '''
    groundtruth: the groundtruth label, after label() function, 16bit
    prediction: the prediction label, after label() function, 16bit
    '''
    mean_instance_dice = 0
    dice_list = []
    for index, row in gt_props_table.iterrows():
        x_start = int(row['bbox_0'])
        y_start = int(row['bbox_1'])
        z_start = int(row['bbox_2'])
        x_range = int(row['bbox_3'])
        y_range = int(row['bbox_4'])
        z_range = int(row['bbox_5'])
        #print(x_start, y_start, z_start, x_range, y_range, z_range)
        gt_instance = groundtruth[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range]
        filtered_instance = filtered[z_start:z_start+z_range, y_start:y_start+y_range, x_start:x_start+x_range]
        gt_instance[gt_instance > 0] == 1
        filtered_instance[filtered_instance > 0] == 1
        gt_instance = gt_instance.astype(np.int8)
        filtered_instance = filtered_instance.astype(np.int8)

        dice = generate_dice_scores(gt_instance, filtered_instance)
        dice_list.append(dice)

        if dice > 0.8:
            instance_dice_score = 1
        elif dice >= 0.3 and dice <= 0.8:
            instance_dice_score = 0.5
        else:
            instance_dice_score = 0
        mean_instance_dice += instance_dice_score
    mean_instance_dice = mean_instance_dice/len(gt_props_table)
    return mean_instance_dice, dice_list