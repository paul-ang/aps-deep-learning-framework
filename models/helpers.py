import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import jaccard_score


def compute_mae(fake_ct, real_ct):
    mae = np.sum(np.abs(fake_ct - real_ct)) / real_ct.size

    return mae

def unscale_image(scaled_image: np.array, original_range: list, scaled_range:list =[0, 1]):
    minmax_form = (scaled_image - scaled_range[0]) / (scaled_range[1] - scaled_range[0])
    original_image = minmax_form * (original_range[1] - original_range[0]) + original_range[0]

    return original_image

def compute_metrics(fake_ct, ori_ct_min, ori_ct_max, real_ct, real_mri=None, create_figure=False, save_figurename:str =None, scale_data=[0, 1]):
    '''
    :param fake_ct: numpy array of fake ct with shape [batch, H, W]
    :param ori_ct_min: original max ct HU value with shape [batch, 1, 1]
    :param ori_ct_max: orignal min ct HU value with shape [batch, 1, 1]
    :param real_ct: numpy array of real ct with shape [batch, H, W]
    :param real_mri: numpy array of real input image with shape [batch, H, W] (for visualization).
    If there is a channel dim and more than one channel, then this func will split into t2_F and t2_W
    :param create_figure: create a figure that visualizes the data
    :param save_figurename: file path to save the figure in
    :param scale_data: the range where the data has been normalized to. Default minmax [0, 1]
    :return: ssim, mae, fig
    '''

    # Calculate SSIM
    # The data is normalized in [1, 0] range, so data_range = max - min = 1
    ssim_score = ssim(real_ct.transpose([1, 2, 0]),
                      fake_ct.transpose([1, 2, 0]), multichannel=True,
                      data_range=1)

    # Reverse minmax scaling to compute the evaluation metric
    # Unscale
    real_ct = unscale_image(real_ct, [ori_ct_min, ori_ct_max], scale_data)
    fake_ct = unscale_image(fake_ct, [ori_ct_min, ori_ct_max], scale_data)

    # Calculate MAE
    mae = compute_mae(fake_ct, real_ct)

    # Calculate PSNR
    mse = np.sum((fake_ct - real_ct) ** 2) / real_ct.size
    data_range = real_ct.max() - real_ct.min()
    psnr = 10 * np.log10((data_range ** 2) / mse)

    # Tissue specific metric
    # Air
    air_mask = (real_ct < -400)
    air_mae = compute_mae(fake_ct[air_mask], real_ct[air_mask])

    # Soft tissue
    tissue_mask = np.logical_and(real_ct >= -400, real_ct <= 160)
    tissue_mae = compute_mae(fake_ct[tissue_mask], real_ct[tissue_mask])

    # Bone
    bone_mask = (real_ct > 160)
    bone_mae = compute_mae(fake_ct[bone_mask], real_ct[bone_mask])

    # Calculate IOC
    pred_seg_map = get_seg_map(fake_ct)
    gt_seg_map = get_seg_map(real_ct)
    mean_iou = jaccard_score(gt_seg_map.flatten(), pred_seg_map.flatten(),
                             average="macro")
    air_iou = jaccard_score(gt_seg_map.flatten(), pred_seg_map.flatten(),
                            labels=[0], average="macro")
    tissue_iou = jaccard_score(gt_seg_map.flatten(), pred_seg_map.flatten(),
                               labels=[1], average="macro")
    bone_iou = jaccard_score(gt_seg_map.flatten(), pred_seg_map.flatten(),
                             labels=[2], average="macro")

    # Log figure
    fig = None
    if create_figure:  # only compatible when batch=1
        assert real_ct.shape[0] == 1, 'batch size is not 1 for visualizing figure.'
        assert real_mri is not None, "Input MRI is needed for visualization."
        fig = plt.figure(figsize=(10, 10))

        # MRI
        if len(real_mri.shape) == 4 and real_mri.shape[1] == 2:
            real_t2_F = real_mri[:, 0, :, :]
            real_t2_W = real_mri[:, 1, :, :]

            plt.subplot(3, 2, 1)
            plt.imshow(real_t2_F.squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Input - T2_F')

            plt.subplot(3, 2, 2)
            plt.imshow(real_t2_W.squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Input - T2_W')
        else:
            plt.subplot(3, 2, 1)
            plt.imshow(real_mri.squeeze(), cmap='gray')
            plt.axis('off')
            plt.title('Input - in-phase MRI')

        plt.subplot(3, 2, 3)
        plt.imshow(fake_ct.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'sCT (SSIM: {ssim_score:.4f}, MAE: {mae:.4f})')

        plt.subplot(3, 2, 4)
        plt.imshow(real_ct.squeeze(), cmap='gray')
        plt.axis('off')
        plt.title(f'Real CT')

        plt.subplot(3, 2, 5)
        im = plt.imshow((real_ct.squeeze() - fake_ct.squeeze()))
        plt.axis('off')
        plt.title('Difference map')
        fig.colorbar(im)

        if save_figurename is not None:
            plt.savefig(save_figurename, dpi=100)

        plt.close(fig)

    # Return metrics
    to_return = {
        'ssim': ssim_score,
        'mae': mae,
        'fig': fig,
        'psnr': psnr,
        'air_mae': air_mae,
        'tissue_mae': tissue_mae,
        'bone_mae': bone_mae,
        'iou': mean_iou,
        'bone_iou': bone_iou,
        'air_iou': air_iou,
        'tissue_iou': tissue_iou
    }
    return to_return


def get_seg_map(ct):
    seg_map = np.zeros_like(ct)
    seg_map[(ct < -400)] = 0  # air
    seg_map[(np.logical_and(ct >= -400, ct <= 160))] = 1  # soft tissue
    seg_map[(ct > 160)] = 2  # bone

    return seg_map
