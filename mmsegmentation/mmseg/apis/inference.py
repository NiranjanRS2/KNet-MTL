# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint

from mmseg.datasets.pipelines import Compose
from mmseg.models import build_segmentor
import matplotlib.image as mpimg

# COLORS= [(0, 0, 0), (121, 38, 209), (192, 38, 209), (161, 93, 5), (217, 0, 0), (223, 242, 7), (79, 171, 111), 
#          (79, 139, 171), (2, 232, 6), (59, 14, 87), (245, 116, 37), (3, 252, 227), (34, 186, 156), (171, 196, 30), 
#          (219, 112, 255),  (0, 156, 160), (136, 216, 165), (130, 232, 255), (204, 88, 113), (161, 109, 88), 
#          (149, 37, 203), (0, 130, 249), (79, 95, 130), (74, 120, 185), (135, 134, 74), 
#           (21, 100, 77), (56, 114, 76), (127, 99, 21), 
#          (200, 255, 0), (143, 67, 80), (0, 111, 45), (36, 200, 99), 
#          (87, 186, 56), (36, 102, 55), (26, 155, 167)]
COLORS= [(0, 0, 0), (121, 38, 209)]


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cuda')
        model.CLASSES = checkpoint['meta']['CLASSES']
        # model.PALETTE = checkpoint['meta']['PALETTE']
        model.PALETTE = COLORS

    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""

    def __call__(self, results):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """

        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        # img = mpimg.imread(results['img'])

        img = results['img']# = img 
        
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


def inference_segmentor(model, img):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded
            images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    # prepare data
    data = dict(img=img)
    # print(data['img'].shape)
    # print("*****************************")
    data = test_pipeline(data)
    # print(data)
    # print(len(data['img']))
    # print(len(data['img'][0]))
    # print(len(data['img'][0][0]))
    # print(len(data['img'][0][0][0]))
    
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]
        #print(data.shape)
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]


    # forward the model
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    # print("result Shape: ", result[0].shape)
    return result


def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
