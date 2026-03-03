import os
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import SimpleITK as sitk
import torch.nn.functional as F
import imageio
from einops import repeat
from icecream import ic


class Focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        super(Focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            print(f'Focal loss alpha={alpha}, will assign alpha values for each class')
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            print(f'Focal loss alpha={alpha}, will shrink the impact in background')
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] = alpha
            self.alpha[1:] = 1 - alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, preds, labels):
        """
        Calc focal loss
        :param preds: size: [B, N, C] or [B, C], corresponds to detection and classification tasks  [B, C, H, W]: segmentation
        :param labels: size: [B, N] or [B]  [B, H, W]: segmentation
        :return:
        """
        self.alpha = self.alpha.to(preds.device)
        preds = preds.permute(0, 2, 3, 1).contiguous()
        preds = preds.view(-1, preds.size(-1))
        B, H, W = labels.shape
        assert B * H * W == preds.shape[0]
        assert preds.shape[-1] == self.num_classes
        preds_logsoft = F.log_softmax(preds, dim=1)  # log softmax
        preds_softmax = torch.exp(preds_logsoft)  # softmax

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.low(1 - preds_softmax) == (1 - pt) ** r

        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(),
                                                                                                  target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calculate_metric_percase(pred, gt, raw_spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt, raw_spacing)
        hd95 = metric.binary.hd95(pred, gt, raw_spacing)
    else:
        print('bad')
        asd = -1
        hd95 = -1
    jc = metric.binary.jc(pred, gt)
    return dice, hd95, asd, jc

def calculate_metric_percase_nospacing(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
    else:
        print('bad')
        asd = -1
        hd95 = -1
    jc = metric.binary.jc(pred, gt)
    return dice, hd95, asd, jc

def calculate_metric_percase_nan(pred, gt, raw_spacing):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    if pred.sum() != 0:
        asd = metric.binary.asd(pred, gt, raw_spacing)
        hd95 = metric.binary.hd95(pred, gt, raw_spacing)
    else:
        asd = np.nan
        hd95 = np.nan
    jc = metric.binary.jc(pred, gt)
    return dice, hd95, asd, jc


def test_single_volume(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0, patch_size[0], patch_size[1]
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            net.eval()
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks = outputs['masks']
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred

        # get resolution
        case_raw = 'C:/ai-agent/data/ACDC/testing/' + case+ '.nii.gz'
        case_raw = sitk.ReadImage(case_raw)
        raw_spacing = case_raw.GetSpacing()
        raw_spacing_new = []
        raw_spacing_new.append(raw_spacing[2])
        raw_spacing_new.append(raw_spacing[1])
        raw_spacing_new.append(raw_spacing[0])
        raw_spacing = raw_spacing_new

    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i,raw_spacing))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        #sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        #sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        print('saved successfully!')
    return metric_list

def test_single_volume_mean(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0, patch_size[0], patch_size[1]
            inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
            net.eval()
            with torch.no_grad():
                outputs = net(inputs, multimask_output, patch_size[0])
                output_masks1 = outputs['masks']
                output_masks2 = outputs['masks2']
                output_masks1 = torch.softmax(output_masks1, dim=1)
                output_masks2 = torch.softmax(output_masks2, dim=1)
                output_masks = (output_masks1 + output_masks2)/2.0
                out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                out_h, out_w = out.shape
                if x != out_h or y != out_w:
                    pred = zoom(out, (x / out_h, y / out_w), order=0)
                else:
                    pred = out
                prediction[ind] = pred
        # get resolution
        case_raw = 'C:/ai-agent/data/ACDC/testing/' + case+ '.nii.gz'
        case_raw = sitk.ReadImage(case_raw)
        raw_spacing = case_raw.GetSpacing()
        raw_spacing_new = []
        raw_spacing_new.append(raw_spacing[2])
        raw_spacing_new.append(raw_spacing[1])
        raw_spacing_new.append(raw_spacing[0])
        raw_spacing = raw_spacing_new

    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase_nan(prediction == i, label == i,raw_spacing))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        #sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        #sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        print('saved successfully!')
    return metric_list

def test_single_image(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        slice = image#[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
       
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0, patch_size[0], patch_size[1]
        inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction = pred
        
    else:
        x, y = image.shape[-2:]
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase_nospacing(prediction == i, label == i))

    return metric_list

def test_single_image_mean(image, label, net, classes, multimask_output, patch_size=[256, 256], input_size=[224, 224],
                       test_save_path=None, case=None, z_spacing=1):
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        slice = image#[ind, :, :]
        x, y = slice.shape[0], slice.shape[1]
        if x != patch_size[0] or y != patch_size[1]:
            slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0, patch_size[0], patch_size[1]
        inputs = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks = outputs['masks']
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            out = out.cpu().detach().numpy()
            out_h, out_w = out.shape
            if x != out_h or y != out_w:
                pred = zoom(out, (x / out_h, y / out_w), order=0)
            else:
                pred = out
            prediction = pred
    else:
        x, y = image.shape[-2:]
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        if x != patch_size[0] or y != patch_size[1]:
            image = zoom(image, (patch_size[0] / x, patch_size[1] / y), order=3)
        inputs = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        inputs = repeat(inputs, 'b c h w -> b (repeat c) h w', repeat=3)
        net.eval()
        with torch.no_grad():
            outputs = net(inputs, multimask_output, patch_size[0])
            output_masks1 = outputs['masks']
            output_masks2 = outputs['masks2']
            output_masks1 = torch.softmax(output_masks1, dim=1)
            output_masks2 = torch.softmax(output_masks2, dim=1)
            output_masks = (output_masks1 + output_masks2)/2.0
            out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase_nospacing(prediction == i, label == i))

    if test_save_path is not None:
        # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
        print('saved successfully!')
    return metric_list


def test_single_image_kvasir(image, label, net, classes, multimask_output, patch_size=[512, 512], input_size=[224, 224],
                             test_save_path=None, case=None):
    """
    Test single 2D image for Kvasir dataset (no 3D volume, no spacing)
    image: (C, H, W) RGB image
    label: (H, W) binary mask
    """
    # image is already (C, H, W) from dataloader
    c, h, w = image.shape
    
    # Resize if needed
    if h != patch_size[0] or w != patch_size[1]:
        image_resized = zoom(image, (1, patch_size[0] / h, patch_size[1] / w), order=3)
    else:
        image_resized = image
    
    # Prepare input (add batch dimension)
    inputs = torch.from_numpy(image_resized).unsqueeze(0).float().cuda()  # (1, C, H, W)
    
    net.eval()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks = outputs['masks']
        out = torch.argmax(torch.softmax(output_masks, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
        
        # Resize back to original size
        if h != patch_size[0] or w != patch_size[1]:
            prediction = zoom(prediction, (h / patch_size[0], w / patch_size[1]), order=0)
    
    # Calculate metrics (no spacing for 2D images)
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase_nospacing(prediction == i, label == i))
    
    # Save prediction if requested
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        # Save as image
        pred_img = (prediction * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(test_save_path, f'{case}_pred.png'), pred_img)
        print(f'Saved prediction for {case}')
    
    return metric_list


def test_single_image_kvasir_mean(image, label, net, classes, multimask_output, patch_size=[512, 512], input_size=[224, 224],
                                   test_save_path=None, case=None):
    """
    Test single 2D image for Kvasir with dual mask averaging
    """
    c, h, w = image.shape
    
    if h != patch_size[0] or w != patch_size[1]:
        image_resized = zoom(image, (1, patch_size[0] / h, patch_size[1] / w), order=3)
    else:
        image_resized = image
    
    inputs = torch.from_numpy(image_resized).unsqueeze(0).float().cuda()
    
    net.eval()
    with torch.no_grad():
        outputs = net(inputs, multimask_output, patch_size[0])
        output_masks1 = outputs['masks']
        output_masks2 = outputs['masks2']
        output_masks1 = torch.softmax(output_masks1, dim=1)
        output_masks2 = torch.softmax(output_masks2, dim=1)
        output_masks = (output_masks1 + output_masks2) / 2.0
        out = torch.argmax(output_masks, dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
        
        if h != patch_size[0] or w != patch_size[1]:
            prediction = zoom(prediction, (h / patch_size[0], w / patch_size[1]), order=0)
    
    metric_list = []
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase_nospacing(prediction == i, label == i))
    
    if test_save_path is not None:
        os.makedirs(test_save_path, exist_ok=True)
        pred_img = (prediction * 255).astype(np.uint8)
        imageio.imwrite(os.path.join(test_save_path, f'{case}_pred.png'), pred_img)
        print(f'Saved prediction for {case}')
    
    return metric_list


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

# utils.py 맨 아래에 추가

from medpy import metric
import torch.nn.functional as F
import torch
import numpy as np

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum() > 0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum() == 0:
        return 1, 0
    else:
        return 0, 0

def test_single_image_kvasir_mean(image, label, model, classes, patch_size=[512, 512], input_size=[512, 512]):
    import cv2  # 리사이즈를 위해 필요
    import torch.nn.functional as F
    
    image = image.cuda()
    # (1, 3, H, W) 형태
    
    # [수정 1] 모델 입력 전, 이미지를 patch_size로 리사이즈
    # (원본 이미지가 너무 크거나 작으면 모델이 처리를 못하거나 잘라버릴 수 있음)
    if image.shape[2] != patch_size[0] or image.shape[3] != patch_size[1]:
        image = F.interpolate(image, size=patch_size, mode='bilinear', align_corners=False)
    
    # 라벨은 원본 크기 유지 (평가는 원본 해상도에서 해야 정확함)
    # label is already numpy array from test_kvasir.py
    if isinstance(label, torch.Tensor):
        label = label.cpu().detach().numpy().squeeze()
    elif isinstance(label, np.ndarray):
        label = label.squeeze() if label.ndim > 2 else label
    # (Original_H, Original_W)
    
    model.eval()
    with torch.no_grad():
        # 모델 예측 (결과는 512x512)
        output = model(image, True, patch_size[0], prompt_idx=0, prompt_mode='point')
        
        if isinstance(output, dict):
            logits = output['masks']
        else:
            logits = output

        # 결과 추출 (0: 배경, 1: 용종)
        out = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy() # (512, 512)

    # [수정 2] 예측된 마스크(512x512)를 원본 라벨 크기(Original_H, Original_W)로 복원
    # 그래야 label과 겹쳐서 정확도를 계산할 수 있음
    if prediction.shape != label.shape:
        prediction = cv2.resize(prediction.astype(np.float32), 
                                (label.shape[1], label.shape[0]), 
                                interpolation=cv2.INTER_NEAREST)

    metric_list = []
    # 배경(0) 제외하고 용종(1) 클래스만 평가
    # classes는 foreground 클래스 수이므로 1부터 classes+1까지 평가
    for i in range(1, classes + 1):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
        
    return metric_list