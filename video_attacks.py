from base_attacks import Attack
import torch
import torch.nn as nn
import scipy.stats as st
import numpy as np
import torchvision
from PIL import Image
import random
import math
import time
import torch.nn.functional as F
from utils import norm_grads
from pick_restore import mask_video_frames, restore_masked_frames

class TemporalTranslation(Attack):
    '''
    paper: Boosting the transferability of video adversarial examples via temporal translation
    Replace conv with multiple queries.
    There are two ways: Cycle and Exchange. 
    Contain momentum or no momentum.
    params = {'kernlen':args.kernlen, # conv1 params
        'momentum':args.momentum
        'weight':args.augmentation_weight,
        'move_type': 'adj',
        'kernel_mode': 'gaussian'}
    '''
    def __init__(self, model, params, epsilon=16/255, steps=10, delay=1.0):
        super(TemporalTranslation, self).__init__("TemporalTranslation", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.delay = delay

        for name, value in params.items():
            setattr(self, name, value)
        
        self.frames = 32
        self.cycle_move_list = self._move_info_generation()
        if self.kernel_mode == 'gaussian':
            kernel = self._initial_kernel_gaussian(self.kernlen).astype(np.float32) # (self.kernlen)
        elif self.kernel_mode == 'linear':
            kernel = self._initial_kernel_linear(self.kernlen).astype(np.float32) # (self.kernlen)
        elif self.kernel_mode == 'random':
            kernel = self._initial_kernel_uniform(self.kernlen).astype(np.float32) # (self.kernlen)
        
        self.kernel = torch.from_numpy(np.expand_dims(kernel, 0)).to(self.device) # 1,self.kernlen

    def _move_info_generation(self):
        max_move = int((self.kernlen - 1) / 2) 
        lists = [i for i in range(-max_move, max_move+1)]
        return lists
    
    def _initial_kernel_linear(self, kernlen):
        k = int((kernlen - 1) / 2)
        kern1d = []
        for i in range(k+1):
            kern1d.append(1 - i / (k+1))
        kern1d = np.array(kern1d[::-1][:-1] + kern1d)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _initial_kernel_uniform(self, kernlen):
        kern1d = np.ones(kernlen)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _initial_kernel_gaussian(self, kernlen):
        assert kernlen%2 == 1
        k = (kernlen - 1) /2
        sigma = k/3
        k = int(k)
        def calculte_guassian(x, sigma):
            return (1/(sigma*np.sqrt(2*np.pi)) * np.math.exp(-(x**2)/(2* (sigma**2))))
        kern1d = []
        for i in range(-k, k+1):
            kern1d.append(calculte_guassian(i, sigma))
        assert len(kern1d) == kernlen
        kern1d = np.array(kern1d)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _conv1d_frame(self, grads):
        '''
        grads: D, N, C, T, H, W
        '''
        # cycle padding for grads
        D,N,C,T,H,W = grads.shape
        grads = grads.reshape(D, -1)
        
        grad = torch.matmul(self.kernel, grads)
        grad = grad.reshape(N,C,T,H,W)
        return grad

    def _cycle_move(self, adv_videos, cycle_move):
        if cycle_move < 0:
            direction = -1
        else:
            direction = 1
        cycle_move = abs(cycle_move)
        cycle_move = cycle_move % self.frames
        new_videos = torch.zeros_like(adv_videos)
        for i in range(self.frames):
            ori_ind = i
            new_ind = (ori_ind + direction * cycle_move) % self.frames 
            new_videos[:,:,new_ind] = adv_videos[:,:,ori_ind]
        return new_videos

    def _cycle_move_large(self, adv_videos, cycle_move):
        if cycle_move < 0:
            direction = -1
        else:
            direction = 1
        cycle_move = abs(cycle_move)
        if cycle_move == 0:
            cycle_move = cycle_move % self.frames
        else:
            cycle_move = (cycle_move + (int(self.frames/2)-1)) % self.frames
        new_videos = torch.zeros_like(adv_videos)
        for i in range(self.frames):
            ori_ind = i
            new_ind = (ori_ind + direction * cycle_move) % self.frames 
            new_videos[:,:,new_ind] = adv_videos[:,:,ori_ind]
        return new_videos

    def _cycle_move_random(self, adv_videos, cycle_move):
        if cycle_move < 0:
            direction = -1
        else:
            direction = 1
        # cycle_move = abs(cycle_move)
        if cycle_move == 0:
            cycle_move = cycle_move % self.frames
        else:
            cycle_move = random.randint(0, 100) % self.frames
        # cycle_move = (cycle_move + int(self.frames/2)) % self.frames
        new_videos = torch.zeros_like(adv_videos)
        for i in range(self.frames):
            ori_ind = i
            new_ind = (ori_ind + direction * cycle_move) % self.frames 
            new_videos[:,:,new_ind] = adv_videos[:,:,ori_ind]
        return new_videos

    def _exchange_move(self, adv_videos, exchange_lists):
        new_videos = adv_videos.clone()
        for exchange in exchange_lists:
            one_frame, ano_frame = exchange
            new_videos[:,:,one_frame] = adv_videos[:,:,ano_frame]
            new_videos[:,:,ano_frame] = adv_videos[:,:,one_frame]
        return new_videos

    def _get_grad(self, adv_videos, labels, loss):
        batch_size = adv_videos.shape[0]
        used_labels = torch.cat([labels]*batch_size, dim=0)
        adv_videos.requires_grad = True
        outputs = self.model(adv_videos)
        cost = self._targeted*loss(outputs, used_labels).to(self.device)
        grad = torch.autograd.grad(cost, adv_videos, 
                                    retain_graph=False, create_graph=False)[0]
        return grad

    def _grad_augmentation(self, grads):
        '''
        Input:
            grads: kernlen, grad.shape
        Return 
            grad
        '''
        same_position_diff_frame = grads.clone()
        diff_position_same_frame = torch.zeros_like(grads)
        for ind, cycle_move in enumerate(self.cycle_move_list):
            diff_position_same_frame[ind] = self._cycle_move(grads[ind], -cycle_move)
        s_conv_grad = self._conv1d_frame(same_position_diff_frame)
        d_conv_grad = self._conv1d_frame(diff_position_same_frame)
        grad = (1-self.weight)*s_conv_grad + self.weight*d_conv_grad
        return grad

    def forward(self, videos, labels):
        r"""
        Overridden.
        """
        videos = videos.to(self.device)
        momentum = torch.zeros_like(videos).to(self.device)
        labels = labels.to(self.device)
        loss = nn.CrossEntropyLoss()
        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back') # [0, 1]
        adv_videos = videos.clone().detach()
        del videos

        start_time = time.time()
        for i in range(self.steps):
            # obtain grads of these variants
            batch_new_videos = []
            for cycle_move in self.cycle_move_list:
                if self.move_type == 'adj':
                    new_videos = self._cycle_move(adv_videos, cycle_move)
                elif self.move_type == 'large':
                    new_videos = self._cycle_move_large(adv_videos, cycle_move)
                elif self.move_type == 'random':
                    new_videos = self._cycle_move_random(adv_videos, cycle_move)
                batch_new_videos.append(new_videos)
            batch_inps = torch.cat(batch_new_videos, dim=0)
            grads = []
            batch_times = 5
            length = len(self.cycle_move_list)
            if self.model_name == 'TPNet':
                batch_times = length
                print (self.model_name, batch_times)
            batch_size = math.ceil(length / batch_times)
            for i in range(batch_times):
                grad = self._get_grad(batch_inps[i*batch_size:min((i+1)*batch_size, length)], labels, loss)
                grads.append(grad)
            # grad augmentation
            grads = torch.cat(grads, dim=0)
            grads = torch.unsqueeze(grads, dim=1)
            grad = self._grad_augmentation(grads)

            # momentum 
            if self.momentum:
                grad = norm_grads(grad)
                grad += momentum * self.delay
                momentum = grad
            else:
                pass

            adv_videos = self._transform_video(adv_videos.detach(), mode='back') # [0, 1]
            adv_videos = adv_videos + self.step_size*grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward') # norm
            print ('now_time', time.time()-start_time)
        return adv_videos


class KLDivergenceLoss(nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, outputs, targets):
        log_softmax_outputs = F.log_softmax(outputs, dim=1)

        if targets.dtype == torch.long:
            targets = F.one_hot(targets, num_classes=outputs.size(1)).float()
        softmax_targets = F.softmax(targets, dim=1)

        loss = F.kl_div(log_softmax_outputs, softmax_targets, reduction='batchmean')
        return loss

class Temporal_Skip_Connection(Attack):

    def __init__(self, model, params, mask_num, top_k, epsilon=16 / 255, steps=1, delay=1.0):
        super(Temporal_Skip_Connection, self).__init__("Temporal_Skip_Connection", model)
        self.epsilon = epsilon
        self.steps = steps
        self.step_size = self.epsilon / self.steps
        self.delay = delay
        self.mask_num = mask_num
        self.top_k = top_k
        self.RKL_loss = KLDivergenceLoss()

        for name, value in params.items():
            setattr(self, name, value)

        self.frames = 32
        if self.kernel_mode == 'gaussian':
            kernel = self._initial_kernel_gaussian(self.kernlen).astype(np.float32)
        elif self.kernel_mode == 'linear':
            kernel = self._initial_kernel_linear(self.kernlen).astype(np.float32)
        elif self.kernel_mode == 'uniform':
            kernel = self._initial_kernel_uniform(self.kernlen).astype(np.float32)

        self.kernel = torch.from_numpy(np.expand_dims(kernel, 0)).to(self.device)

        ti_kernel = self._initial_kernel(15, 3).astype(np.float32)
        stack_kernel = np.stack([ti_kernel, ti_kernel, ti_kernel])
        self.stack_kernel = torch.from_numpy(np.expand_dims(stack_kernel, 1)).to(self.device)

    def _initial_kernel_linear(self, kernlen):
        k = int((kernlen - 1) / 2)
        kern1d = []
        for i in range(k + 1):
            kern1d.append(1 - i / (k + 1))
        kern1d = np.array(kern1d[::-1][:-1] + kern1d)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _initial_kernel_uniform(self, kernlen):
        kern1d = np.ones(kernlen)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _initial_kernel_gaussian(self, kernlen):
        assert kernlen % 2 == 1
        k = (kernlen - 1) / 2
        sigma = k / 3
        k = int(k)

        def calculte_guassian(x, sigma):
            return (1 / (sigma * np.sqrt(2 * np.pi)) * np.math.exp(-(x ** 2) / (2 * (sigma ** 2))))

        kern1d = []
        for i in range(-k, k + 1):
            kern1d.append(calculte_guassian(i, sigma))
        assert len(kern1d) == kernlen
        kern1d = np.array(kern1d)
        kernel = kern1d / kern1d.sum()
        return kernel

    def _conv1d_frame(self, grads):
        '''
        grads: D, N, C, T, H, W
        '''
        D, N, C, T, H, W = grads.shape
        grads = grads.reshape(D, -1)

        grad = torch.matmul(self.kernel, grads)
        grad = grad.reshape(N, C, T, H, W)
        return grad

    def _calculate_loss(self, outputs, loss1, loss2, true_labels, k):
        device = outputs.device
        cost1 = loss1(outputs, true_labels)
        _, topk_labels = self._get_topk_label(outputs, true_labels, k)
        cost2 = 0
        for i in range(k):
            target_distributions = F.one_hot(topk_labels[:, i], num_classes=outputs.size(1)).to(device).float()
            cost2 += loss2(outputs, target_distributions)
        cost2 /= k
        cost = cost1 - cost2
        return cost

    def _get_grad(self, adv_videos, labels, loss1, loss2, top_k):
        batch_size = adv_videos.shape[0]
        used_labels = torch.cat([labels] * batch_size, dim=0)
        adv_videos.requires_grad = True
        outputs = self.model(adv_videos)
        cost = self._targeted * self._calculate_loss(outputs, loss1, loss2, used_labels, top_k)
        grad = torch.autograd.grad(cost, adv_videos,
                                   retain_graph=False, create_graph=False)[0]
        return grad

    def _initial_kernel(self, kernlen, nsig):
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def _conv2d_frame(self, grads):
        '''
        grads: N, C, T, H, W
        '''
        frames = grads.shape[2]
        out_grads = torch.zeros_like(grads)
        for i in range(frames):
            this_grads = grads[:, :, i]
            out_grad = nn.functional.conv2d(this_grads, self.stack_kernel, groups=3, stride=1, padding=7)
            out_grads[:, :, i] = out_grad
        out_grads = out_grads / torch.mean(torch.abs(out_grads), [1, 2, 3], True)
        return out_grads

    def _get_topk_label(self, outputs, true_labels, k):
        probabilities = F.softmax(outputs, dim=1)

        for i, label in enumerate(true_labels):
            probabilities[i, label] = 0

        topk_probabilities, topk_indices = torch.topk(probabilities, k + 1, dim=1)

        topk_filtered_indices = []
        topk_filtered_probabilities = []
        for i in range(len(true_labels)):
            mask = topk_indices[i] != true_labels[i]
            filtered_indices = topk_indices[i][mask][:k]
            filtered_probabilities = topk_probabilities[i][mask][:k]
            topk_filtered_indices.append(filtered_indices)
            topk_filtered_probabilities.append(filtered_probabilities)

        return torch.stack(topk_filtered_probabilities), torch.stack(topk_filtered_indices)

    def forward(self, videos, labels):

        videos = videos.to(self.device)
        momentum = torch.zeros_like(videos).to(self.device)
        labels = labels.to(self.device)

        loss1 = nn.CrossEntropyLoss()
        loss2 = self.RKL_loss

        unnorm_videos = self._transform_video(videos.clone().detach(), mode='back')
        adv_videos = videos.clone().detach()
        del videos

        start_time = time.time()
        for q in range(self.steps):
            batch_new_videos = []
            batch_new_list = []

            for i in range((self.kernlen - 1) // 2):
                new_videos_A, list_A, new_videos_B, list_B = mask_video_frames(adv_videos, self.mask_num)
                batch_new_videos.append(new_videos_A)
                batch_new_videos.append(new_videos_B)
                batch_new_list.append(list_A)
                batch_new_list.append(list_B)

            batch_new_videos.insert(self.kernlen // 2, adv_videos)
            batch_inps = torch.cat(batch_new_videos, dim=0)

            grads = []
            batch_size = 1
            length = self.kernlen

            batch_times = self.kernlen
            for m in range(batch_times):
                grad = self._get_grad(batch_inps[m * batch_size:min((m + 1) * batch_size, length)], labels, loss1,
                                      loss2, self.top_k)
                grad = self._conv2d_frame(grad)
                grads.append(grad)

            grads = torch.cat(grads, dim=0)
            grads = restore_masked_frames(grads, batch_new_list)

            grads = torch.unsqueeze(grads, dim=1)
            grad = self._conv1d_frame(grads)

            if self.momentum:
                grad = norm_grads(grad)
                grad += momentum * self.delay
                momentum = grad

            adv_videos = self._transform_video(adv_videos.detach(), mode='back')
            adv_videos = adv_videos + self.step_size * grad.sign()
            delta = torch.clamp(adv_videos - unnorm_videos, min=-self.epsilon, max=self.epsilon)
            adv_videos = torch.clamp(unnorm_videos + delta, min=0, max=1).detach()
            adv_videos = self._transform_video(adv_videos, mode='forward')
            print('now_time', time.time() - start_time)

        return adv_videos
