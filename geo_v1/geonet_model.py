from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from geonet_nets import *
from utils import *

class GeoNetModel(object):

    def __init__(self, opt, tgt_image, src_image_stack, intrinsics):
        self.opt = opt
        self.tgt_image = self.preprocess_image(tgt_image)
        self.src_image_stack = self.preprocess_image(src_image_stack)
        self.intrinsics = intrinsics

        self.build_model()

        if not opt.mode in ['train_rigid', 'train_flow']:
            return

        self.build_losses()

    def build_model(self):
        opt = self.opt
        self.tgt_image_pyramid = self.scale_pyramid(self.tgt_image, opt.num_scales)
        
        self.tgt_image_tile_pyramid = [tf.tile(img, [opt.num_source, 1, 1, 1]) \
                                      for img in self.tgt_image_pyramid]
        # src images concated along batch dimension
        if self.src_image_stack != None:
            self.src_image_concat = tf.concat([self.src_image_stack[:,:,:,4*i:4*(i+1)] \
                                    for i in range(opt.num_source)], axis=0)
            self.src_image_concat_pyramid = self.scale_pyramid(self.src_image_concat, opt.num_scales)

        if opt.add_dispnet:
            self.build_dispnet()

        if opt.add_posenet:
            self.build_posenet()
        
        if opt.add_dispnet and opt.add_posenet:
            self.build_resflownet()
            self.build_rigid_flow_warping()
            self.build_flow_consistency()

    def build_dispnet(self):
        opt = self.opt

        # build dispnet_inputs
        if opt.mode == 'test_depth':
            # for test_depth mode we only predict the depth of the target image
            self.dispnet_inputs = self.tgt_image
        else:
            # multiple depth predictions; tgt: disp[:bs,:,:,:] src.i: disp[bs*(i+1):bs*(i+2),:,:,:]
            self.dispnet_inputs = self.tgt_image
            for i in range(opt.num_source):
                self.dispnet_inputs = tf.concat([self.dispnet_inputs, self.src_image_stack[:,:,:,4*i:4*(i+1)]], axis=0)
        
        self.dispnet_inputs_mask = tf.cast(self.dispnet_inputs[:,:,:,3:4]>0, 'float32')
        #self.dispnet_inputs_mask_swell = self.swell(self.dispnet_inputs_mask)
        
        self.pred_disp = disp_net(opt, self.dispnet_inputs)

        if opt.scale_normalize:
            # As proposed in https://arxiv.org/abs/1712.00175, this can 
            # bring improvement in depth estimation, but not included in our paper.
            self.pred_disp = [self.spatial_normalize(disp) for disp in self.pred_disp]

        self.pred_depth = [1./d for d in self.pred_disp]

        
    def build_posenet(self):
        opt = self.opt

        # build posenet_inputs
        self.posenet_inputs = tf.concat([self.tgt_image, self.src_image_stack], axis=3)
        
        # build posenet
        self.pred_poses = pose_net(opt, self.posenet_inputs)


    def build_resflownet(self):
        opt = self.opt

        # build flownet_inputs
        self.fwd_flownet_inputs = tf.concat([self.tgt_image_tile_pyramid[0], self.src_image_concat_pyramid[0]], axis=3)
        self.bwd_flownet_inputs = tf.concat([self.src_image_concat_pyramid[0], self.tgt_image_tile_pyramid[0]], axis=3)

        self.flownet_inputs = tf.concat([self.fwd_flownet_inputs, self.bwd_flownet_inputs], axis=0)
        
        # build flownet
        self.pred_flow = flow_net(opt, self.flownet_inputs)

        # unnormalize pyramid flow back into pixel metric
        for s in range(opt.num_scales):
            curr_bs, curr_h, curr_w, _ = self.pred_flow[s].get_shape().as_list()
            scale_factor = tf.cast(tf.constant([curr_w, curr_h], shape=[1,1,1,2]), 'float32')
            scale_factor = tf.tile(scale_factor, [curr_bs, curr_h, curr_w, 1])
            self.pred_flow[s] = self.pred_flow[s] * scale_factor
            
        # split forward/backward flows
        self.fwd_res_flow_pyramid = [self.pred_flow[s][:opt.batch_size*opt.num_source] for s in range(opt.num_scales)]
        self.bwd_res_flow_pyramid = [self.pred_flow[s][opt.batch_size*opt.num_source:] for s in range(opt.num_scales)]
        
        

    def build_rigid_flow_warping(self):
        opt = self.opt
        bs = opt.batch_size

        # build rigid flow (fwd: tgt->src, bwd: src->tgt)
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []
        self.fwd_rigid_flow_origin_pyramid = []
        self.bwd_rigid_flow_origin_pyramid = []
        for index, s in enumerate(range(opt.num_scales)):
            for i in range(opt.num_source):
                fwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][:bs], axis=3),
                                 self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], False)
                bwd_rigid_flow = compute_rigid_flow(tf.squeeze(self.pred_depth[s][bs*(i+1):bs*(i+2)], axis=3),
                                 self.pred_poses[:,i,:], self.intrinsics[:,s,:,:], True)
                if not i:
                    fwd_rigid_flow_concat = fwd_rigid_flow
                    bwd_rigid_flow_concat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_concat = tf.concat([fwd_rigid_flow_concat, fwd_rigid_flow], axis=0)
                    bwd_rigid_flow_concat = tf.concat([bwd_rigid_flow_concat, bwd_rigid_flow], axis=0)
            self.fwd_rigid_flow_origin_pyramid.append(fwd_rigid_flow_concat)
            self.bwd_rigid_flow_origin_pyramid.append(bwd_rigid_flow_concat)
            
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_concat+self.fwd_res_flow_pyramid[index]*tf.tile((self.tgt_image_tile_pyramid[index][:,:,:,3:4] + 1.0), [1, 1, 1, 2]))
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_concat+self.bwd_res_flow_pyramid[index]*tf.tile((self.tgt_image_tile_pyramid[index][:,:,:,3:4] + 1.0), [1, 1, 1, 2]))
            
        # warping by rigid flow
        self.fwd_rigid_warp_pyramid = [flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_origin_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_rigid_warp_pyramid = [flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_origin_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        
        self.fwd_rigid_warp_mask_pyramid = [flow_warp(tf.expand_dims(tf.ones(self.src_image_concat_pyramid[s].get_shape()[:3]),3), self.fwd_rigid_flow_origin_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_rigid_warp_mask_pyramid = [flow_warp(tf.expand_dims(tf.ones(self.tgt_image_tile_pyramid[s].get_shape()[:3]),3), self.bwd_rigid_flow_origin_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        
        self.tgt_image_tile_pyramid_pure = [self.tgt_image_tile_pyramid[s][:,:,:,:3] for s in range(opt.num_scales)]
        self.tgt_image_tile_pyramid_mask = [self.tgt_image_tile_pyramid[s][:,:,:,3:4] for s in range(opt.num_scales)]
        self.fwd_rigid_warp_pyramid_pure = [self.fwd_rigid_warp_pyramid[s][:,:,:,:3] for s in range(opt.num_scales)]
        self.fwd_rigid_warp_pyramid_mask = [self.fwd_rigid_warp_pyramid[s][:,:,:,3:4] for s in range(opt.num_scales)]
        
        self.fwd_tgt_ignore = [tf.cast((self.tgt_image_tile_pyramid_mask[s] + self.fwd_rigid_warp_pyramid_mask[s] - self.fwd_rigid_warp_mask_pyramid[s])> -2,'float32') for s in range(opt.num_scales)]

        
        
        self.src_image_concat_pyramid_pure = [self.src_image_concat_pyramid[s][:,:,:,:3] for s in range(opt.num_scales)]
        self.src_image_concat_pyramid_mask = [self.src_image_concat_pyramid[s][:,:,:,3:4] for s in range(opt.num_scales)]
        self.bwd_rigid_warp_pyramid_pure = [self.bwd_rigid_warp_pyramid[s][:,:,:,:3] for s in range(opt.num_scales)]
        self.bwd_rigid_warp_pyramid_mask = [self.bwd_rigid_warp_pyramid[s][:,:,:,3:4] for s in range(opt.num_scales)]

        self.bwd_src_ignore = [tf.cast((self.src_image_concat_pyramid_mask[s] + self.bwd_rigid_warp_pyramid_mask[s]- self.bwd_rigid_warp_mask_pyramid[s])> -2,'float32') for s in range(opt.num_scales)]

        
        
        # compute reconstruction error  
        self.fwd_rigid_error_pyramid = [self.image_similarity(self.fwd_rigid_warp_pyramid_pure[s], self.tgt_image_tile_pyramid_pure[s], self.fwd_tgt_ignore[s]) \
                                       for s in range(opt.num_scales)]      
        self.bwd_rigid_error_pyramid = [self.image_similarity(self.bwd_rigid_warp_pyramid_pure[s], self.src_image_concat_pyramid_pure[s], self.bwd_src_ignore[s]) \
                                       for s in range(opt.num_scales)]

        
        #full flow
        self.fwd_tgt_ignore_full = [tf.cast(((self.tgt_image_tile_pyramid_mask[s]<0) & (self.fwd_rigid_warp_pyramid_mask[s]>0)) | (self.fwd_rigid_warp_mask_pyramid[s]<0.5), 'float32') for s in range(opt.num_scales)]        
        self.bwd_src_ignore_full = [tf.cast(((self.src_image_concat_pyramid_mask[s]<0) & (self.bwd_rigid_warp_pyramid_mask[s]>0)) | (self.bwd_rigid_warp_mask_pyramid[s]<0.5), 'float32') for s in range(opt.num_scales)]        
        self.fwd_full_warp_pyramid = [flow_warp(self.src_image_concat_pyramid[s], self.fwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]
        self.bwd_full_warp_pyramid = [flow_warp(self.tgt_image_tile_pyramid[s], self.bwd_rigid_flow_pyramid[s]) \
                                      for s in range(opt.num_scales)]

        self.fwd_full_warp_pyramid_pure = [self.fwd_full_warp_pyramid[s][:,:,:,:3] for s in range(opt.num_scales)]
        self.bwd_full_warp_pyramid_pure = [self.bwd_full_warp_pyramid[s][:,:,:,:3] for s in range(opt.num_scales)]
        # compute reconstruction error  
        self.fwd_full_error_pyramid = [self.image_similarity(self.fwd_full_warp_pyramid_pure[s], self.tgt_image_tile_pyramid_pure[s], self.fwd_tgt_ignore_full[s]) \
                                       for s in range(opt.num_scales)]   
        self.bwd_full_error_pyramid = [self.image_similarity(self.bwd_full_warp_pyramid_pure[s], self.src_image_concat_pyramid_pure[s], self.bwd_src_ignore_full[s]) \
                                       for s in range(opt.num_scales)]
            

        
    def build_flow_consistency(self):
        opt = self.opt

        # warp pyramid full flow
        self.bwd2fwd_flow_origin_pyramid = [flow_warp(self.bwd_rigid_flow_origin_pyramid[s], self.fwd_rigid_flow_origin_pyramid[s]) \
                                    for s in range(opt.num_scales)]
        self.fwd2bwd_flow_origin_pyramid = [flow_warp(self.fwd_rigid_flow_origin_pyramid[s], self.bwd_rigid_flow_origin_pyramid[s]) \
                                    for s in range(opt.num_scales)]
        # calculate flow consistency
        self.fwd_flow_diff_origin_pyramid = [tf.abs(self.bwd2fwd_flow_origin_pyramid[s] + self.fwd_rigid_flow_origin_pyramid[s]) for s in range(opt.num_scales)]
        self.bwd_flow_diff_origin_pyramid = [tf.abs(self.fwd2bwd_flow_origin_pyramid[s] + self.bwd_rigid_flow_origin_pyramid[s]) for s in range(opt.num_scales)]

        # warp pyramid full flow
        self.bwd2fwd_flow_pyramid = [flow_warp(self.bwd_rigid_flow_pyramid[s], self.fwd_rigid_flow_pyramid[s]) \
                                    for s in range(opt.num_scales)]
        self.fwd2bwd_flow_pyramid = [flow_warp(self.fwd_rigid_flow_pyramid[s], self.bwd_rigid_flow_pyramid[s]) \
                                    for s in range(opt.num_scales)]
        # calculate flow consistency
        self.fwd_flow_diff_pyramid = [tf.abs(self.bwd2fwd_flow_pyramid[s] + self.fwd_rigid_flow_pyramid[s]) for s in range(opt.num_scales)]
        self.bwd_flow_diff_pyramid = [tf.abs(self.fwd2bwd_flow_pyramid[s] + self.bwd_rigid_flow_pyramid[s]) for s in range(opt.num_scales)]
        
    def build_losses(self):
        opt = self.opt
        bs = opt.batch_size
        self.rigid_warp_loss = 0
        self.disp_smooth_loss = 0
        self.flow_warp_loss = 0
        self.flow_smooth_loss = 0
        self.rigid_smooth_loss = 0
        self.flow_consistency_loss = 0
        self.rigid_consistency_loss = 0

        for s in range(opt.num_scales):

        # disp_smooth_loss
            self.disp_smooth_loss += opt.disp_smooth_weight/(2**s) * self.compute_smooth_loss(self.pred_disp[s],
                            tf.concat([self.tgt_image_pyramid[s], self.src_image_concat_pyramid[s]], axis=0))

        # rigid_warp_loss
            self.rigid_warp_loss += opt.rigid_warp_weight*opt.num_source/2 * \
                            (tf.reduce_mean(self.fwd_rigid_error_pyramid[s]) + \
                             tf.reduce_mean(self.bwd_rigid_error_pyramid[s]))
                
        # rigid_smooth_loss
            self.rigid_smooth_loss += opt.flow_smooth_weight/(2**(s+1)) * \
                            (self.compute_flow_smooth_loss(self.fwd_rigid_flow_origin_pyramid[s], tf.tile(self.pred_depth[s][:bs], [2,1,1,1])) +
                            self.compute_flow_smooth_loss(self.bwd_rigid_flow_origin_pyramid[s], self.pred_depth[s][bs:]))

        # flow_warp_loss
            self.flow_warp_loss += opt.flow_warp_weight*opt.num_source/2 * \
                            (tf.reduce_mean(self.fwd_full_error_pyramid[s]) + tf.reduce_mean(self.bwd_full_error_pyramid[s]))
                
        # flow_smooth_loss
            self.flow_smooth_loss += opt.flow_smooth_weight/(2**(s+1)) * \
                            (self.compute_flow_smooth_loss(self.fwd_rigid_flow_pyramid[s], tf.tile(self.pred_depth[s][:bs], [2,1,1,1])) +
                            self.compute_flow_smooth_loss(self.bwd_rigid_flow_pyramid[s], self.pred_depth[s][bs:]))

        # flow_consistency_loss
            self.flow_consistency_loss += opt.flow_consistency_weight/2 * \
                            (tf.reduce_sum(tf.reduce_mean(self.fwd_flow_diff_pyramid[s] , axis=3, keep_dims=True) * \
                             (1-self.fwd_tgt_ignore_full[s])) / tf.reduce_sum(1-self.fwd_tgt_ignore_full[s]) + \
                             tf.reduce_sum(tf.reduce_mean(self.bwd_flow_diff_pyramid[s] , axis=3, keep_dims=True) * \
                             (1-self.bwd_src_ignore_full[s])) / tf.reduce_sum(1-self.bwd_src_ignore_full[s]))

            self.rigid_consistency_loss += opt.flow_consistency_weight/2 * \
                            (tf.reduce_sum(tf.reduce_mean(self.fwd_flow_diff_origin_pyramid[s] , axis=3, keep_dims=True) * \
                             (1-self.fwd_tgt_ignore[s])) / tf.reduce_sum(1-self.fwd_tgt_ignore_full[s]) + \
                             tf.reduce_sum(tf.reduce_mean(self.bwd_flow_diff_origin_pyramid[s] , axis=3, keep_dims=True) * \
                             (1-self.bwd_src_ignore[s])) / tf.reduce_sum(1-self.bwd_src_ignore_full[s]))
        #regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        self.total_loss = 0  # regularization_loss
        #self.depth_constraint_loss = self.depth_constraint(self.dispnet_inputs_mask, self.dispnet_inputs_mask_swell, self.pred_depth[0])
        self.total_loss += self.rigid_warp_loss + self.disp_smooth_loss + self.flow_warp_loss + self.flow_smooth_loss +self.rigid_smooth_loss + self.flow_consistency_loss + self.rigid_consistency_loss#+ self.depth_constraint_loss


    def swell(self, x):
        #opt = self.opt
        #return_x = []
        #for s in range(opt.num_scales):
        B, H, W, C = x.get_shape().as_list()
        new_x = tf.zeros((B, H+6, W+6, C))
        for i in range(-3,3):
            for j in range(-3,3):
                pad_x = tf.pad(x,[[0,0],[3+i,3-i],[3+j,3-j],[0,0]],"CONSTANT")
                new_x +=pad_x
        new_x = new_x[:,3:-3,3:-3,:]>0.8
         #   return_x.append(tf.cast(new_x, 'float32'))
        return tf.cast(new_x, 'float32')

    def depth_constraint(self, mask, swell, depth):
        return tf.maximum(tf.reduce_mean(depth*mask) - tf.reduce_mean(depth*swell), 0)
    
        
    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = slim.avg_pool2d(x, 3, 1, 'SAME')
        mu_y = slim.avg_pool2d(y, 3, 1, 'SAME')

        sigma_x  = slim.avg_pool2d(x ** 2, 3, 1, 'SAME') - mu_x ** 2
        sigma_y  = slim.avg_pool2d(y ** 2, 3, 1, 'SAME') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y , 3, 1, 'SAME') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def image_similarity(self, x, y, mask):
        #x = tf.concat([x[:,:,:,i:i+1] for i in range(x.shape[3]) if i % 4 != 3], axis=3)
        #y = tf.concat([y[:,:,:,i:i+1] for i in range(y.shape[3]) if i % 4 != 3], axis=3)
        
        return (self.opt.alpha_recon_image * self.SSIM(x, y) + (1-self.opt.alpha_recon_image) * tf.abs(x-y)) * tf.tile(1-mask,[1,1,1,3])

    def L2_norm(self, x, axis=3, keep_dims=True):
        curr_offset = 1e-10
        l2_norm = tf.norm(tf.abs(x) + curr_offset, axis=axis, keep_dims=keep_dims)
        return l2_norm

    def spatial_normalize(self, disp):
        _, curr_h, curr_w, curr_c = disp.get_shape().as_list()
        disp_mean = tf.reduce_mean(disp, axis=[1,2,3], keep_dims=True)
        disp_mean = tf.tile(disp_mean, [1, curr_h, curr_w, curr_c])
        return disp/disp_mean

    def scale_pyramid(self, img, num_scales):
        if img == None:
            return None
        else:
            scaled_imgs = [img]
            _, h, w, _ = img.get_shape().as_list()
            for i in range(num_scales - 1):
                ratio = 2 ** (i + 1)
                nh = int(h / ratio)
                nw = int(w / ratio)
                scaled_imgs.append(tf.image.resize_area(img, [nh, nw]))
            return scaled_imgs

    def gradient_x(self, img):
        gx = img[:,:,:-1,:] - img[:,:,1:,:]
        return gx

    def gradient_y(self, img):
        gy = img[:,:-1,:,:] - img[:,1:,:,:]
        return gy

    def compute_smooth_loss(self, disp, img):
        #img = tf.concat([img[:,:,:,i:i+1] for i in range(img.shape[3]) if i % 4 != 0], axis=3)
        disp_gradients_x = self.gradient_x(disp)
        disp_gradients_y = self.gradient_y(disp)

        image_gradients_x = self.gradient_x(img)
        image_gradients_y = self.gradient_y(img)

        weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
        weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

        smoothness_x = disp_gradients_x * weights_x
        smoothness_y = disp_gradients_y * weights_y

        return tf.reduce_mean(tf.abs(smoothness_x)) + tf.reduce_mean(tf.abs(smoothness_y))

    def compute_flow_smooth_loss(self, flow, img):
        smoothness = 0
        for i in range(2):
            smoothness += self.compute_smooth_loss(tf.expand_dims(flow[:,:,:,i], -1), img)
        return smoothness/2

    def preprocess_image(self, image):
        # Assuming input image is uint8
        if image == None:
            return None
        else:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            return image * 2. -1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.)/2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)
