#!/usr/bin/env python
# -*- coding:utf-8 -*-
# 
#-----------------------------------------------------------------------------------

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import yolo.config as cfg
from keras.layers import Activation
from keras import layers

class Resnet18(object):
    def __init__(self, isTraining = True):
        self.classes = cfg.CLASSES
        self.num_class = len(self.classes)

        self.box_per_cell = cfg.BOX_PRE_CELL
        self.cell_size = cfg.CELL_SIZE
        self.batch_size = cfg.BATCH_SIZE
        self.image_size = cfg.IMAGE_SIZE
        self.anchor = cfg.ANCHOR
        self.alpha = cfg.ALPHA

        self.class_scale = 1.0
        self.object_scale = 5.0
        self.noobject_scale = 1.0
        self.coordinate_scale = 1.0
        
        self.offset = np.transpose(np.reshape(np.array([np.arange(self.cell_size)] * self.cell_size * self.box_per_cell),
                                         [self.box_per_cell, self.cell_size, self.cell_size]), (1, 2, 0))
        self.offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32), [1, self.cell_size, self.cell_size, self.box_per_cell])
        self.offset = tf.tile(self.offset, (self.batch_size, 1, 1, 1))

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.logits = self.build_networks(self.images)

        if isTraining:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5], name = 'labels')
            self.total_loss = self.loss_layer(self.logits, self.labels)
            tf.summary.scalar('total_loss', self.total_loss)
    

    def build_networks(self, inputs):

        net = self.conv_layer(inputs, [3, 3, 3, 32],name = '0_conv')
        net = self.pooling_layer(net, name = '1_pool')

        net = self.conv_layer(net, [3, 3, 32, 64], name = '2_conv')
        net = self.identity_block(net, [3, 3, 32, 64], name = 'identity_1')
        net = self.conv_layer(net, [3, 3, 32, 64], name = '3_conv')
        #net = self.identity_block(net, [3, 3, 64, 128],name = 'identity_2')


        net = self.conv_layer(net, [3, 3, 64, 128], name = '4_conv')
        #net = self.conv_layer(net, [1, 1, 128, 64], name = '5_conv')
        net = self.identity_block(net,[3, 3, 64, 128],name= 'identty_2')
        net = self.conv_layer(net, [3, 3, 64, 128], name = '5_conv')
        net = self.conv_layer(net, [3, 3, 64, 256], name = '6_conv')
        net = self.identity_block(net,[3, 3, 128, 256],name= 'identty_3')

        net = self.conv_layer(net, [3, 3, 128, 256], name = '7_conv')
        net = self.conv_layer(net, [3, 3, 256, 512], name = '8_conv')       
        net = self.identity_block(net,[3, 3, 256, 512],name= 'identty_4')

        net = self.conv_layer(net, [3, 3, 256, 512], name = '9_conv')
        net = self.conv_layer(net, [3, 3, 256, 1024], name = '10_conv')
        #net = self.conv_layer(net, [1, 1, 256, 128], name = '9_conv')
        net = self.identity_block(net,[3, 3, 256, 1024],name= 'identty_5')
        
       
        #net = self.identity_block(net,[3, 3, 256, 512],name= 'identty_8')

        
        net = self.pooling_layer(net, name = '2_pool')

        net = self.conv_layer(net, [3, 3, 512, 1024], name = '11_conv')
        

        net = self.conv_layer(net, [1, 1, 1024, self.box_per_cell * (self.num_class + 5)],name = '11_conv')

        return net


        '''def build_networks(self, inputs):
        net = self.conv_layer(inputs, [3, 3, 3, 32], name = '0_conv')
        net = self.pooling_layer(net, name = '1_pool')

        net = self.conv_layer(net, [3, 3, 32, 64], name = '2_conv')
        net = self.conv_layer(net, [3, 3, 32, 64], name = '3_conv')
        net = self.identity_block(net, [3, 3, 32, 64],name= 'identity_1')
        #net = self.pooling_layer(net, name = '3_pool')
        net = self.conv_layer(net, [3, 3, 32, 64], name = '4_conv')
        net = self.conv_layer(net, [3, 3, 32, 64], name = '5_conv')
        net = self.identity_block(net, [3, 3, 64, 128],name= 'identity_2')


        net = self.conv_layer(net, [3, 3, 64, 128], name = '6_conv')
        #net = self.conv_layer(net, [1, 1, 128, 64], name = '5_conv')
        net = self.conv_layer(net, [3, 3, 64, 128], name = '7_conv')
        net = self.identity_block(net,[3, 3, 64, 128],name= 'identty_3')

        net = self.conv_layer(net, [3, 3, 64, 128], name = '8_conv')
        net = self.conv_layer(net, [3, 3, 64, 128], name = '9_conv')
        net = self.identity_block(net, [3, 3, 64, 256],name= 'identity_4')
        #net = self.pooling_layer(net, name = '7_pool')

        net = self.conv_layer(net, [3, 3, 128, 256], name = '10_conv')
        #net = self.conv_layer(net, [1, 1, 256, 128], name = '9_conv')
        net = self.conv_layer(net, [3, 3, 128, 256], name = '11_conv')
        net = self.identity_block(net,[3, 3, 128, 256],name= 'identty_5')

        net = self.conv_layer(net, [3, 3, 128, 256], name = '12_conv')
        #net = self.conv_layer(net, [1, 1, 256, 128], name = '9_conv')
        net = self.conv_layer(net, [3, 3, 128, 256], name = '13_conv')
        net = self.identity_block(net,[3, 3, 256, 512],name= 'identty_6')
        #net = self.pooling_layer(net, name = '11_pool')

        net = self.conv_layer(net, [3, 3, 256, 512], name = '14_conv')
        #net = self.conv_layer(net, [1, 1, 512, 256], name = '13_conv')
        net = self.conv_layer(net, [3, 3, 256, 512], name = '15_conv')
        net = self.identity_block(net,[3, 3, 256, 512],name= 'identty_7')

        net = self.conv_layer(net, [3, 3, 256, 512], name = '16_conv')
        #net = self.conv_layer(net, [1, 1, 512, 256], name = '13_conv')
        net = self.conv_layer(net, [3, 3, 256, 512], name = '17_conv')
        net = self.identity_block(net,[3, 3, 256, 512],name= 'identty_8')
        #net = self.conv_layer(net, [1, 1, 512, 256], name = '15_conv')
        #net16 = self.conv_layer(net, [3, 3, 256, 512], name = '16_conv')
        net = self.pooling_layer(net, name = '2_pool')

        net = self.conv_layer(net, [3, 3, 512, 1024], name = '18_conv')
        

        net = self.conv_layer(net, [1, 1, 1024, self.box_per_cell * (self.num_class + 5)], batch_norm=False, name = '19_conv')

        return net'''
    

    def conv_layer(self, inputs, shape, name = '0_conv'):
        weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        #filters = 32
        #weight = tf.Variable(tf.he_normal(shape, stddev=0.1), name='weight')
        #biases = tf.Variable(tf.constant(0.1, shape=[shape[3]]), name='biases')

        conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)
        #conv = tf.nn.conv2d(filters, (1,1), strides=[1, 1, 1, 1],padding='SAME',name=name)(inputs)
        conv = Activation('relu')(conv)
        #conv = tf.nn.conv2d(inputs, (1,1), strides=[1, 1, 1, 1], padding='SAME', name=name)
        conv = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)
        shortcut = tf.nn.conv2d(inputs, weight, strides=[1, 1, 1, 1], padding='SAME',name=name)
        #shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

        conv = layers.add([conv, shortcut])
        conv = Activation('relu')(conv)
        ''' if batch_norm:
            depth = shape[3]
            scale = tf.Variable(tf.ones([depth, ], dtype='float32'), name='scale')
            shift = tf.Variable(tf.zeros([depth, ], dtype='float32'), name='shift')
            mean = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_mean')
            variance = tf.Variable(tf.ones([depth, ], dtype='float32'), name='rolling_variance')

            conv_bn = tf.nn.batch_normalization(conv, mean, variance, shift, scale, 1e-05)
            conv = tf.add(conv_bn, biases)
            conv = tf.maximum(self.alpha * conv, conv)
        else:
            conv = tf.add(conv, biases)'''

        return conv

    def identity_block(self,inputs,shape,name):
        
        #x = self.conv_layer(inputs,(1,1),filters)(inputs)
        #weight = tf.Variable(tf.truncated_normal(shape, stddev=0.1), name='weight')
        x = self.conv_layer(inputs,shape,name = name)
        x = Activation('relu')(x)

        x = layers.add([x,inputs])
        x = Activation('relu')(x)

        return x

    def pooling_layer(self, inputs, name = '1_pool'):
        pool = tf.nn.max_pool(inputs, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME', name = name)
        return pool


    def loss_layer(self, predict, label):
        predict = tf.reshape(predict, [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class + 5])
        box_coordinate = tf.reshape(predict[:, :, :, :, :4], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        box_confidence = tf.reshape(predict[:, :, :, :, 4], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 1])
        box_classes = tf.reshape(predict[:, :, :, :, 5:], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])

        boxes1 = tf.stack([(1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 0])) + self.offset) / self.cell_size,
                           (1.0 / (1.0 + tf.exp(-1.0 * box_coordinate[:, :, :, :, 1])) + tf.transpose(self.offset, (0, 2, 1, 3))) / self.cell_size,
                           tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 2]) * np.reshape(self.anchor[:5], [1, 1, 1, 5]) / self.cell_size),
                           tf.sqrt(tf.exp(box_coordinate[:, :, :, :, 3]) * np.reshape(self.anchor[5:], [1, 1, 1, 5]) / self.cell_size)])
        box_coor_trans = tf.transpose(boxes1, (1, 2, 3, 4, 0))
        box_confidence = 1.0 / (1.0 + tf.exp(-1.0 * box_confidence))
        box_classes = tf.nn.softmax(box_classes)

        response = tf.reshape(label[:, :, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell])
        boxes = tf.reshape(label[:, :, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, 4])
        classes = tf.reshape(label[:, :, :, :, 5:], [self.batch_size, self.cell_size, self.cell_size, self.box_per_cell, self.num_class])

        iou = self.calc_iou(box_coor_trans, boxes)
        best_box = tf.to_float(tf.equal(iou, tf.reduce_max(iou, axis=-1, keep_dims=True)))
        confs = tf.expand_dims(best_box * response, axis = 4)

        conid = self.noobject_scale * (1.0 - confs) + self.object_scale * confs
        cooid = self.coordinate_scale * confs
        proid = self.class_scale * confs

        coo_loss = cooid * tf.square(box_coor_trans - boxes)
        con_loss = conid * tf.square(box_confidence - confs)
        pro_loss = proid * tf.square(box_classes - classes)

        loss = tf.concat([coo_loss, con_loss, pro_loss], axis = 4)
        loss = tf.reduce_mean(tf.reduce_sum(loss, axis = [1, 2, 3, 4]), name = 'loss')

        return loss


    def calc_iou(self, boxes1, boxes2):
        boxx = tf.square(boxes1[:, :, :, :, 2:4])
        boxes1_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes1[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes1[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes1[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes1 = tf.transpose(box, (1, 2, 3, 4, 0))

        boxx = tf.square(boxes2[:, :, :, :, 2:4])
        boxes2_square = boxx[:, :, :, :, 0] * boxx[:, :, :, :, 1]
        box = tf.stack([boxes2[:, :, :, :, 0] - boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] - boxx[:, :, :, :, 1] * 0.5,
                        boxes2[:, :, :, :, 0] + boxx[:, :, :, :, 0] * 0.5,
                        boxes2[:, :, :, :, 1] + boxx[:, :, :, :, 1] * 0.5])
        boxes2 = tf.transpose(box, (1, 2, 3, 4, 0))

        left_up = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
        right_down = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

        intersection = tf.maximum(right_down - left_up, 0.0)
        inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]
        union_square = boxes1_square + boxes2_square - inter_square

        return tf.clip_by_value(1.0 * inter_square / union_square, 0.0, 1.0)
