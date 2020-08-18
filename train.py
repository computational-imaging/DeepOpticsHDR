"""
 Modified by Chris Metzler 2020.

 " License:
 " -----------------------------------------------------------------------------
 " Copyright (c) 2017, Gabriel Eilertsen.
 " All rights reserved.
 " 
 " Redistribution and use in source and binary forms, with or without 
 " modification, are permitted provided that the following conditions are met:
 " 
 " 1. Redistributions of source code must retain the above copyright notice, 
 "    this list of conditions and the following disclaimer.
 " 
 " 2. Redistributions in binary form must reproduce the above copyright notice,
 "    this list of conditions and the following disclaimer in the documentation
 "    and/or other materials provided with the distribution.
 " 
 " 3. Neither the name of the copyright holder nor the names of its contributors
 "    may be used to endorse or promote products derived from this software 
 "    without specific prior written permission.
 " 
 " THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 " AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 " IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 " ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 " LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 " CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 " SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 " INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 " CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 " ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE 
 " POSSIBILITY OF SUCH DAMAGE.
 " -----------------------------------------------------------------------------
 "
 " Description: Training script for the HDR-CNN
 " Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
 " Date: February 2018
"""
import matplotlib.pyplot as plt
import time, math, os, sys, random
import tensorflow as tf
import tensorlayer as tl
import threading
import numpy.matlib as matlib

import numpy as np
import scipy.stats as st
import utils.SimLDR_camera as camera

sys.path.insert(0, "../")
from src import network, optics, optics_numpy
from utils import img_io

eps = 1.0/255.0


#=== Settings =================================================================

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("sx",               "320",   "Image width")
tf.flags.DEFINE_integer("sy",               "320",   "Image height")
tf.flags.DEFINE_integer("num_threads",      "4",     "Number of threads for multi-threaded loading of data")
tf.flags.DEFINE_integer("print_batch_freq", "5000",  "Frequency for printing stats and saving images/parameters")
tf.flags.DEFINE_integer("print_batches",    "5",     "Number of batches to output images for at each [print_batch_freq] step")
tf.flags.DEFINE_bool("print_im",            "true",  "If LDR sample images should be printed at each [print_batch_freq] step")
tf.flags.DEFINE_bool("print_hdr",           "false", "If HDR reconstructions should be printed at each [print_batch_freq] step")

tf.flags.DEFINE_string("raw_dir",           "Train_Dataset/AllSources/",                                          "Path to unprocessed dataset")
tf.flags.DEFINE_string("data_dir",          "new_training_data_fewSat",                                         "Path to processed dataset. This data will be created if the flag [preprocess] is set")
tf.flags.DEFINE_string("output_dir",        "TrainedNetworks/MedNetnoLog_OpticalPSF_100xLR_L2Gamma_HDR64/",       "Path to output directory, for weights and intermediate results")#my_Unet
tf.flags.DEFINE_string("parameters",        "PretrainedNetworks/SimulatedPSF_Network/logs/model_step_740000.npz", "Path to trained params for complete network")

tf.flags.DEFINE_float("sub_im_clip1",       "0.98",  "Min saturation limit, i.e. min fraction of non-saturated pixels")
tf.flags.DEFINE_float("sub_im_clip2",       "0.99",  "Max saturation limit, i.e. max fraction of non-saturated pixels")
tf.flags.DEFINE_bool("load_params",         "false", "Load the parameters from the [parameters] path, otherwise the parameters from [vgg_path] or random parameters will be used")

# Data augmentation parameters
tf.flags.DEFINE_bool("preprocess",          "false", "Pre-process HDR input data, to create augmented dataset for training")
tf.flags.DEFINE_integer("sub_im",           "10",    "Number of subimages to pick in a 1 MP pixel image")
tf.flags.DEFINE_integer("sub_im_linearize", "0",     "Linearize input images")
tf.flags.DEFINE_float("sub_im_sc1",         "0.2",   "Min size of crop, in fraction of input image")
tf.flags.DEFINE_float("sub_im_sc2",         "0.6",   "Max size of crop, in fraction of input image")
tf.flags.DEFINE_float("sub_im_noise1",      "0.0",   "Min noise std")
tf.flags.DEFINE_float("sub_im_noise2",      "0.01",  "Max noise std")
tf.flags.DEFINE_float("sub_im_hue_mean",    "0.0",   "Mean hue")
tf.flags.DEFINE_float("sub_im_hue_std",     "7.0",   "Std of hue")
tf.flags.DEFINE_float("sub_im_sat_mean",    "0.0",   "Mean saturation")
tf.flags.DEFINE_float("sub_im_sat_std",     "0.1",   "Std of saturation")
tf.flags.DEFINE_float("sub_im_sigmn_mean",  "0.9",   "Mean sigmoid exponent")
tf.flags.DEFINE_float("sub_im_sigmn_std",   "0.1",   "Std of sigmoid exponent")
tf.flags.DEFINE_float("sub_im_sigma_mean",  "0.6",   "Mean sigmoid offset")
tf.flags.DEFINE_float("sub_im_sigma_std",   "0.1",   "Std of sigmoid offset")
tf.flags.DEFINE_integer("sub_im_min_jpg",   "30",    "Minimum quality level of generated LDR images")

# Learning parameters
tf.flags.DEFINE_float("num_epochs",         "100.0",   "Number of training epochs")
tf.flags.DEFINE_float("start_step",         "0.0",     "Step to start from")
tf.flags.DEFINE_float("learning_rate",      "0.0001",  "Starting learning rate for Adam optimizer")
tf.flags.DEFINE_integer("batch_size",       "4",       "Batch size for training")
tf.flags.DEFINE_bool("sep_loss",            "true",    "Use illumination + reflectance loss")
tf.flags.DEFINE_float("lambda_ir",          "0.5",     "Reflectance weight for the ill+refl loss")
tf.flags.DEFINE_bool("rand_data",           "true",    "Random shuffling of training data")
tf.flags.DEFINE_float("train_size",         "0.995",   "Fraction of data to use for training, the rest is validation data")
tf.flags.DEFINE_integer("buffer_size",      "256",     "Size of load queue when reading training data")

# Additional parameters
tf.flags.DEFINE_bool("Train_PSF",             "false",          "Train the PSF")
tf.flags.DEFINE_bool("Display_PSF",           "false",          "Print the PSF at every step as it trains")
tf.flags.DEFINE_string("PSF_init",            "random",         "How to initialize the PSF, options are GreenStar, star, perfect, offset, noisy_perfect, and random")
tf.flags.DEFINE_bool("Train_network",         "false",          "Train the networks")
tf.flags.DEFINE_bool("Tie_PSF",               "false",          "Use the same PSF for each of the color channels")
tf.flags.DEFINE_string("Loss",                "L2-Gamma-Batch", "Which loss to use")
tf.flags.DEFINE_integer("psf_size",           "91",             "PSF size")
tf.flags.DEFINE_bool("Use_VGG_init",          "false",          "Initialize the encoder with VGG weights or random weights. My encoder cannot use VGG")
tf.flags.DEFINE_string("psf_outdir",          "TrainedNetworks/MedNetnoLog_OpticalPSF_100xLR_L2Gamma_HDR64/", "Path to output directory, for PSF values")
tf.flags.DEFINE_string("psf_infilename",      "MeasuredPSF.npz", "Path to trained params for complete network")
tf.flags.DEFINE_bool("load_psf",              "false", "Load the PSF from [psf_indir], otherwise the PSF_init will be used")
tf.flags.DEFINE_string("Network",             "Medium_net_noLog",    "Which network architecture to use for reconstruction")
tf.flags.DEFINE_float("PSF_LR_Multiplier",    "100.",    "How many times higher to set the PSF learning rate compared to the rest of the network")
tf.flags.DEFINE_bool("DirectPSF", "false",    "Directly learn the PSF, or optimize a phase mask height map")
tf.flags.DEFINE_float("HDR_max_val",          "64",           "Largest value taken by HDR image (Very bright values will smear with PSF). At 64, you overexposed by 6 stops")
tf.flags.DEFINE_bool("ApplyCameraCurve",      "false", "Apply a camera curve while synthesis LDR images")
tf.flags.DEFINE_bool("NoDeconv", "false",     "Apply the network directly to the filtered image, without a deconvolution operation")
tf.flags.DEFINE_bool("Zernike",  "false",     "Parameterize the mask's height map using Zernike basis functions. Only used with an optical parameterization of the PSF, i.e., FLAGS.DirectPSF=False")
tf.flags.DEFINE_bool("Loss_wout_Crop",        "false", "Compute loss over the entire image, or just that of the PSF without boundary conditions.")
tf.flags.DEFINE_bool("ThinLens", "false",     "Model a thin lens placed between teh phase mask and the sensor-plane.")
tf.flags.DEFINE_bool("chromatic_aberrations", "false",  "Does the thin lens have chromatic aberrations?")
tf.flags.DEFINE_float("Laplace_l2_reg", "0.", "Regularize the laplacian of the height map. If 0, l2 reg is applied instead")
tf.flags.DEFINE_float("D_mask_to_aperture",   "0.", "Distance between the phase mask and the aperture ")
tf.flags.DEFINE_bool("load_captured_PSF",     "false", "Load a real-world PSF")
tf.flags.DEFINE_float("noise_std", "0.0",     "Std of noise to add, wrt images in range 0, 1")
tf.flags.DEFINE_bool("pad_symmetric",         "false", "")
#==============================================================================

sx = FLAGS.sx
sy = FLAGS.sy
data_dir_bin = os.path.join(FLAGS.data_dir, "bin")
data_dir_jpg = os.path.join(FLAGS.data_dir, "jpg")
log_dir = os.path.join(FLAGS.output_dir, "logs")
im_dir = os.path.join(FLAGS.output_dir, "im")

print(tf.app.flags.FLAGS.flag_values_dict())

#=== Pre-processing/data augmentation =========================================

# Process training data
if (FLAGS.preprocess):
    cmd = "./virtualcamera/virtualcamera -linearize %d -imsize %d %d 3 -input_path %s -output_path %s \
                                   -subimages %d -cropscale %f %f -clip %f %f -noise %f %f \
                                   -hue %f %f -sat %f %f -sigmoid_n %f %f -sigmoid_a %f %f \
                                   -jpeg_quality %d" % \
                                   (FLAGS.sub_im_linearize, sy, sx, FLAGS.raw_dir, FLAGS.data_dir, FLAGS.sub_im,
                                    FLAGS.sub_im_sc1, FLAGS.sub_im_sc2,
                                    FLAGS.sub_im_clip1, FLAGS.sub_im_clip2,
                                    FLAGS.sub_im_noise1, FLAGS.sub_im_noise2,
                                    FLAGS.sub_im_hue_mean, FLAGS.sub_im_hue_std,
                                    FLAGS.sub_im_sat_mean, FLAGS.sub_im_sat_std,
                                    FLAGS.sub_im_sigmn_mean, FLAGS.sub_im_sigmn_std,
                                    FLAGS.sub_im_sigma_mean, FLAGS.sub_im_sigma_std,
                                    FLAGS.sub_im_min_jpg);
    print("\nRunning processing of training data")
    print("cmd = '%s'\n\n"%cmd)

    # Remove old data, and run new data generation
    os.system("rm -rf %s"%FLAGS.data_dir)
    os.makedirs(data_dir_bin)
    os.makedirs(data_dir_jpg)
    os.system(cmd)
    print("\n")

# Create output directories
tl.files.exists_or_mkdir(log_dir)
tl.files.exists_or_mkdir(im_dir)


#=== Localize training data ===================================================

# Get names of all images in the training path
frames = [name for name in sorted(os.listdir(data_dir_bin)) if os.path.isfile(os.path.join(data_dir_bin, name))]


random.seed(0)
# Randomize the images
if FLAGS.rand_data:
    random.shuffle(frames)

# Split data into training/validation sets
splitPos = len(frames) - math.floor(max(FLAGS.batch_size, min((1-FLAGS.train_size)*len(frames), 1000)))
frames_train, frames_valid = np.split(frames, [int(splitPos)])

# Number of steps per epoch depends on the number of training images
training_samples = len(frames_train)
validation_samples = len(frames_valid)
steps_per_epoch = training_samples/FLAGS.batch_size

print("\n\nData to be used:")
print("\t%d training images" % training_samples)
print("\t%d validation images\n" % validation_samples)


#=== Setup data queues ========================================================

# For single-threaded queueing of frame names
input_frame = tf.placeholder(tf.string)
q_frames = tf.FIFOQueue(FLAGS.buffer_size, [tf.string])
enqueue_op_frames = q_frames.enqueue([input_frame])
dequeue_op_frames = q_frames.dequeue()

# For multi-threaded queueing of training images
input_target = tf.placeholder(tf.float32, shape=[sy, sx, 3])
q_train = tf.FIFOQueue(FLAGS.buffer_size, tf.float32, shapes=[sy,sx,3])
enqueue_op_train = q_train.enqueue(input_target)
y_= q_train.dequeue_many(FLAGS.batch_size)

Deconv_gamma_init = .1#2#Smaller will deconvolve more, larger will keep image unchanged
Deconv_gamma_sqrt=tf.Variable(Deconv_gamma_init,name='Deconv_gamma',dtype=tf.float32)
psf_size=FLAGS.psf_size#Must be odd
mid_pt=int((psf_size-1)/2)

if FLAGS.load_captured_PSF:
    psf_param = tf.constant(0.,dtype=tf.float32)
    effective_PSF_loaded=tf.Variable(np.ones((FLAGS.psf_size, FLAGS.psf_size, 1, 3)), dtype=tf.float32)
    [x, x_noPSF, y_PSF, effective_PSF, y_clipped] = camera.SimLDR_camera(y_, psf_param=None,
                                                                         effective_PSF=effective_PSF_loaded,
                                                                         HDR_max_val=FLAGS.HDR_max_val,
                                                                         apply_camera_curve=FLAGS.ApplyCameraCurve,noise_std=FLAGS.noise_std,pad_symmetric=FLAGS.pad_symmetric)
    if not FLAGS.NoDeconv:
        x_deconvolved = camera.inverse_filter(x, x, effective_PSF, gamma=tf.square(Deconv_gamma_sqrt))
        y_deconvolved = camera.inverse_filter(y_PSF, y_PSF, effective_PSF, gamma=tf.square(Deconv_gamma_sqrt))
    else:
        x_deconvolved = x
        y_deconvolved = y_PSF
else:
    if FLAGS.DirectPSF:
        psf_init = np.zeros((psf_size,psf_size,3,1),np.float32)
        if FLAGS.PSF_init=='random':
            psf_init = 1/psf_size**2 * np.random.rand(psf_size, psf_size, 3, 1)
        elif FLAGS.PSF_init == 'testing':
            psf_init[mid_pt, mid_pt, 0, 0] = .1
            psf_init[mid_pt, mid_pt, 1, 0] = .1
            psf_init[mid_pt, mid_pt, 2, 0] = .1
        elif FLAGS.PSF_init=='perfect' or FLAGS.PSF_init=='noisy_perfect':
            psf_init[mid_pt, mid_pt, 0, 0] = 1.
            psf_init[mid_pt, mid_pt, 1, 0] = 1.
            psf_init[mid_pt, mid_pt, 2, 0] = 1.
        elif FLAGS.PSF_init=='GreenStar':
            for i in range(psf_size):
                mag = 1e-3 * (1 - np.abs(mid_pt - i) / mid_pt)
                psf_init[i, i, 1, 0] = mag
                psf_init[i, psf_size - i - 1, 1, 0] = mag
                psf_init[mid_pt, i, 1, 0] = mag
                psf_init[i, mid_pt, 1, 0] = mag
            psf_init[mid_pt, mid_pt, 0, 0] = 1.
            psf_init[mid_pt, mid_pt, 1, 0] = 1.
            psf_init[mid_pt, mid_pt, 2, 0] = 1.
        elif FLAGS.PSF_init=='offset':
            psf_init[mid_pt-2, mid_pt-2, 0, 0] = 1.
            psf_init[mid_pt-2, mid_pt+2, 1, 0] = 1.
            psf_init[mid_pt+2, mid_pt+2, 2, 0] = 1.
        elif FLAGS.PSF_init=='star':
            for i in range(psf_size):
                # mag = 1e-4 * (1 - np.abs(mid_pt - i) / mid_pt)
                mag = 1e-4 * np.exp( - np.abs(mid_pt - i) / mid_pt*np.log(100))
                psf_init[i, i, 0, 0] = mag
                psf_init[i, psf_size - i - 1, 0, 0] = mag
                psf_init[i, i, 1, 0] = mag
                psf_init[i, psf_size - i - 1, 1, 0] = mag
                psf_init[i, i, 2, 0] = mag
                psf_init[i, psf_size - i - 1, 2, 0] = mag
                psf_init[mid_pt, i, 0, 0] = mag
                psf_init[mid_pt, i, 1, 0] = mag
                psf_init[mid_pt, i, 2, 0] = mag
                psf_init[i, mid_pt, 0, 0] = mag
                psf_init[i, mid_pt, 1, 0] = mag
                psf_init[i, mid_pt, 2, 0] = mag
            psf_init[mid_pt, mid_pt, 0, 0] = 1.-psf_init[:, :, 0, 0].sum()
            psf_init[mid_pt, mid_pt, 1, 0] = 1.-psf_init[:, :, 1, 0].sum()
            psf_init[mid_pt, mid_pt, 2, 0] = 1.-psf_init[:, :, 2, 0].sum()
        def logit(x):
            """ Computes the logit function, i.e. the logistic sigmoid inverse. """
            return - np.log(1. / np.maximum(np.minimum(1-1e-6,x),1e-6) - 1.)#Ensure the incoming value is between 1e-6 and 1-1e-6. Very close to 1 won't budge
        if FLAGS.Train_PSF==False:
            def logit(x):
                return - np.log(1. / np.maximum(np.minimum(1-1e-9,x),1e-9) - 1.)
        else:
            psf_init=psf_init+1e-7*np.random.rand(psf_size, psf_size, 3, 1)
        if FLAGS.PSF_init=='noisy_perfect':
            psf_init = psf_init + 1e-3 * np.random.rand(psf_size, psf_size, 3, 1)
        psf_init=np.abs(psf_init)
        psf_param_init=logit(psf_init)#SimLDR_camera now applies a sigmoid to compute the effective PSF

        if FLAGS.Train_PSF:
            if FLAGS.Tie_PSF:
                psf_param_tied=tf.Variable(psf_param_init[:,:,0,:],name='PSF',dtype=tf.float32)
                psf_tied_reshape=tf.reshape(psf_param_tied,(psf_size,psf_size,1,1))
                psf_param=tf.concat([psf_tied_reshape,psf_tied_reshape,psf_tied_reshape],axis=2)
            else:
                psf_param = tf.Variable(psf_param_init,name='PSF',dtype=tf.float32)
        else:
            psf_param = tf.constant(psf_param_init, name='PSF', dtype=tf.float32)
        [x, x_noPSF, y_PSF, effective_PSF, y_clipped] = camera.SimLDR_camera(y_, psf_param, HDR_max_val=FLAGS.HDR_max_val, apply_camera_curve=FLAGS.ApplyCameraCurve,noise_std=FLAGS.noise_std,pad_symmetric=FLAGS.pad_symmetric)
        if not FLAGS.Train_PSF and FLAGS.PSF_init=='perfect':
            x_deconvolved = x
            y_deconvolved = y_PSF
        else:
            if not FLAGS.NoDeconv:
                x_deconvolved = camera.inverse_filter(x,  x, effective_PSF, gamma=tf.square(Deconv_gamma_sqrt))
                y_deconvolved = camera.inverse_filter(y_PSF, y_PSF, effective_PSF, gamma=tf.square(Deconv_gamma_sqrt))
            else:
                x_deconvolved = x
                y_deconvolved = y_PSF
    else:#Parameterize PSF with a height map
        FLAGS.psf_size = np.maximum(FLAGS.sx, FLAGS.sy)
        psf_size = FLAGS.psf_size
        FLAGS.Loss_wout_Crop = True#With optical PSF, we can't crop the image before computing the loss

        # User-controlled Parameters
        focal_length = 35e-3  # 35 mm lens
        aperture_diameter = 5e-3
        f_number = focal_length / aperture_diameter
        # f_number = 8
        # aperture_diameter = focal_length/f_number
        z_focus = np.inf
        sensor_resolution = np.array([psf_size, psf_size])
        pixel_size = [i * 1 for i in [4.29e-6, 4.29e-6]]  # Allows one to treat multiple camera pixels as one super-pixel
        wave_lengths = np.array([635, 530, 450]) * 1e-9  # Wave lengths to be modeled and optimized for
        refractive_idcs = np.array([1.4295, 1.4349, 1.4421])  # From https://refractiveindex.info/?shelf=organic&book=polydimethylsiloxane&page=Schneider-Sylgard184
        height_map_noise = 20e-9

        # Auto-generated Parameters
        sensor_distance = 1 / (1 / focal_length - 1 / z_focus)
        patch_size = psf_size  # Size of patches to be extracted from images, and resolution of simulated sensor
        sampling_factor = 1
        optical_feature_size = pixel_size[0] / sampling_factor
        phase_mask_size = 5.6e-3
        phase_mask_feature_size = 1e-6
        phase_mask_quantization_res = 21.16e-9
        wave_resolution = int(np.ceil(phase_mask_size / optical_feature_size))
        wave_resolution = [wave_resolution, wave_resolution]
        r_cutoff = .5 * aperture_diameter / optical_feature_size

        #regularizer is a tensor to scalar function
        if FLAGS.Laplace_l2_reg==0.:
            regularizer = tf.contrib.layers.l2_regularizer(.5)
        else:
            regularizer= optics.laplace_l2_regularizer(FLAGS.Laplace_l2_reg)

        # Input field is a planar wave. I think this is where the focus at infinity assumption comes in.
        input_field = tf.cast(tf.ones((1, wave_resolution[0], wave_resolution[1], len(wave_lengths))),dtype=tf.complex64)

        x_inds = matlib.repmat(np.reshape(range(wave_resolution[0]),(wave_resolution[0],1)),1,wave_resolution[1])
        y_inds = matlib.repmat(np.transpose(np.array(range(wave_resolution[1]))),wave_resolution[0],1)
        if FLAGS.Zernike:
            #Parameterize the height_map using Zernike basis functions. Not used in this work.
            num_zernike_coeffs = 21
            if FLAGS.PSF_init=='random':
                zernike_inits = 20.*np.random.randn(num_zernike_coeffs, 1, 1)
            else:
                zernike_inits = np.zeros((num_zernike_coeffs, 1, 1))
            if not FLAGS.ThinLens and not FLAGS.PSF_init=='random':
                zernike_inits[3] = -51.  # This sets the defocus value to approximately focus the image for a distance of 1m.
            zernike_initializer = tf.constant_initializer(zernike_inits)
            zernike_coeffs = tf.get_variable('zernike_coeffs',
                                                  shape=[num_zernike_coeffs, 1, 1],
                                                  dtype=tf.float32,
                                                  trainable=True,
                                                  initializer=zernike_initializer)
            zernike_volume = optics.get_zernike_volume(resolution=wave_resolution[0], n_terms=num_zernike_coeffs).astype(np.float32)#Zernike basis functions. Tensor of shape (num_basis_functions, wave_resolution[0], wave_resolution[1]).
            height_map = tf.reduce_sum(zernike_coeffs * zernike_volume, axis=0)
            init_height_map_value = np.sum(zernike_inits * zernike_volume,axis=0)#For debugging
            height_map = tf.expand_dims(tf.expand_dims(height_map, 0), -1, name='height_map')
            Zern_element = optics.PhasePlate(wave_lengths=wave_lengths,
                                             height_map=height_map,
                                             refractive_idcs=refractive_idcs,
                                             height_tolerance=height_map_noise)
            field = Zern_element(input_field)
        else:
            if FLAGS.PSF_init == 'random':
                init_height_map_sqrt_value = 1e-3 * np.random.rand(wave_resolution[0], wave_resolution[ 1])
            else:
                init_height_map_sqrt_value = np.zeros((wave_resolution[0], wave_resolution[1]))
            init_height_map_value = init_height_map_sqrt_value**2
            height_map_sqrt_initializer = tf.constant_initializer(init_height_map_sqrt_value)
            [field, height_map] = optics.my_height_map_element(input_field,
                                                               wave_lengths=wave_lengths,
                                                               height_map_regularizer = regularizer,
                                                               height_map_sqrt_initializer=height_map_sqrt_initializer,
                                                               height_tolerance=height_map_noise,
                                                               refractive_idcs=refractive_idcs, name='height_map_optics',
                                                               height_max=1.55e-6)
        #Model an air-gap between the phase mask and the lens
        if FLAGS.D_mask_to_aperture>0:
            with tf.variable_scope('BeforeLens', reuse=False):
                field = optics.propagate_fresnel(field, distance=FLAGS.D_mask_to_aperture, sampling_interval=optical_feature_size, wave_lengths=wave_lengths)
                field = optics.circular_aperture(field, r_cutoff=r_cutoff)

        if FLAGS.ThinLens:
            # Model a thin lens between the phase mask and the sensor
            if FLAGS.chromatic_aberrations:
                c=1
                height_map_thinlens = optics_numpy.plano_convex_initializer(focal_length,
                                                                            wave_lengths[c],
                                                                            wave_resolution,
                                                                            optical_feature_size,
                                                                            refractive_idcs[c])
                # init_height_map_value = init_height_map_value + height_map_thinlens#This exact height map is only experience by the green channel.
                height_map_thinlens = np.expand_dims(np.expand_dims(height_map_thinlens, 0), -1)
                thin_lens = optics.PhasePlate(wave_lengths=wave_lengths,
                                              height_map=height_map_thinlens,
                                              refractive_idcs=np.array(3*[refractive_idcs[c]]))#Using the same refractive index for all three channels models a thin lens with no chromatic aberrations
                field=thin_lens(field)
            else:
                #Modify the phase of the field such that a field of ones would converge to a point at the center
                PhaseThinLens = optics_numpy.PhaseShiftThinLens_rgb(focal_length, wave_lengths, wave_resolution,
                                                                    optical_feature_size, refractive_idcs)
                field = tf.multiply(field, PhaseThinLens)
        field = optics.circular_aperture(field, r_cutoff=r_cutoff)
        # Propagate field from aperture to sensor
        field = optics.propagate_fresnel(field, distance=sensor_distance, sampling_interval=optical_feature_size, wave_lengths=wave_lengths)
        # The psf is the intensities of the propagated field.
        psfs = optics.get_intensities(field)

        #Resample field at the pixel resolution
        psfs = optics.area_downsampling_tf(psfs, wave_resolution[0] / sampling_factor)

        #Crop PSF to a subset of the total sensor area
        offset = int(np.ceil(wave_resolution[0]/sampling_factor/2)-np.ceil(patch_size/2))
        psfs = tf.image.crop_to_bounding_box(psfs, offset_height=offset, offset_width=offset, target_height=patch_size, target_width=patch_size)

        #normalize to sum to 1.
        psfs = tf.div(psfs, tf.reduce_sum(psfs, axis=[1, 2], keep_dims=True))

        train_PSF_params = tf.trainable_variables()

        if FLAGS.Zernike:
            psf_param = train_PSF_params[1]
        else:
            psf_param = tf.transpose(train_PSF_params[1], [1, 2, 0, 3])
        # Image formation: PSF is convolved with input HDR image
        effective_PSF = tf.transpose(psfs, [1, 2, 0, 3])
        [x, x_noPSF, y_PSF, effective_PSF, y_clipped] = camera.SimLDR_camera(y_, psf_param=None, effective_PSF=effective_PSF,HDR_max_val=FLAGS.HDR_max_val,apply_camera_curve=FLAGS.ApplyCameraCurve,noise_std=FLAGS.noise_std,pad_symmetric=FLAGS.pad_symmetric)
        if not FLAGS.NoDeconv:
            x_deconvolved = camera.inverse_filter(x, x, effective_PSF, gamma=tf.square(Deconv_gamma_sqrt))
            y_deconvolved = camera.inverse_filter(y_PSF, y_PSF, effective_PSF, gamma=tf.square(Deconv_gamma_sqrt))
        else:
            x_deconvolved = x
            y_deconvolved = y_PSF

#To verify that correct % of pixels are saturated
x_noPSF_sat=tf.reduce_sum(tf.cast(tf.equal(x_noPSF,1),dtype=tf.float32),[1,2,3])/tf.cast(tf.size(x_noPSF[0,:,:,:]),dtype=tf.float32)
print_Sat_op=tf.print("Fraction of Saturated Pixels:", x_noPSF_sat)
y_max=tf.reduce_max(y_,axis=[1,2,3])
print_Max_op=tf.print("Largest pixel value in each non-clipped image:", y_max)
y_max2=tf.reduce_max(y_clipped,axis=[1,2,3])
print_Max_op2=tf.print("Largest pixel value in each clipped image:", y_max2)

#=== Network ==================================================================

# Setup the network
print("Network setup:\n")
if FLAGS.Network=='Small_net':
    net, vgg16_conv_layers = network.my_small_model(x_deconvolved, FLAGS.batch_size, is_training=True)
elif FLAGS.Network=='Medium_net':
    net, vgg16_conv_layers = network.my_medium_model(x_deconvolved, FLAGS.batch_size, is_training=True)
elif FLAGS.Network=='Medium_net_noLog':
    net, vgg16_conv_layers = network.my_medium_model(x_deconvolved, FLAGS.batch_size, is_training=True, log_domain=False)
elif FLAGS.Network=='MediumDeep_net_noLog':
    net, vgg16_conv_layers = network.my_medium_deep_model(x_deconvolved, FLAGS.batch_size, is_training=True, log_domain=False)
elif FLAGS.Network=='Large_net_noLog':
    net, vgg16_conv_layers = network.my_large_model(x_deconvolved, FLAGS.batch_size, is_training=True, log_domain=False)
elif FLAGS.Network=='LargeDeep_net_noLog':
    net, vgg16_conv_layers = network.my_large_deep_model(x_deconvolved, FLAGS.batch_size, is_training=True, log_domain=False)
elif FLAGS.Network=='Original_noLog':
    net, vgg16_conv_layers = network.model(x_deconvolved, FLAGS.batch_size, is_training=True, log_domain=False)
elif FLAGS.Network=='Original':
    net, vgg16_conv_layers = network.model(x_deconvolved, FLAGS.batch_size, is_training=True)

if FLAGS.Train_network:
    if FLAGS.Network is "Medium_net_noLog" or "MediumDeep_net_noLog" or "Large_net_noLog" or "LargeDeep_net_noLog" or "Original_noLog":
        y = net.outputs
    else:
        y_log = net.outputs  # Log domain
        y = tf.math.exp(y_log) - eps
else:
    y = x_deconvolved
y_blended = network.get_final(net, x_deconvolved)#Linear domain


if FLAGS.Train_network:
    train_network_params = net.all_params

if FLAGS.Train_PSF and FLAGS.DirectPSF:
    if FLAGS.Tie_PSF:
        train_PSF_params = [psf_param_tied]
    else:
        train_PSF_params = [psf_param]
    train_PSF_params.append(Deconv_gamma_sqrt)


assert FLAGS.Train_PSF==True or FLAGS.Train_network==True, "Train_network or Train_PSF must be true"

# # The TensorFlow session to be used
sess = tf.InteractiveSession()

# === Load validation data =====================================================

# Load all validation images into memory
print("Loading validation data...")
y_valid = []
for i in range(len(frames_valid)):
    if i % 10 == 0:
        print("\tframe %d of %d" % (i, len(frames_valid)))

    # succ, xv, yv = img_io.load_training_pair(os.path.join(data_dir_bin, frames_valid[i]), os.path.join(data_dir_jpg, frames_valid[i].replace(".bin", ".jpg")))
    succ, _, yv = img_io.load_training_pair(os.path.join(data_dir_bin, frames_valid[i]),
                                            os.path.join(data_dir_jpg, frames_valid[i].replace(".bin", ".jpg")))
    if not succ:
        continue
    yv = yv[np.newaxis, :, :, :]

    # if i == 0:
    if len(y_valid)==0:
        y_valid = yv
    else:
        y_valid = np.concatenate((y_valid, yv), axis=0)
print("...done!\n\n")

del frames


#=== Loss function formulation ================================================

# For masked loss, only using information near saturated image regions
thr = 0.05 # Threshold for blending
msk = tf.reduce_max(y_clipped, reduction_indices=[3])
msk = tf.minimum(1.0, tf.maximum(0.0, msk-1.0+thr)/thr)
msk = tf.reshape(msk, [-1, sy, sx, 1])
msk = tf.tile(msk, [1,1,1,3])

if FLAGS.Loss=='no_alpha':
    msk=tf.ones(msk.shape)

# Loss separated into illumination and reflectance terms
if FLAGS.sep_loss:
    y_log = tf.log(y + eps)
    y_log_ = tf.log(y_clipped+eps)
    x_deconv_log = tf.log(tf.pow(x_deconvolved, 2.0)+eps)
    x_noPSF_log = tf.log(tf.pow(x_noPSF, 2.0) + eps)

    # Luminance
    lum_kernel = np.zeros((1, 1, 3, 1))
    lum_kernel[:, :, 0, 0] = 0.213
    lum_kernel[:, :, 1, 0] = 0.715
    lum_kernel[:, :, 2, 0] = 0.072
    y_lum_lin_ = tf.nn.conv2d(y_clipped, lum_kernel, [1, 1, 1, 1], padding='SAME')
    y_lum_lin = tf.nn.conv2d(y, lum_kernel, [1, 1, 1, 1], padding='SAME')
    xdeconv_lum_lin = tf.nn.conv2d(x_deconvolved, lum_kernel, [1, 1, 1, 1], padding='SAME')
    xnoPSF_lum_lin = tf.nn.conv2d(x_noPSF, lum_kernel, [1, 1, 1, 1], padding='SAME')

    # Log luminance
    y_lum_ = tf.log(y_lum_lin_ + eps)
    y_lum = tf.log(y_lum_lin + eps)
    xdeconv_lum = tf.log(xdeconv_lum_lin + eps)
    xnoPSF_lum = tf.log(xnoPSF_lum_lin + eps)

    # Gaussian kernel
    nsig = 2
    filter_size = 13
    interval = (2*nsig+1.)/(filter_size)
    ll = np.linspace(-nsig-interval/2., nsig+interval/2., filter_size+1)
    kern1d = np.diff(st.norm.cdf(ll))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()

    # Illumination, approximated by means of Gaussian filtering
    weights_g = np.zeros((filter_size, filter_size, 1, 1))
    weights_g[:, :, 0, 0] = kernel
    y_ill_ = tf.nn.conv2d(y_lum_, weights_g, [1, 1, 1, 1], padding='SAME')
    y_ill = tf.nn.conv2d(y_lum, weights_g, [1, 1, 1, 1], padding='SAME')
    xdeconv_ill = tf.nn.conv2d(xdeconv_lum, weights_g, [1, 1, 1, 1], padding='SAME')
    xnoPSF_ill = tf.nn.conv2d(xnoPSF_lum, weights_g, [1, 1, 1, 1], padding='SAME')

    # Reflectance
    y_refl_ = y_log_ - tf.tile(y_ill_, [1,1,1,3])
    y_refl = y_log - tf.tile(y_ill, [1,1,1,3])
    xdeconv_refl = x_deconv_log - tf.tile(xdeconv_ill, [1,1,1,3])
    xnoPSF_refl = x_noPSF_log - tf.tile(xnoPSF_ill, [1, 1, 1, 3])

    # Crop y and x s.t. loss doesn't care about boundary affects from the PSF
    y_illcropped=y_ill[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]
    y_ill_cropped=y_ill_[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]
    xdeconv_illcropped=xdeconv_ill[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]
    xnoPSF_illcropped = xnoPSF_ill[:, mid_pt:-mid_pt, mid_pt:-mid_pt, :]
    y_reflcropped=y_refl[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]
    y_refl_cropped=y_refl_[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]
    xdeconv_reflcropped = xdeconv_refl[:, mid_pt:-mid_pt, mid_pt:-mid_pt, :]
    xnoPSF_reflcropped = xnoPSF_refl[:, mid_pt:-mid_pt, mid_pt:-mid_pt, :]
    msk_cropped=msk[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]


    cost_net_HDR = tf.reduce_mean( ( FLAGS.lambda_ir*tf.square( tf.subtract(y_illcropped, y_ill_cropped) ) + (1.0-FLAGS.lambda_ir)*tf.square( tf.subtract(y_reflcropped, y_refl_cropped) ) )*msk_cropped )
    cost_noPSFLDR_HDR = tf.reduce_mean( ( FLAGS.lambda_ir*tf.square( tf.subtract(xnoPSF_illcropped, y_ill_cropped) ) + (1.0-FLAGS.lambda_ir)*tf.square( tf.subtract(xnoPSF_reflcropped, y_refl_cropped) ) )*msk_cropped )
    cost_DeconvLDR_HDR = tf.reduce_mean( ( FLAGS.lambda_ir*tf.square( tf.subtract(xdeconv_illcropped, y_ill_cropped) ) + (1.0-FLAGS.lambda_ir)*tf.square( tf.subtract(xnoPSF_reflcropped, y_refl_cropped) ) )*msk_cropped )
else:
    cost_net_HDR = tf.reduce_mean( tf.square( tf.subtract(tf.log(y+eps), tf.log(y_clipped+eps) )*msk ) )
    cost_noPSFLDR_HDR = tf.reduce_mean( tf.square( tf.subtract(tf.log(y_clipped+eps), tf.log(tf.pow(x_deconvolved, 2.0)+eps) )*msk ) );
    cost_DeconvLDR_HDR = tf.reduce_mean( tf.square(tf.subtract(tf.log(y_clipped + eps), tf.log(tf.pow(x_noPSF, 2.0) + eps)) * msk));


if FLAGS.Loss_wout_Crop:
    ycropped = y
    y_cropped = y_clipped
    x_noPSF_cropped = x_noPSF
    y_cropped = y_clipped
    x_deconv_cropped = x_deconvolved
else:
    ycropped = y[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]
    y_cropped = y_clipped[:,mid_pt:-mid_pt,mid_pt:-mid_pt,:]
    x_noPSF_cropped = x_noPSF[:, mid_pt:-mid_pt, mid_pt:-mid_pt, :]
    y_cropped = y_clipped[:, mid_pt:-mid_pt, mid_pt:-mid_pt, :]
    x_deconv_cropped = x_deconvolved[:, mid_pt:-mid_pt, mid_pt:-mid_pt, :]

if FLAGS.Loss=='L2-Log':
    cost_net_HDR =  tf.sqrt(tf.reduce_mean( tf.square( tf.subtract(tf.log(tf.maximum(ycropped,0.)+eps), tf.log(tf.maximum(y_cropped,0.)+eps) ) ) ))
    cost_noPSFLDR_HDR = tf.sqrt(tf.reduce_mean( tf.square( tf.subtract(tf.log(y_cropped+eps), tf.log(x_noPSF_cropped+eps) ) ) ))
    cost_DeconvLDR_HDR = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(tf.log(y_cropped + eps), tf.log(x_deconv_cropped + eps)))))
if FLAGS.Loss=='MSE':
    cost_net_HDR =  (tf.reduce_mean( tf.square( tf.subtract(ycropped, y_cropped ) ) ))
    cost_noPSFLDR_HDR = (tf.reduce_mean( tf.square( tf.subtract(y_cropped, x_noPSF_cropped ) ) ))
    cost_DeconvLDR_HDR = (tf.reduce_mean(tf.square(tf.subtract(y_cropped, x_deconv_cropped ))))
if FLAGS.Loss=='L2':#RMSE
    cost_net_HDR =  tf.sqrt(tf.reduce_mean( tf.square( tf.subtract(ycropped, y_cropped ) ) ))
    cost_noPSFLDR_HDR = tf.sqrt(tf.reduce_mean( tf.square( tf.subtract(y_cropped, x_noPSF_cropped ) ) ))
    cost_DeconvLDR_HDR = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_cropped, x_deconv_cropped ))))
if FLAGS.Loss=='L2-Gamma-Batch':
    y_cropped_gamma = tf.pow(tf.maximum(y_cropped, 0.0)+eps, 0.5)
    ycropped_gamma = tf.pow(tf.maximum(ycropped, 0.0)+eps, 0.5)
    x_noPSF_cropped_gamma = tf.pow(tf.maximum(x_noPSF_cropped, 0.0)+eps, 0.5)
    x_deconv_cropped_gamma = tf.pow(tf.maximum(x_deconv_cropped, 0.0)+eps, 0.5)
    cost_net_HDR =  tf.sqrt(tf.reduce_mean( tf.square( tf.subtract(ycropped_gamma, y_cropped_gamma ) ) ))
    cost_noPSFLDR_HDR = tf.sqrt(tf.reduce_mean( tf.square( tf.subtract(y_cropped_gamma, x_noPSF_cropped_gamma) ) ))
    cost_DeconvLDR_HDR = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_cropped_gamma, x_deconv_cropped_gamma ))))
if FLAGS.Loss=='L2-Gamma-Full':
    y_cropped_gamma = tf.pow(tf.maximum(y_cropped, 0.0)+eps, 0.5)
    ycropped_gamma = tf.pow(tf.maximum(ycropped, 0.0)+eps, 0.5)
    x_noPSF_cropped_gamma = tf.pow(tf.maximum(x_noPSF_cropped, 0.0)+eps, 0.5)
    x_deconv_cropped_gamma = tf.pow(tf.maximum(x_deconv_cropped, 0.0)+eps, 0.5)
    cost_net_HDR =  (tf.reduce_mean( tf.square( tf.subtract(ycropped_gamma, y_cropped_gamma ) ) ))
    cost_noPSFLDR_HDR = (tf.reduce_mean( tf.square( tf.subtract(y_cropped_gamma, x_noPSF_cropped_gamma) ) ))
    cost_DeconvLDR_HDR = (tf.reduce_mean(tf.square(tf.subtract(y_cropped_gamma, x_deconv_cropped_gamma ))))
if FLAGS.Loss=='L1':
    cost_net_HDR =  tf.reduce_mean( tf.abs( tf.subtract(ycropped, y_cropped ) ) )
    cost_noPSFLDR_HDR = tf.reduce_mean( tf.abs( tf.subtract(y_cropped, x_noPSF_cropped ) ) );
    cost_DeconvLDR_HDR = tf.reduce_mean(tf.abs(tf.subtract(y_cropped, x_noPSF_cropped )));
if FLAGS.Loss=='SSIM':
    cost_net_HDR = -tf.reduce_mean(tf.image.ssim(ycropped, y_cropped, max_val=1.))#Flags.HDR_max_val))
    cost_noPSFLDR_HDR = -tf.reduce_mean(tf.image.ssim(y_cropped, x_noPSF_cropped, max_val=1.))
    cost_DeconvLDR_HDR = -tf.reduce_mean(tf.image.ssim(y_cropped, x_noPSF_cropped, max_val=1.))

# Optimizer
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = FLAGS.learning_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           int(steps_per_epoch), 0.99, staircase=True)


#Add the regularization loss to the training loss
training_loss = cost_net_HDR
reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
training_loss = tf.add(training_loss, reg_loss)


## Efficient training code with multiple learning rates
# For info on using multiple LRs see https://stackoverflow.com/questions/34945554/how-to-set-layer-wise-learning-rate-in-tensorflow
if FLAGS.Train_PSF and FLAGS.Train_network:
    opt1 = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False)
    opt2 = tf.train.AdamOptimizer(FLAGS.PSF_LR_Multiplier * learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, use_locking=False)
    grads = tf.gradients(training_loss, train_network_params + train_PSF_params)#Need to make train_PSF_params into a list
    grads1 = grads[:len(train_network_params)]
    grads2 = grads[len(train_network_params):]
    train_network = opt1.apply_gradients(zip(grads1,train_network_params))
    train_psf = opt2.apply_gradients(zip(grads2,train_PSF_params))
    train_op = tf.group(train_network, train_psf)
elif FLAGS.Train_network:
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                           epsilon=1e-8, use_locking=False).minimize(training_loss, global_step=global_step,
                                                                                    var_list=train_network_params)
else:
    train_op = tf.train.AdamOptimizer(FLAGS.PSF_LR_Multiplier * learning_rate, beta1=0.9, beta2=0.999,
                                       epsilon=1e-8, use_locking=False).minimize(training_loss, global_step=global_step,
                                                                                 var_list=train_PSF_params)

#=== Data enqueueing functions ================================================

# For enqueueing of frame names
def enqueue_frames(enqueue_op, coord, frames):
    
    num_frames = len(frames)
    i, k = 0, 0

    try:
        while not coord.should_stop():
            if k >= training_samples*FLAGS.num_epochs:
                    sess.run(q_frames.close())
                    break

            if i == num_frames:
                i = 0
                if FLAGS.rand_data:
                    random.shuffle(frames)

            fname = frames[i];

            i += 1
            k += 1
            sess.run(enqueue_op, feed_dict={input_frame: fname})
    except tf.errors.OutOfRangeError:
        pass
    except Exception as e:
        coord.request_stop(e)

# For multi-threaded reading and enqueueing of frames
def load_and_enqueue(enqueue_op, coord):
    try:
        while not coord.should_stop():
            fname = sess.run(dequeue_op_frames).decode("utf-8")

            # Load HDR images
            succ, _, input_target_r = img_io.load_training_pair(os.path.join(data_dir_bin, fname), os.path.join(data_dir_jpg, fname.replace(".bin", ".jpg")))
            if not succ:
                continue
            sess.run(enqueue_op, feed_dict={input_target: input_target_r})
    except Exception as e:
        try:
            sess.run(q_train.close())
        except Exception as e:
            pass


#=== Error and output function ================================================

# For calculation of loss and output of intermediate validations images to disc
def calc_loss_and_print(x_data, y_data, print_dir, step, N, psf_val=None):
    val_loss, orig_loss, n_batch = 0, 0, 0



    for b in range(int(x_data.shape[0] / FLAGS.batch_size)):
        x_batch = x_data[b * FLAGS.batch_size:(b + 1) * FLAGS.batch_size, :, :, :]
        y_batch = y_data[b * FLAGS.batch_size:(b + 1) * FLAGS.batch_size, :, :, :]
        feed_dict = {x: x_batch, y_: y_batch}#This will overwrite my existing definition of x, which depends on y_
        err1, err2, y_predict, y_gt_clipped, M = sess.run([cost_net_HDR, cost_DeconvLDR_HDR, y, y_clipped, msk], feed_dict=feed_dict)

        y_gt =y_batch

        val_loss += err1;
        orig_loss += err2;
        n_batch += 1
        batch_dir = print_dir

        if x_data.shape[0] > x_batch.shape[0]:
            batch_dir = '%s/batch_%03d' % (print_dir, n_batch)

        if n_batch <= N or N < 0:
            if not os.path.exists(batch_dir):
                os.makedirs(batch_dir)
            for i in range(0, x_batch.shape[0]):
                yy_p = np.squeeze(y_predict[i])
                xx = np.squeeze(x_batch[i])
                yy = np.squeeze(y_gt[i])
                mm = np.squeeze(M[i])

                if FLAGS.ApplyCameraCurve:
                    # Apply inverse camera curve
                    x_lin = np.power(np.divide(0.6 * xx, np.maximum(1.6 - xx, 1e-10)), 1.0 / 0.9)
                else:
                    x_lin=xx


                # Gamma correction
                yy_p = np.power(np.maximum(yy_p, 0.0), 0.5)
                yy = np.power(np.maximum(yy, 0.0), 0.5)
                xx = np.power(np.maximum(x_lin, 0.0), 0.5)

                # Print LDR samples
                if FLAGS.print_im:
                    img_io.writeLDR(xx, "%s/%06d_%03d_in_m3.png" % (batch_dir, step, i + 1), -3)
                    img_io.writeLDR(yy, "%s/%06d_%03d_gt_m3.png" % (batch_dir, step, i + 1), -3)
                    img_io.writeLDR(yy_p, "%s/%06d_%03d_out_m3.png" % (batch_dir, step, i + 1), -3)
                    img_io.writeLDR(xx, "%s/%06d_%03d_in_0.png" % (batch_dir, step, i + 1), 0)
                    img_io.writeLDR(yy, "%s/%06d_%03d_gt_0.png" % (batch_dir, step, i + 1), 0)
                    img_io.writeLDR(yy_p, "%s/%06d_%03d_out_0.png" % (batch_dir, step, i + 1), 0)

                # Print HDR samples
                if FLAGS.print_hdr:
                    img_io.writeEXR(xx, "%s/%06d_%03d_in.exr" % (batch_dir, step, i + 1))
                    img_io.writeEXR(yy, "%s/%06d_%03d_gt.exr" % (batch_dir, step, i + 1))
                    img_io.writeEXR(yy_p, "%s/%06d_%03d_out.exr" % (batch_dir, step, i + 1))
    if psf_val is not None:
        import scipy.misc
        r_psf=psf_val[:,:,0,0]
        g_psf=psf_val[:,:,0,1]
        b_psf=psf_val[:,:,0,2]
        log_r_psf=np.log(r_psf+1e-12)
        log_r_psf=log_r_psf-log_r_psf.min()
        log_r_psf=log_r_psf/log_r_psf.max()
        log_g_psf = np.log(g_psf + 1e-12)
        log_g_psf = log_g_psf - log_g_psf.min()
        log_g_psf = log_g_psf / log_g_psf.max()
        log_b_psf = np.log(b_psf + 1e-12)
        log_b_psf = log_b_psf - log_b_psf.min()
        log_b_psf = log_b_psf / log_b_psf.max()
        r_psf = r_psf - r_psf.min()
        r_psf = r_psf / (r_psf.max() + 1e-12)
        g_psf = g_psf - g_psf.min()
        g_psf = g_psf / (g_psf.max() + 1e-12)
        b_psf = b_psf - b_psf.min()
        b_psf = b_psf / (b_psf.max() + 1e-12)
        scipy.misc.toimage(np.squeeze(r_psf), cmin=0.0, cmax=1.0).save("%s/%06d_R_PSF.png" % (print_dir, step))
        scipy.misc.toimage(np.squeeze(g_psf), cmin=0.0, cmax=1.0).save("%s/%06d_G_PSF.png" % (print_dir, step))
        scipy.misc.toimage(np.squeeze(b_psf), cmin=0.0, cmax=1.0).save("%s/%06d_B_PSF.png" % (print_dir, step))
        scipy.misc.toimage(np.squeeze(log_r_psf), cmin=0.0, cmax=1.0).save("%s/%06d_R_logPSF.png" % (print_dir, step))
        scipy.misc.toimage(np.squeeze(log_g_psf), cmin=0.0, cmax=1.0).save("%s/%06d_G_logPSF.png" % (print_dir, step))
        scipy.misc.toimage(np.squeeze(log_b_psf), cmin=0.0, cmax=1.0).save("%s/%06d_B_logPSF.png" % (print_dir, step))
    return (val_loss / n_batch, orig_loss / n_batch)

#=== Setup threads and load parameters ========================================

# Summary for Tensorboard
tf.summary.scalar("learning_rate", learning_rate)
summaries = tf.summary.merge_all()
file_writer = tf.summary.FileWriter(log_dir, sess.graph)

sess.run(tf.global_variables_initializer())

# Threads and thread coordinator
coord = tf.train.Coordinator()
thread1 = threading.Thread(target=enqueue_frames, args=[enqueue_op_frames, coord, frames_train])
thread2 = [threading.Thread(target=load_and_enqueue, args=[enqueue_op_train, coord]) for i in range(FLAGS.num_threads)]
thread1.start()
for t in thread2:
    t.start()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

# Loading model weights
if(FLAGS.load_params):
    # Load model weights
    print("\n\nLoading trained parameters from '%s'..." % FLAGS.parameters)
    load_params = tl.files.load_npz(name=FLAGS.parameters)
    try:
        tl.files.assign_params(sess, load_params, net)
    except:
        tl.files.assign_params(sess, load_params[0:-1], net)#Old versions of the code stored the PSF along with the rest of the parameters. Need to account for this.
    print("...done!\n")
elif(FLAGS.Use_VGG_init):
    # Load pretrained VGG16 weights for encoder
    print("\n\nLoading parameters for VGG16 convolutional layers, from '%s'..." % FLAGS.vgg_path)
    network.load_vgg_weights(vgg16_conv_layers, FLAGS.vgg_path, sess)
    print("...done!\n")
else:
    print("Initializing with random weights")




if(FLAGS.load_psf):
    # Load psf weights
    print("\n\nLoading psf parameters from '%s'..." % FLAGS.psf_infilename)
    # load_psf = tl.files.load_npz(name=FLAGS.psf_infilename)
    load_psf = np.load(FLAGS.psf_infilename)['psf_val']
    load_psf_param = np.load(FLAGS.psf_infilename)['psf_param_val']
    if FLAGS.load_captured_PSF:
        sess.run([effective_PSF_loaded.assign(load_psf)])
    else:
        sess.run([psf_param.assign(load_psf_param)])
    print("...done!\n")
else:
    print("Using psf defined by PSF_init")
psf_val, psf_param_val = sess.run([effective_PSF, psf_param])
np.savez(("%s/Initial_PSF.npz" % (FLAGS.psf_outdir)), psf_val=psf_val, psf_param_val=psf_param_val)


#=== Test initial conditions ==================================================
step=0
print_dir = '%s/step_%06d' % (im_dir, step)

print("Generating validation xs with current PSF value ...")
x_valid = []
for b in range(int(y_valid.shape[0] / FLAGS.batch_size)):
    y_batch = y_valid[b * FLAGS.batch_size:(b + 1) * FLAGS.batch_size, :, :, :]
    feed_dict = {y_: y_batch}
    x_valid_batch = sess.run(x, feed_dict=feed_dict)

    if b == 0:
        x_valid = x_valid_batch
    else:
        x_valid = np.concatenate((x_valid, x_valid_batch), axis=0)
psf_val, psf_param_val = sess.run([effective_PSF, psf_param])
val_loss, orig_loss = calc_loss_and_print(x_valid, y_valid, print_dir, step, FLAGS.print_batches, psf_val)



#=== Run training loop ========================================================

print("\nStarting training...\n")

step = FLAGS.start_step
train_loss = 0.0
start_time = time.time()
start_time_tot = time.time()

print_each_batch=False

# The training loop
y_examples = y_valid[0:FLAGS.batch_size, :, :, :]
# try:
while not coord.should_stop():
    step += 1

    if FLAGS.Loss=='VGG+MSE+GAN' or FLAGS.Loss=='VGG+L1+GAN' or FLAGS.Loss=='VGG+L2Gamma+GAN':
        errD, _ = sess.run([d_loss, d_optim], feed_dict={y_: y_examples})

    [cost_net_HDR_eg, cost_noPSFLDR_HDR_eg, cost_DeconvLDR_HDR_eg, reg_loss_eg] = sess.run([cost_net_HDR, cost_noPSFLDR_HDR, cost_DeconvLDR_HDR, reg_loss], feed_dict={y_: y_examples})

    print("Error with Net, Filter, and Deconv : " + str(cost_net_HDR_eg))
    print("Error with Filter and Deconv / GAN Loss: " + str(cost_DeconvLDR_HDR_eg))
    print("Error without Filter / VGG Loss: " + str(cost_noPSFLDR_HDR_eg))
    print("Regularization Loss: " + str(reg_loss_eg))
    _, err_t = sess.run([train_op,cost_net_HDR])

    if print_each_batch:
        [cost_net_HDR_eg, cost_noPSFLDR_HDR_eg, cost_DeconvLDR_HDR_eg, y_PSF_examples,x_deconv_eg, x_examples, x_noPSF_examples, y_clipped_examples, y_recons] = sess.run([cost_net_HDR, cost_noPSFLDR_HDR, cost_DeconvLDR_HDR, y_PSF, x_deconvolved, x, x_noPSF, y_clipped, y], feed_dict={y_: y_examples})
        print("Error with Net, Filter, and Deconv: " + str(cost_net_HDR_eg))
        print("Error with Filter and Deconv: " + str(cost_DeconvLDR_HDR_eg))
        print("Error without Filter: " + str(cost_noPSFLDR_HDR_eg))
        for r in range(2):  # range(FLAGS.batch_size):
            import matplotlib.pyplot as plt

            plt.subplot(221)
            imgplot = plt.imshow(y_clipped_examples[r, :, :, :])
            plt.title('Ground Truth HDR Image')
            plt.subplot(222)
            imgplot = plt.imshow(x_noPSF_examples[r, :, :, :])
            plt.title('LDR Image Without PSF (Unused)')
            plt.subplot(223)
            imgplot = plt.imshow(x_deconv_eg[r, :, :, :])
            plt.title('Deconvolved LDR Image (Input to NN)')
            plt.subplot(224)
            imgplot = plt.imshow(y_recons[r, :, :, :])
            plt.title('Reconstruction from NN')
            plt.show()


        psf_val, psf_param_val = sess.run([effective_PSF, psf_param])
        print(psf_val[mid_pt - 2:mid_pt + 3, mid_pt - 2:mid_pt + 3, 0, 0])
        #print(psf_param_val[mid_pt - 2:mid_pt + 3, mid_pt - 2:mid_pt + 3, 0, 0])
        print(psf_val[0:5, 0:5, 0, 0])
        #print(psf_param_val[0:5, 0:5, 0, 0])
        plt.subplot(121)
        imgplot = plt.imshow(psf_val[:, :, 0, 0])
        plt.title('Red Channel PSF')
        plt.subplot(122)
        imgplot = plt.imshow(np.log(psf_val[:, :, 0, 0] + 1e-12))
        plt.colorbar(imgplot)
        plt.title('Red Channel log(PSF)')
        plt.show()
        plt.subplot(121)
        imgplot = plt.imshow(psf_val[:, :, 0, 1])
        plt.title('Green Channel PSF')
        plt.subplot(122)
        imgplot = plt.imshow(np.log(psf_val[:, :, 0, 1] + 1e-12))
        plt.colorbar(imgplot)
        plt.title('Green Channel log(PSF)')
        plt.show()
        plt.subplot(121)
        imgplot = plt.imshow(psf_val[:, :, 0, 2])
        plt.title('Blue Channel PSF')
        plt.subplot(122)
        imgplot = plt.imshow(np.log(psf_val[:, :, 0, 2] + 1e-12))
        plt.colorbar(imgplot)
        plt.title('Blue Channel log(PSF)')
        plt.show()

    if FLAGS.Display_PSF:
        import matplotlib.pyplot as plt
        if ('psf_val' in locals()) and ('PSF_parameters' in locals()):
            prev_psf_val=psf_val
            prev_psf_parameters = psf_param_val
        psf_val,psf_param_val = sess.run([effective_PSF,psf_param])
        print(psf_val[mid_pt - 2:mid_pt + 3, mid_pt - 2:mid_pt + 3, 0, 0])
        #print(psf_param_val[mid_pt - 2:mid_pt + 3, mid_pt - 2:mid_pt + 3, 0, 0])
        print(psf_val[0:5,0:5, 0, 0])
        #print(psf_param_val[0:5,0:5, 0, 0])
        plt.subplot(121)
        imgplot = plt.imshow(psf_val[:, :, 0,0])
        plt.title('Red Channel PSF')
        plt.subplot(122)
        imgplot = plt.imshow(np.log(psf_val[:, :, 0,0]+1e-12))
        plt.colorbar(imgplot)
        plt.title('Red Channel log(PSF)')
        plt.show()
        plt.subplot(121)
        imgplot = plt.imshow(psf_val[:, :, 0,1])
        plt.title('Green Channel PSF')
        plt.subplot(122)
        imgplot = plt.imshow(np.log(psf_val[:, :, 0,1]+1e-12))
        plt.colorbar(imgplot)
        plt.title('Green Channel log(PSF)')
        plt.show()
        plt.subplot(121)
        imgplot = plt.imshow(psf_val[:, :, 0,2])
        plt.title('Blue Channel PSF')
        plt.subplot(122)
        imgplot = plt.imshow(np.log(psf_val[:, :, 0,2]+1e-12))
        plt.colorbar(imgplot)
        plt.title('Blue Channel log(PSF)')
        plt.show()
        if np.isnan(psf_val).any():
            print('NaN observed')



    train_loss += err_t

    # Statistics on intermediate progress
    v = int(max(1.0,FLAGS.print_batch_freq/5.0))
    if (int(step) % v)  == 0:
        val_loss, n_batch = 0, 0

        # Validation loss
        for b in range(int(y_valid.shape[0]/FLAGS.batch_size)):
            y_batch = y_valid[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size,:,:,:]
            feed_dict = {y_: y_batch}
            err = sess.run(cost_net_HDR, feed_dict=feed_dict)
            val_loss += err; n_batch += 1

        # Training and validation loss for Tensorboard
        train_summary = tf.Summary()
        valid_summary = tf.Summary()
        valid_summary.value.add(tag='validation_loss',simple_value=val_loss/n_batch)
        file_writer.add_summary(valid_summary, step)
        train_summary.value.add(tag='training_loss',simple_value=train_loss/v)
        file_writer.add_summary(train_summary, step)

        # Other statistics for Tensorboard
        summary = sess.run(summaries)
        file_writer.add_summary(summary, step)
        file_writer.flush()

        # Intermediate training statistics
        print('  [Step %06d of %06d. Processed %06d of %06d samples. Train loss = %0.6f, valid loss = %0.6f]' % (step, steps_per_epoch*FLAGS.num_epochs, (step % steps_per_epoch)*FLAGS.batch_size, training_samples, train_loss/v, val_loss/n_batch))
        psf_val,psf_param_val = sess.run([effective_PSF,psf_param])
        print(psf_val[mid_pt - 2:mid_pt + 3, mid_pt - 2:mid_pt + 3, 0, 0])
        print(psf_val[0:5,0:5, 0, 0])
        train_loss = 0.0

    # Print statistics, and save weights and some validation images
    if step % FLAGS.print_batch_freq == 0:
    # if True:
        duration = time.time() - start_time
        duration_tot = time.time() - start_time_tot

        print_dir = '%s/step_%06d' % (im_dir, step)

        print("Generating validation xs with current PSF value ...")
        x_valid = []
        for b in range(int(y_valid.shape[0] / FLAGS.batch_size)):
            y_batch = y_valid[b * FLAGS.batch_size:(b + 1) * FLAGS.batch_size, :, :, :]
            feed_dict = {y_: y_batch}
            x_valid_batch = sess.run(x, feed_dict=feed_dict)

            if b == 0:
                x_valid = x_valid_batch
            else:
                x_valid = np.concatenate((x_valid, x_valid_batch), axis=0)
        psf_val, psf_param_val = sess.run([effective_PSF, psf_param])
        val_loss, orig_loss = calc_loss_and_print(x_valid, y_valid, print_dir, step, FLAGS.print_batches, psf_val)

        # Training statistics
        print('\n')
        print('-------------------------------------------')
        print('Currently at epoch %0.2f of %d.' % (step/steps_per_epoch, FLAGS.num_epochs))
        print('Valid loss input   = %.5f' % (orig_loss))
        print('Valid loss trained = %.5f' % (val_loss))
        print('Timings:')
        print('       Since last: %.3f sec' % (duration))
        print('         Per step: %.3f sec' % (duration/FLAGS.print_batch_freq))
        print('        Per epoch: %.3f sec' % (duration*steps_per_epoch/FLAGS.print_batch_freq))
        print('')
        print('   Per step (avg): %.3f sec' % (duration_tot/step))
        print('  Per epoch (avg): %.3f sec' % (duration_tot*steps_per_epoch/step))
        print('')
        print('       Total time: %.3f sec' % (duration_tot))
        print('   Exp. time left: %.3f sec' % (duration_tot*steps_per_epoch*FLAGS.num_epochs/step - duration_tot))
        print('-------------------------------------------')

        # Save current weights
        tl.files.save_npz(net.all_params , name=("%s/model_step_%06d.npz"%(log_dir,step)))
        psf_val,psf_param_val = sess.run([effective_PSF,psf_param])
        np.savez(("%s/psf_step_%06d.npz"%(FLAGS.psf_outdir,step)),psf_val=psf_val,psf_param_val=psf_param_val)
        print('\n')
        if FLAGS.Loss == 'VGG+MSE+GAN' or FLAGS.Loss == 'VGG+L1+GAN' or FLAGS.Loss=='VGG+L2Gamma+GAN':
            tl.files.save_npz(net_d.all_params , name=("%s/descriminator_step_%06d.npz"%(log_dir,step)))
        start_time = time.time()

#=== Final stats and weights ==================================================

print_dir = '%s/step_%06d' % (im_dir, step)
x_valid = sess.run(x,feed_dict={y_:y_valid[0:FLAGS.batch_size,:,:,:]})
psf_val, psf_param_val = sess.run([effective_PSF, psf_param])
val_loss, orig_loss = calc_loss_and_print(x_valid, y_valid, print_dir, step, FLAGS.print_batches,psf_val)

# Final statistics
print('\n')
print('-------------------------------------------')
print('Finished at epoch %0.2f of %d.' % (step/steps_per_epoch, FLAGS.num_epochs))
print('Valid loss input   = %.5f' % (orig_loss))
print('Valid loss trained = %.5f' % (val_loss))
print('Timings:')
print('   Per step (avg): %.3f sec' % (duration_tot/step))
print('  Per epoch (avg): %.3f sec' % (duration_tot*steps_per_epoch/step))
print('')
print('       Total time: %.3f sec' % (duration_tot))
print('-------------------------------------------')

# Save final weights
tl.files.save_npz(net.all_params , name=("%s/model_step_%06d.npz"%(log_dir,step)))
if FLAGS.Loss == 'VGG+MSE+GAN' or FLAGS.Loss == 'VGG+L1+GAN' or FLAGS.Loss=='VGG+L2Gamma+GAN':
    tl.files.save_npz(net_d.all_params, name=("%s/descriminator_step_%06d.npz" % (log_dir, step)))
print('\n')


#=== Shut down ================================================================

# Stop threads
print("Shutting down threads...")
try:
    coord.request_stop()
except Exception as e:
    print("ERROR: ", e)

# Wait for threads to finish
print("Waiting for threads...")
coord.join(threads)

file_writer.close()
sess.close()