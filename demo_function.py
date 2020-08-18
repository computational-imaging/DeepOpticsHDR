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
 " Description: TensorFlow prediction script, for reconstructing HDR images
                from single expousure LDR images.
 " Author: Gabriel Eilertsen, gabriel.eilertsen@liu.se
 " Date: Aug 2017
"""
import os, sys
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from src import network
from utils import img_io
from skimage.transform import resize

from PIL import Image


eps = 1e-5

def print_(str, color='', bold=False):
    if color == 'w':
        sys.stdout.write('\033[93m')
    elif color == "e":
        sys.stdout.write('\033[91m')
    elif color == "m":
        sys.stdout.write('\033[95m')

    if bold:
        sys.stdout.write('\033[1m')

    sys.stdout.write(str)
    sys.stdout.write('\033[0m')
    sys.stdout.flush()

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("TestNumber", "2", "Use lookup table to determine which test parameters to use")
tf.flags.DEFINE_float("gamma", "1.0", "Gamma/exponential curve applied before, and inverted after, prediction. This can be used to control the boost of reconstructed pixels.")
tf.flags.DEFINE_string("mynet_params", "PretrainedNetworks/RealPSF_Network/logs/model_step_515000.npz", "Path to trained CNN weights")
tf.flags.DEFINE_string("suffix", "_RealPSF_Network", "Path to trained CNN weights")

mynet_params = FLAGS.mynet_params

if FLAGS.TestNumber == 1:
    input_dir = "ExperimentalData/Fruit"

    orig_exp = 2.5
    reduced_exps = [15., 10., 5., 2.5, 1., 1./2., 1./4., 1./8., 1./20.,1./30.,1./50.]#[1./50., 1. / 80.]
    exp_diffs = [np.log2(i / orig_exp) for i in reduced_exps]

    woutPSF_filenames = ['%s/woutPSF_%.2f_stop.png' % (input_dir, exp_diff) for exp_diff in exp_diffs]

    out_dir = "Reconstructions/Fruit/"  # , "Path to output directory"
if FLAGS.TestNumber == 2:
    input_dir = "ExperimentalData/Outdoors"

    orig_exp = 30.
    reduced_exps = [30.,20.,10.,2., 1., 1./2.]
    exp_diffs = [np.log2(i / orig_exp) for i in reduced_exps]

    woutPSF_filenames = ['%s/woutPSF_%.2f_stop.png' % (input_dir, exp_diff) for exp_diff in exp_diffs]

    out_dir = "Reconstructions/Outdoors/"  # , "Path to output directory"

if FLAGS.TestNumber == 3:
    input_dir = "ExperimentalData/Kitchen"

    orig_exp = 1./2.
    reduced_exps = [5., 2., 1., 1./2.,1./4.,1./8.,1./10.,1./20.,1./30.,1./50., 1./100.]
    exp_diffs = [np.log2(i / orig_exp) for i in reduced_exps]

    woutPSF_filenames = ['%s/woutPSF_%.2f_stop.png' % (input_dir, exp_diff) for exp_diff in exp_diffs]

    out_dir = "Reconstructions/Kitchen/"  # , "Path to output directory"
if FLAGS.TestNumber == 4:
    input_dir = "ExperimentalData/David"

    orig_exp = 1./2.
    reduced_exps = [5., 2., 1., 1./2., 1./4., 1./10., 1./20., 1./30., 1./50., 1./100., 1./500.]#[1./50., 1. / 80.]
    exp_diffs = [np.log2(i / orig_exp) for i in reduced_exps]

    woutPSF_filenames = ['%s/woutPSF_%.2f_stop.png' % (input_dir, exp_diff) for exp_diff in exp_diffs]

    out_dir = "Reconstructions/David/"  # , "Path to output directory"

#Read in the two images
img_noPSF_name = '%s/woutPSF_%.2f_stop.png' % (input_dir, 0.)
img_wPSF_name = '%s/wPSF_0_stop.png' % (input_dir)
x_wPSF = np.array(Image.open(img_wPSF_name),dtype=np.float64)/255.
x_noPSF = np.array(Image.open(img_noPSF_name),dtype=np.float64)/255.
x_wPSF=np.expand_dims(x_wPSF,axis=0)
x_noPSF=np.expand_dims(x_noPSF,axis=0)


width = 640
height = 640

x_tf = tf.placeholder(tf.float32, shape=[1, height, width, 3])
x_wPSF_tf = tf.placeholder(tf.float32, shape=[1, height, width, 3])

# HDR reconstruction autoencoder model
print_("Network setup:\n")
net = network.my_medium_model(x_wPSF_tf, batch_size=1, is_training=False, log_domain=False)
y = net.outputs

# TensorFlow session for running inference
sess = tf.InteractiveSession()

suffix=FLAGS.suffix
# Load trained my network and PSF
print_("\nLoading trained parameters from '%s'..." % mynet_params)
load_params = tl.files.load_npz(name=mynet_params)
tl.files.assign_params(sess, load_params, net)
print_("\tdone\n")

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

def crop_and_scale(y_buffer, sz):
    sz_in = [float(x) for x in y_buffer.shape]
    sz_out = [float(x) for x in sz]

    r_in = sz_in[1] / sz_in[0]
    r_out = sz_out[1] / sz_out[0]

    if r_out / r_in > 1.0:
        sx = sz_in[1]
        sy = sx / r_out
    else:
        sy = sz_in[0]
        sx = sy * r_out

    yo = np.maximum(0.0, (sz_in[0] - sy) / 2.0)
    xo = np.maximum(0.0, (sz_in[1] - sx) / 2.0)

    y_buffer = y_buffer[int(yo):int(yo + sy), int(xo):int(xo + sx), :]
    y_buffer = resize(y_buffer, sz, anti_aliasing=True)
    y_buffer = y_buffer[np.newaxis, :, :, :]
    return y_buffer


print_("\nStarting prediction...\n\n")
k = 0

# Read frame

print_("\t(Saturation: %0.2f%%)\n" % (100.0*(x_noPSF>=1).sum()/x_noPSF.size), 'm')

#Predict with my code
print_("\tInference...")
feed_dict = {x_wPSF_tf: x_wPSF}
y_predict = sess.run([y], feed_dict=feed_dict)
y_predict = np.power(np.maximum(y_predict, 0.0), FLAGS.gamma)

print_("\tdone\n")


# Write to disc
print_("\tWriting...")
k += 1;
# Gamma corrected output
y_gamma = np.power(np.maximum(y_predict, 0.0), 0.5)
xwPSF_gamma = np.power(np.maximum(x_wPSF, 0.0), 0.5)
x_gamma = np.power(np.maximum(x_noPSF, 0.0), 0.5)

for i in range(len(reduced_exps)):
    exp_diff=exp_diffs[i]
    woutPSF_filename=woutPSF_filenames[i]
    x_noPSF_reduced_exp = np.array(Image.open(woutPSF_filename),dtype=np.float64)/255.
    x_noPSF_reduced_exp_gamma=np.power(np.maximum(x_noPSF_reduced_exp, 0.0), 0.5)

    img_io.writeLDR(xwPSF_gamma, '%s/input_wPSF_synth_%.2f_stop.png' % (out_dir, exp_diff), exp_diff)
    img_io.writeLDR(x_gamma, '%s/input_woutPSF_synth_%.2f_stop.png' % (out_dir, exp_diff), exp_diff)
    img_io.writeLDR(x_noPSF_reduced_exp_gamma, '%s/GT_%.2f_stop.png' % (out_dir, exp_diff), 0.)
    img_io.writeLDR(y_gamma, '%s/recon%s_%.2f_stop.png' % (out_dir, suffix, exp_diff), exp_diff)


#Synthesize reduced exposure images of the input with a PSF
img_io.writeLDR(xwPSF_gamma, '%s/input_wPSF_synth_%.2f_stop.png' % (out_dir, 0), 0)

#Synthesize reduced exposure images of the input without a PSF
img_io.writeLDR(x_gamma, '%s/input_woutPSF_synth_%.2f_stop.png' % (out_dir, 0), 0.)

#Save actual low exposure images of the GT
img_io.writeLDR(x_gamma, '%s/GT_%.2f_stop.png' % (out_dir, 0.), 0.)

#Save reconstructed images
img_io.writeLDR(y_gamma, '%s/recon%s_%.2f_stop.png' % (out_dir, suffix, 0.), 0.)

img_io.writeEXR(y_predict, '%s/hdr%s.exr' % (out_dir, suffix))

print_("Done!\n")

sess.close()

