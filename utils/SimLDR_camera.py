#Chris Metzler 2020

import numpy as np
import tensorflow as tf
from numpy.fft import ifftshift

def SimLDR_camera(HDR_input,psf_param,effective_PSF=None,HDR_max_val=1e15,apply_camera_curve=False,noise_std=2**-10,pad_symmetric=False,clip=False):
    #=HDR_input#Batch_size x height x width x n_channels

    # #Apply PSF
    if effective_PSF is None:
        h_f,w_f,n_c,_=psf_param.shape.as_list()
        psf_param=tf.reshape(psf_param,[h_f,w_f,1,n_c])
        effective_PSF=tf.sigmoid(psf_param)
        if clip:
        	effective_PSF=effective_PSF*tf.dtypes.cast(effective_PSF>5e-5,tf.float32)#Clip the really tiny values to prevent the network from cheating the sum to 1 constraint
        effective_PSF = tf.div(effective_PSF, tf.reduce_sum(effective_PSF, axis=[0,1], keep_dims=True))

    HDR_clipped = tf.clip_by_value(HDR_input, 1e-5, HDR_max_val)
    HDR_filtered = my_img_psf_conv(HDR_clipped,effective_PSF)
    HDR_filtered = tf.clip_by_value(HDR_filtered, 1e-5, 1e10)


    if apply_camera_curve:
        # #Apply PSF-dependent camera curve
        n=.9
        sigma = .6
        Hn=HDR_filtered**n
        LDR_output=(1.+sigma)*(Hn/(Hn+sigma))

        #Apply camera curve to HDR image that has not been filtered
        n=.9
        sigma = .6
        Hn=HDR_clipped**n
        LDR_no_PSF=(1.+sigma)*(Hn/(Hn+sigma))
    else:
        LDR_output=HDR_filtered
        LDR_no_PSF=HDR_clipped

    #Read noise
    LDR_output = LDR_output + np.random.randn(*LDR_output.shape)*noise_std
    LDR_no_PSF = LDR_no_PSF + np.random.randn(*LDR_no_PSF.shape)*noise_std

    #Clip the saturated pixels
    LDR_output=tf.clip_by_value(LDR_output,1e-5, 1.)
    LDR_no_PSF=tf.clip_by_value(LDR_no_PSF, 1e-5, 1.)

    return LDR_output, LDR_no_PSF, HDR_filtered, effective_PSF, HDR_clipped


def inverse_filter(blurred, estimate, psf, gamma=None, init_gamma=2.):
    """Inverse filtering in the frequency domain.

    Args:
        blurred: image with shape (batch_size, height, width, num_img_channels)
        estimate: image with shape (batch_size, height, width, num_img_channels)
        psf: filters with shape (kernel_height, kernel_width, num_img_channels, num_filters)
        gamma: Optional. Scalar that determines regularization (higher --> more regularization, output is closer to
               "estimate", lower --> less regularization, output is closer to straight inverse filtered-result). If
               not passed, a trainable variable will be created.
        init_gamma: Optional. Scalar that determines the square root of the initial value of gamma.
    """
    img_shape = blurred.shape.as_list()

    if gamma is None:  # Gamma (the regularization parameter) is also a trainable parameter.
        gamma_initializer = tf.constant_initializer(init_gamma)
        gamma = tf.get_variable(name="gamma",
                                shape=(),
                                dtype=tf.float32,
                                trainable=True,
                                initializer=gamma_initializer)
        gamma = tf.square(gamma)  # Enforces positivity of gamma.
        tf.summary.scalar('gamma', gamma)

    a_tensor_transp = tf.transpose(blurred, [0, 3, 1, 2])
    estimate_transp = tf.transpose(estimate, [0, 3, 1, 2])

    # Everything has shape (batch_size, num_channels, height, width)
    img_fft = tf.fft2d(tf.complex(a_tensor_transp, 0.))
    # otf = my_psf2otf(psf, output_size=img_shape[1:3])
    otf = psf2otf(psf, output_size=img_shape[1:3])
    otf = tf.transpose(otf, [2, 3, 0, 1])

    adj_conv = img_fft * tf.conj(otf)

    # This is a slight modification to standard inverse filtering - gamma not only regularizes the inverse filtering,
    # but also trades off between the regularized inverse filter and the unfiltered estimate_transp.
    numerator = adj_conv + tf.fft2d(tf.complex(gamma * estimate_transp, 0.))

    kernel_mags = tf.square(tf.abs(otf))  # Magnitudes of the blur kernel.

    denominator = tf.complex(kernel_mags + gamma, 0.0)
    filtered = tf.div(numerator, denominator)
    cplx_result = tf.ifft2d(filtered)
    real_result = tf.real(cplx_result)  # Discard complex parts.
    real_result = tf.maximum(1e-5,real_result)

    # Get back to (batch_size, num_channels, height, width)
    result = tf.transpose(real_result, [0, 2, 3, 1])
    return result


def WienerFilter(filtered_input,psf_param,SNR=1e5,effective_PSF=None):
    img_shape = filtered_input.shape.as_list()
    if effective_PSF is None:
        h_f,w_f,n_c,_=psf_param.shape.as_list()
        psf_param=tf.reshape(psf_param,[h_f,w_f,1,n_c])
        effective_PSF=tf.sigmoid(psf_param)
    output_size=[2*img_shape[1],2*img_shape[2]]
    H=my_psf2otf(effective_PSF,2*output_size)
    H_abs2 = tf.cast(tf.abs(H)**2, tf.complex64)
    H_wiener=1./H*(H_abs2/(H_abs2+tf.cast(1./SNR,tf.complex64)))
    H_wiener = tf.transpose(H_wiener, [2, 0, 1, 3])
    deconvolved_input=my_img_psf_conv(filtered_input,psf=None,otf=H_wiener)
    deconvolved_input=tf.maximum(1e-5,deconvolved_input)
    #Even with non-negative PSFs, numerical errors can cause FFT-based conv to produce negative values. This will produce NaNs with the logs.
    return deconvolved_input

### New optics functions
def my_img_psf_conv(img, psf, otf=None, adjoint=False, circular=False,pad_symmetric=False):
    #Uses RFFT2D
    '''Performs a convolution of an image and a psf in frequency space.

    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    '''
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    img_shape = img.shape.as_list()

    if not circular:
        target_side_length = 2 * img_shape[1]

        # height_pad = (target_side_length - img_shape[1]) / 2
        height_pad = (target_side_length - img_shape[1]) / 2
        width_pad = (target_side_length - img_shape[2]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        if pad_symmetric:
            img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "SYMMETRIC")
        else:
            img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")

        img_shape = img.shape.as_list()

    img_fft = transp_rfft2d(img)

    if otf is None:
        psf = tf.convert_to_tensor(psf, dtype=tf.float32)
        otf = my_psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(otf, [2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transp_irfft2d(img_fft * tf.conj(otf))
    else:
        result = transp_irfft2d(img_fft * otf)

    #result = tf.abs(result)#For some reason this prevents divergence
    #result = tf.cast(tf.real(result), tf.float32)#Shouldn't be necessary with irfft2d

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result

def my_psf2otf(input_filter, output_size):
    '''Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    '''
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(input_filter, [[pad_top, pad_bottom],
                                       [pad_left, pad_right], [0, 0], [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(padded, [2, 0, 1, 3])
    padded = ifftshift2d_tf(padded)
    padded = tf.transpose(padded, [1, 2, 0, 3])

    ## Take FFT
    tmp = tf.transpose(padded, [2, 3, 0, 1])
    # tmp = tf.fft2d(tf.complex(tmp, 0.))
    tmp = tf.spectral.rfft2d(tmp)
    return tf.transpose(tmp, [2, 3, 0, 1])

def transp_rfft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    # a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.spectral.rfft2d(a_tensor_transp)
    # a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d

def transp_irfft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.spectral.irfft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    # a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d

#### Old optics functions
def img_psf_conv(img, psf, otf=None, adjoint=False, circular=False):
    '''Performs a convolution of an image and a psf in frequency space.

    :param img: Image tensor.
    :param psf: PSF tensor.
    :param otf: If OTF is already computed, the otf.
    :param adjoint: Whether to perform an adjoint convolution or not.
    :param circular: Whether to perform a circular convolution or not.
    :return: Image convolved with PSF.
    '''
    img = tf.convert_to_tensor(img, dtype=tf.float32)

    img_shape = img.shape.as_list()

    if not circular:
        target_side_length = 2 * img_shape[1]

        # height_pad = (target_side_length - img_shape[1]) / 2
        height_pad = (target_side_length - img_shape[1]) / 2
        width_pad = (target_side_length - img_shape[2]) / 2

        pad_top, pad_bottom = int(np.ceil(height_pad)), int(np.floor(height_pad))
        pad_left, pad_right = int(np.ceil(width_pad)), int(np.floor(width_pad))

        img = tf.pad(img, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], "CONSTANT")
        img_shape = img.shape.as_list()

    img_fft = transp_fft2d(img)

    if otf is None:
        psf = tf.convert_to_tensor(psf, dtype=tf.float32)
        otf = psf2otf(psf, output_size=img_shape[1:3])
        otf = tf.transpose(otf, [2, 0, 1, 3])

    otf = tf.cast(otf, tf.complex64)
    img_fft = tf.cast(img_fft, tf.complex64)

    if adjoint:
        result = transp_ifft2d(img_fft * tf.conj(otf))
    else:
        result = transp_ifft2d(img_fft * otf)

    result = tf.cast(tf.real(result), tf.float32)

    if not circular:
        result = result[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return result

def psf2otf(input_filter, output_size):
    '''Convert 4D tensorflow filter into its FFT.

    :param input_filter: PSF. Shape (height, width, num_color_channels, num_color_channels)
    :param output_size: Size of the output OTF.
    :return: The otf.
    '''
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    fh, fw, _, _ = input_filter.shape.as_list()

    if output_size[0] != fh:
        pad = (output_size[0] - fh) / 2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = tf.pad(input_filter, [[pad_top, pad_bottom],
                                       [pad_left, pad_right], [0, 0], [0, 0]], "CONSTANT")
    else:
        padded = input_filter

    padded = tf.transpose(padded, [2, 0, 1, 3])
    padded = ifftshift2d_tf(padded)
    padded = tf.transpose(padded, [1, 2, 0, 3])

    ## Take FFT
    tmp = tf.transpose(padded, [2, 3, 0, 1])
    tmp = tf.fft2d(tf.complex(tmp, 0.))
    return tf.transpose(tmp, [2, 3, 0, 1])

def transp_ifft2d(a_tensor, dtype=tf.complex64):
    a_tensor = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_tensor = tf.cast(a_tensor, tf.complex64)
    a_ifft2d_transp = tf.ifft2d(a_tensor)
    # Transpose back to [batch_size, x, y, channels]
    a_ifft2d = tf.transpose(a_ifft2d_transp, [0, 2, 3, 1])
    a_ifft2d = tf.cast(a_ifft2d, dtype)
    return a_ifft2d

def ifftshift2d_tf(a_tensor):
    input_shape = a_tensor.shape.as_list()

    new_tensor = a_tensor
    for axis in range(1, 3):
        n = input_shape[axis]
        split = n - (n + 1) // 2
        mylist = np.concatenate((np.arange(split, n), np.arange(split)))
        new_tensor = tf.gather(new_tensor, mylist, axis=axis)
    return new_tensor

def transp_fft2d(a_tensor, dtype=tf.complex64):
    """Takes images of shape [batch_size, x, y, channels] and transposes them
    correctly for tensorflows fft2d to work.
    """
    # Tensorflow's fft only supports complex64 dtype
    a_tensor = tf.cast(a_tensor, tf.complex64)
    # Tensorflow's FFT operates on the two innermost (last two!) dimensions
    a_tensor_transp = tf.transpose(a_tensor, [0, 3, 1, 2])
    a_fft2d = tf.fft2d(a_tensor_transp)
    a_fft2d = tf.cast(a_fft2d, dtype)
    a_fft2d = tf.transpose(a_fft2d, [0, 2, 3, 1])
    return a_fft2d
