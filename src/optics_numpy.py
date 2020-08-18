#Julie Chang and Chris Metzler 2020
import abc

# import tensorflow as tf
import numpy as np
# import matplotlib as mpl
# mpl.use('TKAgg')
import matplotlib.pyplot as plt
from PIL import Image

from numpy.fft import ifftshift
import fractions
# import layers.optics_no_transpose as optics
#import optics_no_transpose as optics
from skimage.transform import resize
from skimage.measure import block_reduce
from scipy.ndimage import gaussian_filter
# from scipy.interpolate import RectBivariateSpline
import scipy.interpolate as interp
from skimage.io import imsave

def phaseshifts_from_height_map(height_map, wave_lengths, refractive_idcs, dtype=np.complex64):
    '''Calculates the phase shifts created by a height map with certain
    refractive index for light with specific wave length.
    '''
    # refractive index difference
    delta_N = refractive_idcs.reshape([1,-1,1,1]) - 1.
    # wave number
    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1,-1,1,1])
    # phase delay indiced by height field
    phi = wave_nos * delta_N * height_map
    phase_shifts = np.exp(1j*phi)
    return phase_shifts

def get_vanilla_zernike_height_map(zernike_volume, zernike_coeffs, output_resolution=None):
    heightmap_zernike = np.sum(zernike_coeffs * zernike_volume, axis=0)
    if output_resolution is not None:
        heightmap_zernike = resize(heightmap_zernike, output_resolution)
    return heightmap_zernike

class PhasePlate():
    def __init__(self,
                 wave_lengths,
                 height_map,
                 refractive_idcs,
                 height_tolerance=None,
                 lateral_tolerance=None,
                 dtype=np.complex64):

        self.wave_lengths = wave_lengths
        self.height_map = height_map
        self.resolution = np.array(np.shape(height_map))
        self.refractive_idcs=refractive_idcs
        self.height_tolerance=height_tolerance
        self.lateral_tolerance=lateral_tolerance
        self.dtype = dtype

    def __call__(self, input_field):
        # Add manufacturing tolerances in the form of height map noise
        if self.height_tolerance is not None:
            self.height_map += np.random.uniform(low=-self.height_tolerance,
                                                 high=self.height_tolerance,
                                                 size=self.height_map.shape)
            print("Phase plate with manufacturing tolerance %0.2e"%self.height_tolerance)

        self.phase_shifts = phaseshifts_from_height_map(self.height_map,
                                                        self.wave_lengths,
                                                        self.refractive_idcs,
                                                        dtype=self.dtype)

        input_field = input_field.astype(self.dtype)
        return input_field * self.phase_shifts

def psf2otf(input_filter, output_size):
    """Convert 4D tensorflow filter into its FFT.
    Input shape: [in_channels, out_channels, height, width]
    """
    # pad out to output_size with zeros
    # circularly shift so center pixel is at 0,0
    _, _, fh, fw = np.shape(input_filter)
    
    if output_size[0] != fh:
        pad = (output_size[0] - fh)/2

        if (output_size[0] - fh) % 2 != 0:
            pad_top = pad_left = int(np.ceil(pad))
            pad_bottom = pad_right = int(np.floor(pad))
        else:
            pad_top = pad_left = int(pad) + 1
            pad_bottom = pad_right = int(pad) - 1

        padded = np.pad(input_filter, ((0,0), (0,0), (pad_top, pad_bottom),
                                      (pad_left, pad_right)), mode='constant')
    else:
        padded = input_filter

    padded = np.fft.ifftshift(padded, axes=(2,3))
    tmp = np.fft.fft2(padded)

    return tmp

def propagate_exact(input_field, kernels):

    _, _, M_orig, N_orig = np.shape(input_field)

    # zero padding.
    Mpad = M_orig//2
    Npad = N_orig//2

    M = M_orig + 2*Mpad
    N = N_orig + 2*Npad

    padded_input_field = np.pad(input_field,
                                ((0,0), (0,0), (Mpad,Mpad), (Npad,Npad)),
                                mode='constant')

    objFT = np.fft.fft2(padded_input_field)
    out_field = np.fft.ifft2( objFT * kernels)

    out_field = out_field[:,:,Npad:-Npad,Npad:-Npad]

    return out_field

def plano_convex_initializer(focal_length,
                             wave_lengths,
                             wave_resolution,
                             discretization_step,
                             refractive_idx):
    convex_radius = (refractive_idx - 1.) * focal_length
    N,M = wave_resolution
    [x, y] = np.mgrid[-N//2:N//2,
                      -M//2:M//2].astype(np.float64)

    x = x * discretization_step
    y = y * discretization_step
    x = x.reshape([N,M])
    y = y.reshape([N,M])

    # This approximates the spherical surface with qaudratic-phase surfaces.
    height_map = -(x ** 2 + y ** 2) / 2. * (1. / convex_radius)
    # height_map = np.mod(height_map, get_one_phase_shift_thickness(wave_lengths[0], refractive_idcs[0]))
    # return tf.constant(np.sqrt(height_map), dtype=dtype)
        
    return height_map

def circular_aperture(input_field, r_cutoff=None):
    try:
        input_shape = np.shape(input_field)
    except:
        input_shape = input_field.shape

    [x, y] = np.mgrid[-input_shape[2] // 2: input_shape[2] // 2,
                      -input_shape[3] // 2: input_shape[3] // 2].astype(np.float64)

    if r_cutoff is None:
        r_cutoff = np.amax(x)

    r = np.sqrt(x ** 2 + y ** 2)[None,None,:,:]
    aperture = (r<r_cutoff).astype(np.float32)
    return aperture * input_field

def get_psfs(optical_element,
             depth_values,
             wave_lengths,
             optical_feature_size,
             sensor_distance,
             propagation_kernel,
             psf_resolution=None,
             sampling_factor=None,
             use_circular_aperture=True,
             r_cutoff=None,
             amplitude_mask=None,
             use_planar_incidence=False,
             dtype=np.complex64,
             sigma=None,
             get_otfs=True,
             otf_resolution=None):

    wave_resolution = optical_element.resolution
    physical_size = wave_resolution[0] * optical_feature_size
    # what about magnification
    
    N, M = wave_resolution
    [x, y] = np.mgrid[-N//2:N//2,
                      -M//2:M//2].astype(np.float64)

    x = x/N * physical_size
    y = y/M * physical_size

    squared_sum = x**2 + y**2
    squared_sum = squared_sum[None,None,:,:]

    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1,-1,1,1])

    input_fields = np.tile(squared_sum, [len(depth_values), len(wave_lengths), 1, 1])
    input_fields = np.sqrt(input_fields + np.array(depth_values).reshape([-1, 1, 1, 1])**2)
    input_fields = np.exp(1.j * wave_nos * input_fields)

    if use_circular_aperture:
        input_fields = circular_aperture(input_fields, r_cutoff)
    if amplitude_mask is not None:
        input_fields = input_fields * amplitude_mask

    psfs = []
    otfs = []
    # calculate PSF for each depth
    for depth_idx in range(len(depth_values)):
        # propagate through optical element
        input_field = input_fields[depth_idx:depth_idx+1,:,:,:]
        field = optical_element(input_field)

        # propagate field to sensor
        sensor_incident_field = propagate_exact(field, propagation_kernel)
        psf = np.square(np.abs(sensor_incident_field))
        psf_edit = []
        for wavelength in range(np.shape(psf)[1]):
            psf_image = np.squeeze(psf[0,wavelength,:,:])
            if psf_resolution is not None:
                psf_image = np.array(Image.fromarray(psf_image).resize((psf_resolution[0], psf_resolution[1]),
                                                                       resample=Image.BILINEAR))
            if sampling_factor is not None:
                psf_image = block_reduce(psf_image, block_size=(sampling_factor,sampling_factor), func=np.mean)
            if sigma is not None:
                psf_image = gaussian_filter(psf_image, sigma)
            psf_image /= np.sum(psf_image)
            psf_edit.append(np.expand_dims(np.expand_dims(psf_image, axis=0), axis=0))
            
        psf = np.concatenate(psf_edit, axis=1)
        psfs.append(psf)
        
        # calculate OTF as well
        if get_otfs:
            if otf_resolution is None:
                otf_resolution = np.shape(psf)[2:3]
            otf = psf2otf(psf, otf_resolution)
            otfs.append(otf)

    return psfs, otfs


def get_psfs_coherent(optical_element,
             depth_values,
             wave_lengths,
             optical_feature_size,
             sensor_distance,
             propagation_kernel,
             psf_resolution=None,
             use_circular_aperture=True,
             r_cutoff=None,
             use_planar_incidence=False,
             dtype=np.complex64,
             get_otfs=True,
             otf_resolution=None):

    wave_resolution = optical_element.resolution
    physical_size = wave_resolution[0] * optical_feature_size
    # what about magnification
    
    N, M = wave_resolution
    [x, y] = np.mgrid[-N//2:N//2,
                      -M//2:M//2].astype(np.float64)

    x = x/N * physical_size
    y = y/M * physical_size

    squared_sum = x**2 + y**2
    squared_sum = squared_sum[None,None,:,:]

    wave_nos = 2. * np.pi / wave_lengths
    wave_nos = wave_nos.reshape([1,-1,1,1])

    input_fields = np.tile(squared_sum, [len(depth_values), len(wave_lengths), 1, 1])
    input_fields = np.sqrt(input_fields + np.array(depth_values).reshape([-1, 1, 1, 1])**2)
    input_fields = np.exp(1.j * wave_nos * input_fields)

    if use_circular_aperture:
        input_fields = circular_aperture(input_fields, r_cutoff)

    psfs = []
    otfs = []
    # calculate PSF for each depth
    for depth_idx in range(len(depth_values)):
        # propagate through optical element
        input_field = input_fields[depth_idx:depth_idx+1,:,:,:]
        field = optical_element(input_field)

        # propagate field to sensor
        sensor_incident_field = propagate_exact(field, propagation_kernel)
        psf = sensor_incident_field
        # psf_edit = []
        # for wavelength in range(np.shape(psf)[1]):
        #     psf_image = np.squeeze(psf[0,wavelength,:,:])
            # if psf_resolution is not None:
            #     psf_image = np.array(Image.fromarray(psf_image).resize((psf_resolution[0], psf_resolution[1])))
        #     psf_image /= np.sum(np.abs(psf_image))
        #     psf_edit.append(np.expand_dims(np.expand_dims(psf_image, axis=0), axis=0))
            
        # psf = np.concatenate(psf_edit, axis=1)
        psfs.append(psf)
        
        # calculate OTF as well
        if get_otfs:
            otf = np.fft.fft2(psf)
            otfs.append(otf)

    return psfs, otfs

def PhaseShiftThinLens_rgb(focal_length,wave_lengths,wave_resolution,optical_feature_size,refractive_idcs):
    #Output is 1 x wave_resolution x wave_resolution x 3
    height_map_thinlens_0 = plano_convex_initializer(focal_length,
                                                                  wave_lengths[0],
                                                                  wave_resolution,
                                                                  optical_feature_size,
                                                                  refractive_idcs[0])
    PhaseThinLens_0 = phaseshifts_from_height_map(height_map_thinlens_0, wave_lengths[0],
                                                               refractive_idcs[0])
    height_map_thinlens_1 = plano_convex_initializer(focal_length,
                                                                  wave_lengths[1],
                                                                  wave_resolution,
                                                                  optical_feature_size,
                                                                  refractive_idcs[1])
    PhaseThinLens_1 = phaseshifts_from_height_map(height_map_thinlens_1, wave_lengths[1],
                                                               refractive_idcs[1])
    height_map_thinlens_2 = plano_convex_initializer(focal_length,
                                                                  wave_lengths[2],
                                                                  wave_resolution,
                                                                  optical_feature_size,
                                                                  refractive_idcs[2])
    PhaseThinLens_2 = phaseshifts_from_height_map(height_map_thinlens_2, wave_lengths[2],
                                                               refractive_idcs[2])
    PhaseThinLens = np.concatenate((PhaseThinLens_0, PhaseThinLens_1, PhaseThinLens_2), axis=1)
    PhaseThinLens = np.transpose(PhaseThinLens, [0, 2, 3, 1])
    return PhaseThinLens

def SaveHeightasTiff(height_map,filename,input_feature_size=4.29e-6,output_feature_size=1e-6,mask_size=5.6e-3,quantization_res=21.16e-9,Interp_Method='Nearest'):
    #height_map is given in meters and should be saved as a 32-bit integer where 0=0 nm and 1=21.16 nm (quantization_res)
    #Interpolate the height_map to a higher resolution, then resample at the output_feature_size
    #Nearest neighbor interpolation works by far the best
    assert (np.allclose(np.mod(mask_size, output_feature_size), 0.)), "mask_size must be a common multiple of the output_feature_size"
    height_map = height_map/1e-6#Perform interpolation in um
    x_input = np.arange(height_map.shape[0]) * input_feature_size
    y_input = np.arange(height_map.shape[1]) * input_feature_size
    if Interp_Method=='Nearest':
        f = interp.RegularGridInterpolator((x_input,y_input), height_map,method='nearest',bounds_error=False,fill_value=0.)
    elif Interp_Method=='Linear':
        f = interp.RegularGridInterpolator((x_input, y_input), height_map, method='linear', bounds_error=False, fill_value=0.)
    else:
        f = interp.RectBivariateSpline(x_input, y_input, height_map, bbox=[None, None, None, None], kx=3, ky=3, s=0)
    n_pixel_out = int(mask_size / output_feature_size)
    if Interp_Method=='Nearest' or Interp_Method=='Linear':
        grid_x_out, grid_y_out = np.mgrid[0:n_pixel_out, 0:n_pixel_out]*output_feature_size
        grid_x_out=grid_x_out.flatten()
        grid_y_out=grid_y_out.flatten()
        points_out = np.array((grid_x_out,grid_y_out)).T
        resampled_height_map = f(points_out)
        resampled_height_map=np.reshape(resampled_height_map,(n_pixel_out,n_pixel_out))
    else:
        x_output = np.arange(n_pixel_out) * output_feature_size
        y_output = np.arange(n_pixel_out) * output_feature_size
        resampled_height_map = f(x_output,y_output)
    resampled_height_map = np.clip(resampled_height_map,height_map.min(),height_map.max())

    # Quantize the height map to the nearest quantization_res. Save as a fp value in um and as a integer value, where 0 = 0 and 1 = quantization_res
    quantized_resampled_height_map_fp = (np.floor((resampled_height_map)/(quantization_res/1e-6))*(quantization_res/1e-6)).astype(np.float32)
    quantized_resampled_height_map_int = (np.floor((resampled_height_map) / (quantization_res / 1e-6))).astype(np.int32) # In um, quantized to nearest 21.16nm

    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # imgplot = plt.imshow((height_map))
    # plt.colorbar(imgplot)
    # plt.title('Height Map After Interpolation')
    # plt.subplot(122)
    # imgplot = plt.imshow((resampled_height_map))
    # plt.colorbar(imgplot)
    # plt.title('Height Map After Interpolation')
    # plt.show()
    #
    # import matplotlib.pyplot as plt
    # plt.subplot(121)
    # height_map_slice = height_map[1000,:]
    # imgplot = plt.hist(height_map_slice)
    # plt.title('Height Map Slice After Interpolation')
    # plt.subplot(122)
    # resampled_height_map_slice =  resampled_height_map[2500,:]
    # imgplot = plt.hist(resampled_height_map_slice)
    # plt.title('Height Map Slice After Interpolation')
    # plt.show()

    filename_fp=filename + "_fp32_wrt_um.tiff"
    imsave(filename_fp, quantized_resampled_height_map_fp)
    filename_int=filename + "_integer.tiff"
    imsave(filename_int, quantized_resampled_height_map_int)
    return [resampled_height_map,quantized_resampled_height_map_fp,quantized_resampled_height_map_int]
