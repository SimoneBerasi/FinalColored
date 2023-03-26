from DataLoader import *
import math
import tensorflow as tf
import cv2
from skimage.color import rgb2xyz, xyz2lab
from SpatialFilters import *
from SpatialFilters import get_spatial_kernels

#TODO check for color change when they non reach the area threshold

from Utils import visualize_results
from matchers import *

num_samples = 56
LAB_const = 127

referenceX = 95.047
referenceY = 100
referenceZ = 108.883
patch_size = 128

# Size of the neighborhood in the color histogram
D_size = 3
domined_color_treshold = 2  #ratio of a color being domined by another
dominant_color_treshold = 0.03 #area percentage needed for a color to be considered dominant
total_dominant_colors = 30      #Suggested value between 10-100

# Li = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]

Li = [0, 10, 20, 30, 40, 50, 60, 70, 80, 900, 100]


# Li = [0, 10, 40, 65, 85, 94, 100]

# expect an RGB image as input
def prepare_image_colorssim(image):

    l, rg, by = get_spatial_kernels()


    image = rgb2xyz(image)

    image = xyz2QQQ(image)

    q1 = cv2.filter2D(src=image[:,:,0], ddepth=-1, kernel=l)
    q2 = cv2.filter2D(src=image[:,:,1], ddepth=-1, kernel=rg)
    q3 = cv2.filter2D(src=image[:,:,2], ddepth=-1, kernel=by)

    image = np.stack([q1, q2, q3], axis=-1)

    image = QQQ2xyz(image)

    image = xyz2lab(image)

    return image


# Expect batch of rgb images, [N, H, W, C]  where N is the number of images, C is the channel number (3)
def prepare_dataset_colorssim(batch):
    processed = []
    for image in batch:
        processed.append(prepare_image_colorssim(image))

    return np.array(processed)


def xyz2QQQ(XYZimage):
    image = XYZimage
    Q1 = 0.279* image[:,:,0] + 0.72* image[:,:,1] - 0.107* image[:,:,2]
    Q2 = -0.449* image[:,:,0] + 0.29* image[:,:,1] - 0.077* image[:,:,2]
    Q3 = 0.086* image[:,:,0] - 0.59* image[:,:,1] + 0.501* image[:,:,2]

    return np.stack([Q1, Q2, Q3], axis=-1)

def QQQ2xyz(QQQimage):
    image = QQQimage
    x = 0.6204* image[:,:,0] - 1.8704* image[:,:,1] - 0.1553*image[:,:,2]
    y = 1.3661* image[:,:,0] + 0.9316* image[:,:,1] + 0.4339* image[:,:,2]
    z = 1.5013* image[:,:,0] + 1.4176* image[:,:,1] + 2.5331* image[:,:,2]

    return np.stack([x, y, z], axis=-1)




@tf.function
def round_keep_sum(value):
    total = tf.reduce_sum(value, 0)
    new_value = tf.math.round(value)

    if len(new_value.shape) == 2:
        new_value = tf.squeeze(new_value, axis=-1)

    if len(value.shape) == 2:
        value = tf.squeeze(value)
    new_total = tf.reduce_sum(new_value, axis=-1)

    #print(total)
    #print(new_total)



    while(new_total != total):

        if new_total > total:
            new_value = tf.tensor_scatter_nd_sub(new_value, tf.reshape(tf.argmax((new_value-value)**2), (1,1)), [1])

        else:
            new_value = tf.tensor_scatter_nd_add(new_value, tf.reshape(tf.argmax((new_value-value)**2), (1,1)), [1])

        new_total = tf.reduce_sum(new_value)

    print("done rounding")
    return new_value

#used for debugging
def show_color(color):
    image = []
    for i in range(0, 1024):
        row = []
        for j in range(0,1024):
            row.append(color)
        image.append(row)

    image = np.array(image)
    visualize_results(image, image, "color")


#For denugging and presentation:
def quantize_with_only_dominant_color(image, dominant_color_indexes, codebook):
    pass

# Convert color from LAB to XYZ
def lab2xyz(l, a, b):
    var_Y = (l + 16) / 116
    var_X = a / 500 + var_Y
    var_Z = var_Y - b / 200

    if var_Y ** 3 > 0.008856:
        var_Y = var_Y ** 3
    else:
        var_Y = (var_Y - 16 / 116) / 7.787

    if var_X ** 3 > 0.008856:
        var_X = var_X ** 3
    else:
        var_X = (var_X - 16 / 116) / 7.787

    if var_Z ** 3 > 0.008856:
        var_Z = var_Z ** 3
    else:
        var_Z = (var_Z - 16 / 116) / 7.787

    x_val = var_X * referenceX
    y_val = var_Y * referenceY
    z_val = var_Z * referenceZ

    return x_val, y_val, z_val


# Convert color from XYZ to RGB
def xyz2rgbmy(x, y, z):
    var_X = x / 100
    var_Y = y / 100
    var_Z = z / 100

    var_R = var_X * 3.2406 + var_Y * -1.5372 + var_Z * -0.4986
    var_G = var_X * -0.9689 + var_Y * 1.8758 + var_Z * 0.0415
    var_B = var_X * 0.0557 + var_Y * -0.2040 + var_Z * 1.0570

    if var_R > 0.0031308:
        var_R = 1.055 * (var_R ** (1 / 2.4)) - 0.055
    else:
        var_R = 12.92 * var_R
    if var_G > 0.0031308:
        var_G = 1.055 * (var_G ** (1 / 2.4)) - 0.055
    else:
        var_G = 12.92 * var_G
    if var_B > 0.0031308:
        var_B = 1.055 * (var_B ** (1 / 2.4)) - 0.055
    else:
        var_B = 12.92 * var_B

    sR = var_R * 255
    sG = var_G * 255
    sB = var_B * 255

    return sR, sG, sB


def lab2xyz_array(l_array, a_array, b_array):
    result = []
    array_len = l_array.shape[0]
    for i in range(array_len):
        result.append(lab2xyz(l_array[i], a_array[i], b_array[i]))

    return result


def xyz2rgb_array(x_array, y_array, z_array):
    result = []
    array_len = x_array.shape[0]
    for i in range(array_len):
        result.append(xyz2rgbmy(x_array[i], y_array[i], z_array[i]))

    return result


def xyz2rgb_matrix(x_matrix, y_matrix, z_matrix):
    x = x_matrix.shape[0]

    image = []
    for i in range(x):
        image.append(xyz2rgb_array(x_matrix[i], y_matrix[i], z_matrix[i]))
    return image


def lab2xyz_matrix(l_matrix, a_matrix, b_matrix):
    x = l_matrix.shape[0]

    image = []
    for i in range(x):
        image.append(lab2xyz_array(l_matrix[i], a_matrix[i], b_matrix[i]))
    return image


# Sample colors from lab color space
def fibonacci_lattice(samples=num_samples):
    points = []
    phi = math.pi * (-1. + math.sqrt(5.)) / 2  # golden angle in radians

    for i in range(-samples, samples):
        # y = i / -samples
        # y = 1 - (i / float(2*samples - 1))*2   # y goes from 1 to -1
        # radius = math.sqrt(1 - y * y)  # radius at y

        theta = 2 * phi * i + 0.05  # golden angle increment

        if (i >= 0):
            x = math.cos(theta) * math.sqrt(i) * LAB_const / math.sqrt(samples)
            z = math.sin(theta) * math.sqrt(i) * LAB_const / math.sqrt(samples)
        else:
            z = math.cos(theta) * math.sqrt(-i) * LAB_const / math.sqrt(samples)
            x = math.sin(theta) * math.sqrt(-i) * (-1) * LAB_const / math.sqrt(samples)
            # x = math.cos(theta) * math.sqrt(-i) * LAB_const/math.sqrt(samples)
            # z = math.sin(theta) * math.sqrt(-i) * LAB_const/math.sqrt(samples)

        points.append((x, z))

    return points


# For debugging
def show_fibonacci_lattice(samples=num_samples):
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111)
    points = fibonacci_lattice(samples)

    for i in range(len(points)):
        ax.scatter(points[i][0], points[i][1])

    plt.show()


def build_codebook(num_samples=num_samples):
    points = fibonacci_lattice(num_samples)

    colors = []
    for lum in Li:
        for point in points:
            colors.append((lum, point[0], point[1]))

    colors = np.array(colors)

    xyz_colors = lab2xyz_array(colors[:, 0], colors[:, 1], colors[:, 2])
    xyz_colors = np.array(xyz_colors)
    rgb_colors = xyz2rgb_array(xyz_colors[:, 0], xyz_colors[:, 1], xyz_colors[:, 2])

    colors = []
    for item in rgb_colors:
        if not (item[0] > 255 or item[1] > 255 or item[2] > 255 or item[0] < 0 or item[1] < 0 or item[
            2] < 0):
            colors.append(item)

    colors.append((255,255,255))        #add white to codebook for debugging
    colors = np.array(colors)

    return colors


# Compute matrix with all color combination distances
def compute_all_color_distances(colors):
    distances = np.zeros((len(colors), len(colors)), dtype=float)
    max_distance = tf.sqrt((255.0 ** 2) * 3)
    for i in range(len(colors)):
        for j in range(len(colors)):
            distances[i][j] = np.sqrt((colors[i][0] - colors[j][0]) ** 2 + (colors[i][1] - colors[j][1]) ** 2 + (
                    colors[i][2] - colors[j][2]) ** 2) / max_distance

    return distances


# TODO all these function need to be rewritten for batch working



#TODO rewrite for tf.function
#@tf.function
def remove_specked_noise(quantized_image, codebook):
    original_image = quantized_image
    neighborhood_color_histogram = tf.zeros(shape=(len(codebook), len(codebook)), dtype=tf.int32)
    quantized_image = tf.reshape(quantized_image, (1, patch_size, patch_size, 3))
    # extract every D_size * D_size patches from image
    D_patches = tf.image.extract_patches(quantized_image, sizes=(1, D_size, D_size, 1), strides=(1, 1, 1, 1),
                                         padding='SAME', rates=[1, 1, 1, 1])
    # reshape them
    D_patches = tf.reshape(D_patches, (-1, D_size, D_size, 3))

    # D_patches contains all D_size*D_size patches of the quantized image
    # reshape to remove one useless dimension. The central color is the one of the pixel center of the patch, used later
    D_patches = tf.reshape(D_patches, (patch_size * patch_size, D_size * D_size, 3))

    print("Dpatches.shape")
    print(D_patches.shape)

    #tf.gather(D_patches, )

    for ind in range(0,len(codebook)):
        #ind=2
        color = codebook[ind]

        # Find all patches with the selected color in the center, need to loop over all colors.
        # TODO be rewritten in tf, dont know how without cartesian product (too heavy on memory)
        indexes = (tf.where(D_patches[:, tf.cast((D_size*D_size - 1)/2, tf.int32), :] == color)[:, 0])


        if(len(indexes)==0):
            continue

        # Gather all the patches find above and flatten them
        colors = tf.gather(D_patches, indexes)
        colors = tf.reshape(colors, (colors.shape[0] * D_size * D_size, 3))

        # Reduce one dimension to use simple functions. Semantically not correct but works in practice
        colors = tf.reduce_sum(colors, axis=-1)
        summed_codebook = tf.reduce_sum(codebook, -1)
        colors = tf.cast(colors, tf.float32)
        summed_codebook = tf.cast(summed_codebook, tf.float32)

        # Concat to have all possible colors. Semantically not correct but works in practice
        colors = tf.concat([summed_codebook, colors], axis=-1)

        # Remember that colors are ordered in the same way as in codebook! Huge advantage here
        y, _, count = tf.unique_with_counts(colors)

        count = count / len(colors)



        dominant_color_index = tf.where(count == tf.reduce_max(count))[0]
        #print(count)
        #print(dominant_color_index)
        #print(tf.gather(count, dominant_color_index))
        dominant_actual_color_ratio = tf.gather(count, dominant_color_index) / tf.gather(count, ind)
        dominant_color = tf.gather(codebook, dominant_color_index)


        quantized_image = tf.reshape(quantized_image, (patch_size, patch_size, 3))

        #print(tf.where(quantized_image == codebook[2])[:,-1])



        to_change_indexes = (tf.gather(tf.where(quantized_image == codebook[ind]), tf.where(tf.where(quantized_image == codebook[ind])[:,-1]==0)))[:,:,0:2]

        to_change_indexes = tf.reshape(to_change_indexes, (to_change_indexes.shape[0], 2))



        if dominant_actual_color_ratio > domined_color_treshold:
            #print(quantized_image[454,888,:])
            quantized_image = tf.where(quantized_image == codebook[ind], dominant_color, quantized_image)

            #print(quantized_image[454, 888, :])

            #print(tf.where(tf.equal(quantized_image, tf.gather(codebook, dominant_color_index))))

        #print(new_color_index)
    visualize_results(original_image / 255, quantized_image / 255, "quantized vs without speckle")

    return quantized_image


#TODO adding the % of the domined colors to the closer dominant
#@tf.function
def extract_dominant_colors(quantized_smooth_image, codebook):

    quantized_smooth_image = tf.reshape(quantized_smooth_image, (patch_size * patch_size, 3))

    b, a = quantized_smooth_image[None, :], codebook[:, None]

    a = tf.cast(a, dtype=tf.float32)
    b = tf.cast(b, dtype=tf.float32)
    min_indexes = ((b - a) ** 2)

    min_indexes = tf.reduce_sum(min_indexes, axis=2)
    min_indexes = tf.sqrt(min_indexes)
    min_index = tf.argmin(min_indexes, axis=0)

    colors, _, counts = tf.unique_with_counts(min_index)



    colors_percentage = counts / (patch_size*patch_size)


    domined_colors_indexes = tf.gather(colors, tf.where(colors_percentage < dominant_color_treshold))
    dominant_colors_indexes = tf.gather(colors, tf.where(colors_percentage >= dominant_color_treshold))

    domined_colors = tf.gather(codebook, domined_colors_indexes)
    all_colors = tf.gather(codebook, colors)
    domined_colors = tf.squeeze(domined_colors)
    dominant_colors = tf.gather(codebook, dominant_colors_indexes)
    dominant_colors = tf.squeeze(dominant_colors, 1)
    all_colors = tf.squeeze(all_colors)
    #print(all_colors.shape)
    #print(domined_colors.shape)
    #print(domined_colors)
    #print(dominant_colors)


    b, a = all_colors[None, :], dominant_colors[:, None]



    a = tf.cast(a, dtype=tf.float32)
    b = tf.cast(b, dtype=tf.float32)
    min_indexes = ((b - a) ** 2)



    min_indexes = tf.reduce_sum(min_indexes, axis=2)
    min_indexes = tf.sqrt(min_indexes)
    min_index = tf.argmin(min_indexes, axis=0)

    colors_percentage = colors_percentage*100

    #print(min_index)

    #print(domined_colors)
    #print(colors)

    dominant_colors_percentages = tf.where(colors_percentage > dominant_color_treshold*100, colors_percentage, 0)
    dominant_colors_percentages = tf.gather(colors_percentage, tf.where(colors_percentage> dominant_color_treshold*100))


    dominant_colors_percentages = tf.squeeze(dominant_colors_percentages)
    dominant_colors_percentages = tf.math.round(dominant_colors_percentages / (dominant_color_treshold*100))
    dominant_colors_indexes = tf.squeeze(dominant_colors_indexes)

    #colors_percentage = saferound(colors_percentage, places=0)
    #print(tf.reduce_sum(colors_percentage))
    #print(colors)
    #print(min_index)
    #print(tf.gather(colors, min_index))
    colors_percentage = tf.expand_dims(colors_percentage, -1)

    colors_components = round_keep_sum((colors_percentage*total_dominant_colors)/100)



    index_with_percentage = tf.concat([tf.expand_dims(tf.cast(tf.gather(colors, min_index), dtype=tf.float64), -1), tf.expand_dims(colors_components, -1)], -1)
    colors_components = tf.cast(colors_components, tf.int64)
    component_dataset = tf.stack((tf.gather(colors, min_index), colors_components), axis = -1)



    #show_color(tf.gather(codebook, 23)/255)
    #show_color(tf.gather(codebook, 76) / 255)

    return component_dataset, dominant_colors_indexes


    #colors = tf.where(colors_percentage < dominant_color_treshold*100, )

    return dominant_colors_indexes, dominant_colors_percentages
    #print(domined_colors_percentage)
    #tf.gather(domined_colors_percentage)
    #tf.where(dominant_colors_percentages != 0, dominant_colors_percentages + 1, 0)
    #print(min_index)


@tf.function
def get_quantized_image(image, codebook):
    original_image = image

    image = tf.reshape(image, (patch_size * patch_size, 3))

    b, a = image[None, :], codebook[:, None]

    a = tf.cast(a, dtype=tf.float32)
    b = tf.cast(b, dtype=tf.float32)

    min_indexes = ((b - a) ** 2)


    min_indexes = tf.reduce_sum(min_indexes, axis=2)
    min_indexes = tf.sqrt(min_indexes)
    min_index = tf.argmin(min_indexes, axis=0)


    #y, idx, counts = tf.unique_with_counts(min_index)

    # Quantized image, for debugging
    quantized_image = tf.gather(codebook, min_index)
    quantized_image = tf.reshape(quantized_image, (patch_size, patch_size, 3))
    # visualize_results(original_image/255, quantized_image/255, "quantized")

    return quantized_image


def OCCD_computation(img0_color_indexes, img0_color_quantity, img1_color_indexes, img1_color_quantity, codebook):
    pass


@tf.function
def calculate_weight_matrix(color_components0, color_components1, color_difference_matrix):
    '''
    vertex0_prop_list = tf.repeat(color_components0[:, 0],
                                  color_components0[:,1])
    vertex1_prop_list = tf.repeat(color_components1[:, 0],
                                  color_components1[:, 1])
    '''
    print(color_components0)

    color_components0 = tf.convert_to_tensor(color_components0)
    color_components0 = tf.reshape(color_components0, (color_components0.shape[0], color_components0.shape[2]))

    color_components1 = tf.convert_to_tensor(color_components1)
    color_components1 = tf.reshape(color_components1, (color_components1.shape[0], color_components1.shape[2]))

    vertex0_prop_list = tf.repeat(color_components0[ :, 0],
                                  color_components0[ :, 1])
    vertex1_prop_list = tf.repeat(color_components1[:,  0],
                                  color_components1[:,  1])

    mesh = tf.convert_to_tensor(tf.meshgrid(vertex0_prop_list, vertex1_prop_list))
    mesh = tf.transpose(mesh)
    #mesh = tf.reshape(mesh, (-1, 2))

    color_difference_list = tf.gather_nd(color_difference_matrix, mesh)

    return color_difference_list

def simple_color_metric(y_true, y_pred):

    r_true_mean = tf.math.reduce_mean(y_true[:,:,:,0], axis = 1)
    r_true_mean = tf.math.reduce_mean(r_true_mean, axis = 1)

    g_true_mean = tf.math.reduce_mean(y_true[:,:,:,1], axis = 1)
    g_true_mean = tf.math.reduce_mean(g_true_mean, axis = 1)

    b_true_mean = tf.math.reduce_mean(y_true[:,:,:,2], axis = 1)
    b_true_mean = tf.math.reduce_mean(b_true_mean, axis = 1)


    r_pred_mean = tf.math.reduce_mean(y_pred[:, :, :, 0], axis=1)
    r_pred_mean = tf.math.reduce_mean(r_pred_mean, axis=1)

    g_pred_mean = tf.math.reduce_mean(y_pred[:, :, :, 1], axis=1)
    g_pred_mean = tf.math.reduce_mean(g_pred_mean, axis=1)

    b_pred_mean = tf.math.reduce_mean(y_pred[:, :, :, 2], axis=1)
    b_pred_mean = tf.math.reduce_mean(b_pred_mean, axis=1)


    r = tf.sqrt((r_pred_mean - r_true_mean)**2)
    g = tf.sqrt((g_pred_mean - g_true_mean)**2)
    b = tf.sqrt((b_pred_mean - b_true_mean)**2)




    # = (tf.reduce_sum(y_true[:,:,:,0]) - tf.reduce_sum(y_pred[:,:,:,0]))**2
    #g = (tf.reduce_sum(y_true[:,:,:,1]) - tf.reduce_sum(y_pred[:,:,:,1]))**2
    #b = (tf.reduce_sum(y_true[:,:,:,2]) - tf.reduce_sum(y_pred[:,:,:,2]))**2

    #print((r+g+b)/3)

    return (r+g+b)/3

def OCCD(y_true, y_pred, codebook, color_distance_matrix):



    true_quantized = get_quantized_image(y_true, codebook)
    pred_quantized = get_quantized_image(y_pred, codebook)
    true_components = extract_dominant_colors(true_quantized, codebook)
    pred_components = extract_dominant_colors(pred_quantized, codebook)
    weight_matrix = calculate_weight_matrix(true_components, pred_components, color_distance_matrix)
    weight_matrix = tf.reshape(weight_matrix, (1, total_dominant_colors, total_dominant_colors))

    matching = hungarian_matching(weight_matrix)
    mask = tf.convert_to_tensor(matching[1])
    mask = tf.cast(mask, dtype=tf.float64)

    print(":here")

    OCCD = tf.reduce_sum(mask * weight_matrix)
    return OCCD




