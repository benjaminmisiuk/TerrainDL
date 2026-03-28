# import necessary libraries
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas
import sklearn
import copy
from rasterio.plot import show
from rasterio.fill import fillnodata
from numpy.random import randint

# PREPARING FOR MODELLING
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TensorFlow warnings
import tensorflow as tf

tf.get_logger().setLevel('ERROR')  # suppress Keras messages
from tensorflow import keras
from keras import layers
from keras import utils
from keras import metrics
from keras.callbacks import EarlyStopping
from sklearn import model_selection
import glob
import gc

# check available GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# define convolutional blocks used in the CNN
def conv_block(input_layer, filters=32, depth=1, kernel=3, activation=keras.layers.PReLU):
    z = input_layer
    for _ in range(depth):
        # if the image dimensions are greater than 3x3, use pooling, else, use a convolutional layer
        if z.shape[1] > kernel and z.shape[2] > kernel:
            z = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same')(z)
            z = activation()(z)
            z = layers.MaxPooling2D((3, 3))(z)
        else:
            z = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same')(z)
            z = activation()(z)
    return z

# define variables for the model run(s)

# these variables are not iterated
trial = 'ponui_11000_81' # the name of the run
gis_dir = 'D:/GIS/Ponui/' # GIS directory containing a folder with DTM raster patches
dtm_dir = 'D:/GitHub/terrain_learning/python/out/'
out_dir = 'D:/GIS/Ponui/terrain_NN_learning/individual_tests/' # directory to write the terrain prediction
results_dir = 'D:/GIS/Ponui/results/individual_tests/' # directory to write numerical results
temp_dir = 'D:/python/temp/' # indicate a temporary directory
w = 81 # the size of input images
act = keras.layers.PReLU  # activation function
act_name = 'PReLU'  # activation function name

# these are hyperparameters that are iterated but not loopable
# i.e. you have to change them then run the code again
scale = 3 # scale of the terrain calc
n = 10000 # sample size
norm = 'sdmean' # how should normalization be performed for image patches?
#norm = ['normzero', 'normmean', 'global', 'sdmean']

# these are hyperparameters that are loopable
sizes = [81] # size of the input
depth = [1, 2, 3]  # depth of the CNN
filters = [16, 32, 64, 128]  # number of filters
response = ['qslope', 'qeastness', 'meanc', 'tpi', 'adjSD'] # target variable

# run analysis with the selected hyperparameters
for w2 in sizes:
    df = pandas.read_csv(gis_dir + "points/" + trial + "_" + str(scale) + ".csv")
    df.columns
    df.isnull().sum()

    # keep the first n rows
    df = df.head(int(n + n * 0.1))

    # Open and display sample patches
    # r = rasterio.open(gis_dir + "patches/w_" + str(scale) + "/sample_42.tif")
    # plt.imshow(r.read(masked=True)[0,:,:])
    # r = fillnodata(r.read(masked=True))
    # plt.imshow(r[0,:,:])
    # plt.show()

    # read in each raster patch, fill nodata, crop, and stack them
    center = w // 2
    dist = w2 // 2
    l_stack = list()
    for i in df['sample']:
        print(i, end='\r')
        # open and fill nodata in each raster
        r = rasterio.open(gis_dir + 'patches/w_' + str(scale) + '/sample_' + str(i) + ".tif")
        r = fillnodata(r.read(masked=True))
        # crop to size w2 x w2
        r = r[0, center - dist:center + dist + 1, center - dist:center + dist + 1]
        l_stack.append(r)

    # clean up variables
    del (r, center, dist, i)

    # concatenate the arrays along the first axis
    l_stack = np.stack(l_stack)
    l_stack.shape

    # Randomly select an index
    #i = randint(0, l_stack.shape[0])
    #d = 0
    #plt.imshow(l_stack[i, :, :])

    # get the minimum value of a specific dimension in the stacked raster arrays
    np.nanmin(l_stack)

    # calculate global minimum and maximum values of the stacked raster arrays
    mins_glob = np.nanmin(l_stack)
    maxs_glob = np.nanmax(l_stack)

    # normalize the stacked raster arrays
    #l_stack_glob = (l_stack - mins_glob) / (maxs_glob - mins_glob)
    l_stack_glob = copy.deepcopy(l_stack)
    print(np.nanmin(l_stack_glob), np.nanmax(l_stack_glob))
    
    # copy DataFrame for preprocessing
    df_p = df.copy()
    df_p.describe()

    # define a dictionary containing minimum and maximum values for selected columns
    d = {
        'qslope': [df_p['qslope'].min(), df_p['qslope'].max()],
        'qeastness': [df_p['qeastness'].min(), df_p['qeastness'].max()],
        'meanc': [df_p['meanc'].min(), df_p['meanc'].max()],
        'tpi': [df_p['tpi'].min(), df_p['tpi'].max()],
        'adjSD': [df_p['adjSD'].min(), df_p['adjSD'].max()]
    }

    # create a DataFrame from the dictionary
    mnmx = pandas.DataFrame(d)
    del d

    # normalize selected columns in the DataFrame
    df_p['qslope'] = (df_p['qslope'] - mnmx['qslope'][0]) / (mnmx['qslope'][1] - mnmx['qslope'][0])
    df_p['qeastness'] = (df_p['qeastness'] - mnmx['qeastness'][0]) / (mnmx['qeastness'][1] - mnmx['qeastness'][0])
    df_p['meanc'] = (df_p['meanc'] - mnmx['meanc'][0]) / (mnmx['meanc'][1] - mnmx['meanc'][0])
    df_p['tpi'] = (df_p['tpi'] - mnmx['tpi'][0]) / (mnmx['tpi'][1] - mnmx['tpi'][0])
    df_p['adjSD'] = (df_p['adjSD'] - mnmx['adjSD'][0]) / (mnmx['adjSD'][1] - mnmx['adjSD'][0])

    # check for missing values in the DataFrame after preprocessing
    #df_p.isnull().sum()

    # display descriptive statistics of the DataFrame after preprocessing
    #df_p.describe()

    # assign global training data to x_train_glob as deep copy
    x_train_glob = copy.deepcopy(l_stack_glob)

    # normalize each image in x_train_glob using either norm_zero or norm_mean
    if norm == 'normzero':
        for i in range(x_train_glob.shape[0]):
            x_train_glob[i] = (x_train_glob[i] - np.min(x_train_glob[i])) / (np.max(x_train_glob[i]) - np.min(x_train_glob[i]))
    elif norm == 'normmean':
        for i in range(x_train_glob.shape[0]):
            x_train_glob[i] = (x_train_glob[i] - np.mean(x_train_glob[i]))
    elif norm == 'global':
        x_train_glob = (2 * ((x_train_glob - mins_glob) / (maxs_glob - mins_glob))) - 1
    elif norm == 'sdmean':
        # first mean center
        x_train_glob -= np.mean(x_train_glob, axis=(1, 2), keepdims=True)
        # get average range of centered data
        mean_range = np.mean(np.max(x_train_glob, axis=(1, 2)) - np.min(x_train_glob, axis=(1, 2)))
        # rescale to range of 2 (an average range of -1 to 1)
        x_train_glob *= (2 / mean_range)

        #OLD
        #mean_range = np.mean(np.max(x_train_glob, axis=(1,2)) - np.min(x_train_glob, axis=(1,2)))
        #mean_min = (mean_range/2)*-1
        #mean_max = (mean_range/2)*1
        #rescale so the average range is 2 for each image
        #x_train_glob = (2 * (x_train_glob - mean_min) / (mean_max - mean_min)) - 1
        #mean center
        #for i in range(x_train_glob.shape[0]):
        #   x_train_glob[i] = (x_train_glob[i] - np.mean(x_train_glob[i]))

    # START THE TRAINING LOOP HERE

    for filt in filters:
        for d in depth:
            for y_h in response:
                # check if this set of parameters has been run and saved
                if not os.path.exists(results_dir + "scale" + str(scale) + "_" + norm + "/" + y_h + "_" + str(w) + "_" + str(w2) + "_" + str(n) + "_" + str(d) + "_" + str(
                        filt) + "_" + act_name + ".csv"):

                    print(
                        "Sample size: " + str(n) + ", Depth: " + str(d) + ", Filters: " + str(filt) +
                        ", Response: " + str(y_h) + ", Window: " + str(w2) + ", Norm: " + str(norm)
                    )
                    # convert DataFrame columns to NumPy arrays for response variable
                    y_np = df_p[[y_h]].to_numpy()

                    # define keras CNN model
                    input = keras.Input(shape=(x_train_glob.shape[1], x_train_glob.shape[2], 1))
                    c1 = conv_block(input, depth=d, filters=filt, kernel=3, activation=act)
                    out = layers.Flatten(name='out')(c1)
                    out = layers.Dense(1)(out)
                    out = keras.activations.sigmoid(out)

                    model = keras.Model(inputs=[input], outputs=[out])
                    model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
                    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25, min_delta=0.0001,
                                       restore_best_weights=True)

                    # train the model
                    history = model.fit(
                        x=[x_train_glob],
                        y=[y_np[:, 0]],
                        epochs=100,
                        batch_size=32,
                        validation_split=0.1,
                        verbose=0,
                        callbacks=[es]
                    )

                    #histogram of y_np[:, 0]
                    #plt.hist(y_np[:, 0], bins=10, color='skyblue', edgecolor='black')

                    #plt.figure()
                    #plt.plot(history.history['loss'], linewidth=1, color = 'blue', label='loss')
                    #plt.plot(history.history['val_loss'], linewidth=1, color = 'red', label='val_loss')

                    # check if there is a full DTM to use for prediction
                    #if not, read in DTM, fill no data, and normalize based on mins_glob and maxs_glob

                    #r = rasterio.open("D:/GIS/Ponui/layers/ponui_dtm_setnull_5m_ext.tif")
                    #r1 = r.read(1, masked=True)
                    #mask = r.read_masks(1)
                    #nodata = r.nodata
                    #r1 = fillnodata(r1, max_search_distance=10000)
                    #plt.imshow(r1)
                    #np.savez(dtm_dir + "dtm.npz", r1=r1)

                    dtm = np.load(dtm_dir + "dtm.npz")
                    dtm = dtm['r1']
                    #dtm = (dtm - mins_glob) / (maxs_glob - mins_glob)
                    dtm.min()
                    dtm.max()

                    # format patches for prediction
                    m = 50  # number of rows processed at a time
                    batches = int(np.floor((dtm.shape[0] - w2) / m))  # number of loops required for processing

                    # remove existing files in the specified output directory
                    for g in glob.glob(temp_dir + '*'):
                        os.remove(g)

                    # stop if temp directory is not empty
                    if len(os.listdir(temp_dir)) != 0:
                        print("Temp directory is not empty")
                        sys.exit()

                    # loop through the data in batches of size 'm'
                    for l in range(batches):
                        print(l, end='\r')
                        raw_list = list()
                        # extract patches for prediction
                        for i in range(m * l, m * (l + 1), 1):
                            for j in range(0, int(int(dtm.shape[1])) - w2, 1):
                                raw_list.append(
                                    dtm[i:i + w2, j:j + w2]
                                )

                        raw_stack = np.stack(raw_list, axis=-1)
                        blocks = np.transpose(raw_stack, [2, 0, 1])
                        blocks = np.expand_dims(blocks, axis=-1)
                        del (raw_stack)

                        # normalize each image in blocks
                        if norm == 'normzero':
                            for i in range(blocks.shape[0]):
                                blocks[i] = (blocks[i] - np.min(blocks[i])) / (np.max(blocks[i]) - np.min(blocks[i]))
                        elif norm == 'normmean':
                            for i in range(blocks.shape[0]):
                                blocks[i] = (blocks[i] - np.mean(blocks[i]))
                        elif norm == 'global':
                            blocks = (2 * ((blocks - mins_glob) / (maxs_glob - mins_glob))) - 1
                        elif norm == 'sdmean':
                            blocks -= np.mean(blocks, axis=(1, 2), keepdims=True)
                            # rescale to range of 2 (an average range of -1 to 1)
                            blocks *= (2 / mean_range)

                            #OLD
                            #blocks = (2 * (blocks - mean_min) / (mean_max - mean_min)) - 1
                            #for i in range(blocks.shape[0]):
                            #    blocks[i] = (blocks[i] - np.mean(blocks[i]))

                        # predict using the model
                        pred = model.predict([blocks], verbose=0)
                        del (blocks)

                        # save predictions to npz
                        np.savez(file=temp_dir + str(l) + '.npz', pred=pred)
                        del (pred)
                        gc.collect()

                    # process the remaining rows after the loop if there are any left
                    if batches * m + w2 < dtm.shape[0]:
                        l = batches
                        print(l, end='\r')
                        raw_list = list()
                        # extract patches for prediction
                        for i in range(m * l, int(np.floor(dtm.shape[0] - w2)), 1):
                            for j in range(0, int(int(dtm.shape[1])) - w2, 1):
                                raw_list.append(
                                    dtm[i:i + w2, j:j + w2]
                                )
                        raw_stack = np.stack(raw_list, axis=-1)
                        blocks = np.transpose(raw_stack, [2, 0, 1])
                        blocks = np.expand_dims(blocks, axis=-1)
                        del (raw_stack)

                        # normalize each image in blocks
                        if norm == 'normzero':
                            for i in range(blocks.shape[0]):
                                blocks[i] = (blocks[i] - np.min(blocks[i])) / (np.max(blocks[i]) - np.min(blocks[i]))
                        elif norm == 'normmean':
                            for i in range(blocks.shape[0]):
                                blocks[i] = (blocks[i] - np.mean(blocks[i]))
                        elif norm == 'global':
                            blocks = (2 * ((blocks - mins_glob) / (maxs_glob - mins_glob))) - 1
                        elif norm == 'sdmean':
                            blocks -= np.mean(blocks, axis=(1, 2), keepdims=True)
                            # rescale to range of 2 (an average range of -1 to 1)
                            blocks *= (2 / mean_range)

                            # OLD
                            # blocks = (2 * (blocks - mean_min) / (mean_max - mean_min)) - 1
                            # for i in range(blocks.shape[0]):
                            #    blocks[i] = (blocks[i] - np.mean(blocks[i]))


                        # predict using the model
                        pred = model.predict([blocks], verbose=0)
                        # save predictions to npz files
                        np.savez(file=temp_dir + str(l) + '.npz', pred=pred)
                        del (blocks)
                        del (pred)
                        gc.collect()

                    # read in predictions and write to raster layers
                    # open the raster file containing the original data
                    raw = rasterio.open("D:/GIS/Ponui/layers/ponui_dtm_setnull_5m_ext.tif")

                    mask = raw.read_masks(1)
                    mask[mask != 0] = 1
                    # change the mask value
                    mask = mask.astype(float)
                    mask[mask == 0] = np.nan

                    l = list()
                    # loop through each prediction file in the directory
                    for i in range(len(os.listdir(temp_dir))):
                        # load the prediction data
                        with np.load(os.path.join(temp_dir + str(i) + '.npz')) as pred:
                            l.append(pred['pred'])

                    # concatenate the predictions
                    l = np.concatenate(l)
                    # reshape the concatenated predictions to match the original raster shape
                    l = np.reshape(l, (dtm.shape[0] - w2, dtm.shape[1] - w2))

                    # rescale the predictions to the original data range
                    l = l * (mnmx[y_h][1] - mnmx[y_h][0]) + mnmx[y_h][0]

                    # pad the resulting raster to match the original raster dimensions
                    row_pad = np.ones(shape=(int(np.floor(w2 / 2)), l.shape[1]), dtype=l.dtype)
                    l = np.append(row_pad, l, axis=0)
                    l = np.append(l, np.ones(shape=(int(raw.shape[0] - l.shape[0]), l.shape[1]), dtype=l.dtype), axis=0)

                    col_pad = np.ones(shape=(l.shape[0], int(np.floor(w2 / 2))), dtype=l.dtype)
                    l = np.append(col_pad, l, axis=1)
                    l = np.append(l, np.ones(shape=(l.shape[0], int(raw.shape[1] - l.shape[1])), dtype=l.dtype), axis=1)

                    # apply the mask to the predictions
                    l = l * mask

                    # define affine transformation and reference CRS
                    aff = rasterio.transform.from_origin(raw.bounds[0], raw.bounds[3], raw.res[0], raw.res[1])
                    ref = raw.crs

                    # define the directory name for the output layer
                    dir_name = out_dir + y_h + str(scale) + "_" + str(w) + "_" + str(w2) + "_" + str(n) + "_" + str(
                        d) + "_" + str(filt) + "_" + act_name + "_" + norm + '.tif'

                    # create a new raster file for the output layer
                    cnn_ft = rasterio.open(
                        dir_name,
                        'w',
                        driver='GTiff',
                        height=l.shape[0],
                        width=l.shape[1],
                        count=1,
                        dtype=raw.read(1).dtype,
                        crs=ref,
                        transform=aff,
                        nodata=np.nan
                    )
                    # write the predicted values to the raster file
                    cnn_ft.write(l, 1)
                    cnn_ft.close()

                    # STATISTICAL ANALYSIS

                    # load original layer
                    r = rasterio.open("D:/GIS/Ponui/layers/" + y_h + "_" + str(scale) + ".tif")
                    r1 = r.read(1)

                    mask = r.read_masks(1)

                    # stop if dimensions don't match l
                    if r1.shape != l.shape:
                        print("Dimensions don't match")
                        sys.exit()

                    r1 = r1.astype(float)
                    r1[mask == 0] = np.nan
                    r1 = r1.flatten()
                    l = l.flatten()

                    if len(l) != len(r1):
                        print("Lengths don't match")
                        sys.exit()

                    # bind l and r1 and remove rows with nans
                    all = np.column_stack((r1, l))
                    all = all[~np.isnan(all).any(axis=1)]

                    # write the results
                    np.savetxt(
                        results_dir + "scale" + str(scale) + "_" + norm + "/" + y_h + "_" + str(w) + "_" + str(w2) + "_" + str(n) + "_" + str(d) + "_" + str(
                            filt) + "_" + act_name + ".csv",
                        all,
                        delimiter=","
                    )

                    # cleanup
                    del (r, r1, mask, l, all, raw, dtm, model, history, aff, batches, pred, input, c1, out)
                    gc.collect()
print('DONE')








#compute the RMSE and R-squared, ignoring nan values
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

r2_score(all[:,0], all[:,1])
mean_squared_error(all[:,0], all[:,1], squared=False)

#plot the data
plt.scatter(all[:,0], all[:,1], s=1, alpha=0.5)
plt.show()

print("Sample size: " + str(n) + ", Depth: " + str(d) + ", Filters: " + str(filt) + ", Response: " + str(y_h))
# Convert DataFrame columns to NumPy arrays
y_np = df_p[[y_h]].to_numpy()

def conv_block(input_layer, filters=32, depth=1, kernel=3, activation=keras.layers.PReLU):
    z = input_layer
    for _ in range(depth):
        # if the image dimensions are greater than 3x3, use pooling, else, use a convolutional layer
        if z.shape[1] > kernel and z.shape[2] > kernel:
            z = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same')(z)
            z = activation()(z)
            z = layers.MaxPooling2D((3, 3))(z)
        else:
            z = layers.Conv2D(filters=filters, kernel_size=kernel, padding='same')(z)
            z = activation()(z)
    return z

c1 = conv_block(input, depth=d, filters=filt, kernel=3, activation=act)

input = keras.Input(shape=(x_train_glob.shape[1], x_train_glob.shape[2], 1))
z = layers.Conv2D(filters=filt, kernel_size=3, padding='same')(input)
z = act()(z)
z = layers.Conv2D(filters=filt, kernel_size=3, padding='same')(z)
z = act()(z)
out = layers.Flatten(name='out')(z)
out = layers.Dense(1)(out)
out = keras.activations.sigmoid(out)

model = keras.Model(inputs=[input], outputs=[out])
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
#keras.utils.plot_model(model, to_file='D:/python/scratch/model.png', show_shapes=True, rankdir='TB', show_layer_names=False, show_dtype=False)

history = model.fit(
    x=[x_train_glob],
    y=[y_np[:,0]],
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

plt.figure()
plt.plot(history.history['loss'], linewidth=1, color = 'blue', label='loss')
plt.plot(history.history['val_loss'], linewidth=1, color = 'red', label='val_loss')

#predict the training samples
pred = model.predict([x_train_glob], verbose=1)

#calculate r2
sklearn.metrics.r2_score(y_np[:,0], pred)

# Read in the full DEM, fill no date, and normalize based on mins_glob and maxs_glob
#r = rasterio.open("D:/GIS/Geomorphometry/layers/ponui_dtm_setnull.tif")
#r1 = r.read(1, masked=True)
#mask = r.read_masks(1)
#plt.imshow(r1)
#nodata = r.nodata

#r1 = fillnodata(r1, max_search_distance=10000)
#plt.imshow(r1)

#np.savez("D:/python/ponui/filled/dtm.npz", r1=r1)

dtm = np.load("D:/python/ponui/filled/dtm.npz")
dtm = dtm['r1']
dtm = (dtm - mins_glob) / (maxs_glob - mins_glob)
dtm.min()
dtm.max()

# format patches for prediction
# Set parameters for processing
m = 30  # Number of rows processed at a time
batches = int(np.floor((dtm.shape[0] - w2) / m))  # Number of loops required for processing

# Remove existing files in the specified output directory
for g in glob.glob(temp_dir + '*'):
    os.remove(g)

#stop if temp directory is not empty
len(os.listdir(temp_dir)) != 0

# Loop through the data in batches of size 'm'
for l in range(batches):
    print(l, end='\r')
    raw_list = list()
    # Extract patches for prediction
    for i in range(m * l, m * (l + 1), 1):
        for j in range(0, int(int(dtm.shape[1])) - w2, 1):
            raw_list.append(
                dtm[i:i+w2, j:j+w2]
            )

    raw_stack = np.stack(raw_list, axis=-1)
    blocks = np.transpose(raw_stack, [2, 0, 1])
    blocks = np.expand_dims(blocks, axis=-1)
    del (raw_stack)

    #normalize each image in blocks
    for i in range(blocks.shape[0]):
        blocks[i] = (blocks[i] - np.min(blocks[i])) / (np.max(blocks[i]) - np.min(blocks[i]))

    # Predict using the model
    pred = model.predict([blocks], verbose=0)
    del (blocks)

    # Save predictions to npz
    np.savez(file=temp_dir + str(l) + '.npz', pred=pred)
    del (pred)
    gc.collect()

# Process the remaining rows after the loop if there are any left
if batches * m + w2 < dtm.shape[0]:
    l = batches
    print(l, end='\r')
    raw_list = list()
    # Extract patches for prediction
    for i in range(m * l, int(np.floor(dtm.shape[0] - w2)), 1):
        for j in range(0, int(int(dtm.shape[1])) - w2, 1):
            raw_list.append(
                dtm[i:i + w2, j:j + w2]
            )
    raw_stack = np.stack(raw_list, axis=-1)
    blocks = np.transpose(raw_stack, [2, 0, 1])
    blocks = np.expand_dims(blocks, axis=-1)
    del (raw_stack)
    # Predict using the model
    pred = model.predict([blocks], verbose=1)
    # Save predictions to npz files
    np.savez(file=temp_dir + str(l) + '.npz', pred=pred)
    del (blocks)
    del (pred)
    gc.collect()

# Read in predictions and write to raster layers
# Open the raster file containing the original data
raw = rasterio.open("D:/GIS/geomorphometry/layers/ponui_dtm_setnull.tif")

mask = raw.read_masks(1)
mask[mask != 0] = 1
#change the mask value
mask = mask.astype(float)
mask[mask == 0] = np.nan

l = list()
# Loop through each prediction file in the directory
for i in range(len(os.listdir(temp_dir))):
    # Load the prediction data
    with np.load(os.path.join(temp_dir + str(i) + '.npz')) as pred:
        l.append(pred['pred'])

# Concatenate the predictions
l = np.concatenate(l)
# Reshape the concatenated predictions to match the original raster shape
l = np.reshape(l, (dtm.shape[0]-w2, dtm.shape[1]-w2))

# Rescale the predictions to the original data range
l = l * (mnmx[y_h][1] - mnmx[y_h][0]) + mnmx[y_h][0]

# Pad the resulting raster to match the original raster dimensions
row_pad = np.ones(shape=(int(np.floor(w2/2)), l.shape[1]), dtype=l.dtype)
l = np.append(row_pad, l, axis=0)
l = np.append(l, np.ones(shape=(int(raw.shape[0]-l.shape[0]), l.shape[1]), dtype=l.dtype), axis=0)

col_pad = np.ones(shape=(l.shape[0], int(np.floor(w2/2))), dtype=l.dtype)
l = np.append(col_pad, l, axis=1)
l = np.append(l, np.ones(shape=(l.shape[0], int(raw.shape[1]-l.shape[1])), dtype=l.dtype), axis=1)

# Apply the mask to the predictions
l = l * mask

# Define affine transformation and reference CRS
aff = rasterio.transform.from_origin(raw.bounds[0], raw.bounds[3], raw.res[0], raw.res[1])
ref = raw.crs

# Define the directory name for the output layer
dir_name = out_dir + y_h + "_" + str(w) + "_" + str(w2) + "_" + str(n) + "_" + str(d) + "_" + str(filt) + "_" + act_name + '.tif'

# Create a new raster file for the output layer
cnn_ft = rasterio.open(
    dir_name,
    'w',
    driver='GTiff',
    height=l.shape[0],
    width=l.shape[1],
    count=1,
    dtype=raw.read(1).dtype,
    crs=ref,
    transform=aff,
    nodata=np.nan
)
# Write the predicted values to the raster file
cnn_ft.write(l, 1)
cnn_ft.close()

#STATISTICAL ANALYSIS

#load original slope
r = rasterio.open("D:/GIS/Geomorphometry/layers/" + y_h + "_1" + ".tif")
r1 = r.read(1)
mask = r.read_masks(1)

#stop if dimensions don't match l
if r1.shape != l.shape:
    print("Dimensions don't match")
    sys.exit()

r1 = r1.astype(float)
r1[mask == 0] = np.nan
r1 = r1.flatten()
l = l.flatten()

if len(l) != len(r1):
    print("Lengths don't match")
    sys.exit()

#bind l and r1 and remove rows with nans
all = np.column_stack((r1, l))
all = all[~np.isnan(all).any(axis=1)]

#compute the RMSE and R-squared, ignoring nan values
sklearn.metrics.r2_score(all[:,0], all[:,1])

#write the results
np.savetxt(
    results_dir + y_h + "_" + str(w) + "_" + str(w2) + "_" + str(n) + "_" + str(d) + "_" + str(filt) + "_" + act_name + ".csv",
    all,
    delimiter=","
)

#cleanup
del (r, r1, mask, l, all, raw, dtm, model, history, aff, batches, pred, input, c1, out)
gc.collect()