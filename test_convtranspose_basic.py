import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import tensorflow as tf
import ROOT
from tensorflow.keras.layers import Conv2DTranspose, Input
from tensorflow.keras.models import Model
import numpy as np

# Create simple Conv2DTranspose model
inp = Input(shape=(16,16,1))
x = Conv2DTranspose(4, (3,3), strides=(2,2), padding="same")(inp)

model = Model(inp, x)

# Save model
model.save("convtranspose_basic.keras")

# Parse with SOFIE
parsed_model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse("convtranspose_basic.keras")

# Generate inference code
parsed_model.Generate()

print("Conv2DTranspose basic parsing successful")

# Run TensorFlow inference to verify model
data = np.random.rand(1,16,16,1).astype(np.float32)
output = model.predict(data)

print("Output shape:", output.shape)

# Run TensorFlow inference to verify model
data = np.random.rand(1,16,16,1).astype(np.float32)
output = model.predict(data)

print("Output shape:", output.shape)
