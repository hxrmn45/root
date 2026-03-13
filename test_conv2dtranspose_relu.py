import ROOT
import tensorflow as tf

# load the keras model
keras_model = tf.keras.models.load_model("convtranspose_relu.keras")

# parse with SOFIE
model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse(
    keras_model,
    batch_size=1
)

print("Conv2DTranspose + ReLU parsing successful")
