import tensorflow as tf
import ROOT

# Parse saved Keras model
model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse("convtranspose_relu.keras", batch_size=1)

# Generate C++ inference code
model.Generate()

print("Conv2DTranspose + ReLU parsing successful")

# Parse saved Keras model
model = ROOT.TMVA.Experimental.SOFIE.PyKeras.Parse("convtranspose_relu.keras", batch_size=1)

# Generate C++ inference code
model.Generate()

print("Conv2DTranspose + ReLU parsing successful")
