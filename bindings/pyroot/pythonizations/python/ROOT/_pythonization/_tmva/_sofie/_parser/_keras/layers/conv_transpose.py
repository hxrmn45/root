import math
from .. import get_keras_version

def MakeKerasConvTranspose(layer):
    """
    Create a Keras-compatible Conv2DTranspose layer operation using SOFIE framework.
    Conv2DTranspose (transposed convolution) is used in decoder networks and upsampling.
    Parameters:
    layer (dict): layer info dict with input, output, dtype, weights, attributes.
    Returns:
    ROperator_ConvTranspose: SOFIE operator for the Conv2DTranspose layer.
    """
    from ROOT.TMVA.Experimental import SOFIE
    finput           = layer["layerInput"]
    foutput          = layer["layerOutput"]
    fLayerDType      = layer["layerDType"]
    fLayerInputName  = finput[0]
    fLayerOutputName = foutput[0]
    attributes       = layer["layerAttributes"]
    fWeightNames     = layer["layerWeight"]
    fKernelName      = fWeightNames[0]
    fBiasName        = fWeightNames[1] if len(fWeightNames) > 1 else ""
    fAttrDilations     = list(attributes.get("dilation_rate", (1, 1)))
    fAttrGroup         = int(attributes.get("groups", 1))
    fAttrKernelShape   = list(attributes.get("kernel_size", (3, 3)))
    fAttrStrides       = list(attributes.get("strides", (1, 1)))
    fKerasPadding      = str(attributes.get("padding", "valid"))
    fAttrOutputPadding = [0, 0]
    fAttrOutputShape   = []
    fAttrPads          = []
    if fKerasPadding == "valid":
        fAttrAutopad = "VALID"
    elif fKerasPadding == "same":
        fAttrAutopad = "SAME_UPPER"
    else:
        raise RuntimeError("TMVA::SOFIE - Conv2DTranspose padding not supported: " + fKerasPadding)
    if SOFIE.ConvertStringToType(fLayerDType) == SOFIE.ETensorType.FLOAT:
        op = SOFIE.ROperator_ConvTranspose["float"](
            fAttrAutopad, fAttrDilations, fAttrGroup, fAttrKernelShape,
            fAttrOutputPadding, fAttrOutputShape, fAttrPads, fAttrStrides,
            fLayerInputName, fKernelName, fBiasName, fLayerOutputName,
        )
        return op
    else:
        raise RuntimeError("TMVA::SOFIE - Conv2DTranspose unsupported type: " + fLayerDType)
