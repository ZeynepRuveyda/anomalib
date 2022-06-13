"""Anomalib Gradio Script.

This script provide a gradio web interface
"""

from argparse import ArgumentParser, Namespace
from importlib import import_module
from pathlib import Path
from sys import meta_path
from typing import Optional, Tuple

import gradio as gr
import gradio.inputs
import gradio.outputs
import numpy as np
from skimage.segmentation import mark_boundaries

from anomalib.config import get_configurable_parameters
from anomalib.deploy.inferencers.base import Inferencer
from anomalib.post_processing import compute_mask, superimpose_anomaly_map

models_dict_paths = { 
    1: {
        "color" : "red",
        "weights" : "/home/zeynep/zeynep/anomalib/results/padim/mvtec/FL_yedek/weights/model.ckpt",
        "config": "/home/zeynep/zeynep/anomalib/anomalib/models/padim/config_red.yaml",
        "meta_data": ""
    },
    2: {
        "color" : "black",
        "weights" : "/home/zeynep/zeynep/anomalib/results/padim/mvtec/FNG_Black/weights/model2.ckpt",
        "config": "/home/zeynep/zeynep/anomalib/anomalib/models/padim/config_black.yaml",
        "meta_data": ""
    },
    3: {
        "color" : "white",
        "weights" : "/home/zeynep/zeynep/anomalib/results/padim/mvtec/FNG_White/weights/model1.ckpt",
        "config": "/home/zeynep/zeynep/anomalib/anomalib/models/padim/config_white.yaml",
        "meta_data": ""
    }
}

def infer(
    image: np.ndarray, threshold: float = 50.0, color: str = "red"
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    """Inference function, return anomaly map, score, heat map, prediction mask ans visualisation.

    Args:
        image (np.ndarray): image to compute
        inferencer (Inferencer): model inferencer
        threshold (float, optional): threshold between 0 and 100. Defaults to 50.0.

    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        anomaly_map, anomaly_score, heat_map, pred_mask, vis_img
    """
    # Get inferencer 
    for color_value in models_dict_paths:
        print(color_value)
        if models_dict_paths[color_value]["color"] == color:
            print(models_dict_paths[color_value])
            weights_path = Path(models_dict_paths[color_value]["weights"])
            config_path = Path(models_dict_paths[color_value]["config"])
            meta_path = Path(models_dict_paths[color_value]["meta_data"])
            break
    inferencer =get_inferencer(config_path, weights_path)

    # Perform inference for the given image.
    threshold = threshold / 100
    anomaly_map, anomaly_score  = inferencer.predict(image=image, superimpose=False)
    heat_map = superimpose_anomaly_map(anomaly_map, image)
    pred_mask = compute_mask(anomaly_map, threshold)
    vis_img = mark_boundaries(image, pred_mask, color=(1, 1, 0.2), mode="thick")
    return  heat_map, vis_img


# def get_args() -> Namespace:
#     """Get command line arguments.

#     Example:

#         >>> python tools/inference_gradio.py \
#              --config_path ./anomalib/models/padim/config.yaml \
#              --weight_path ./results/padim/mvtec/bottle/weights/model.ckpt

#     Returns:
#         Namespace: List of arguments.
#     """
#     parser = ArgumentParser()
#     parser.add_argument("--config_path", type=Path, required=True, help="Path to a model config file")
#     parser.add_argument("--weight_path", type=Path, required=True, help="Path to a model weights")
#     parser.add_argument(
#         "--meta_data_path", type=Path, required=False, help="Path to JSON file containing the metadata."
#     )

#     parser.add_argument(
#         "--threshold",
#         type=float,
#         required=False,
#         default=75.0,
#         help="Value to threshold anomaly scores into 0-100 range",
#     )

#     parser.add_argument("--share", type=bool, required=False, default=False, help="Share Gradio `share_url`")

#     args = parser.parse_args()

#     return args


def get_inferencer(config_path: Path, weight_path: Path, meta_data_path: Optional[Path] = None) -> Inferencer:
    """Parse args and open inferencer.

    Args:
        config_path (Path): Path to model configuration file or the name of the model.
        weight_path (Path): Path to model weights.
        meta_data_path (Optional[Path], optional): Metadata is required for OpenVINO models. Defaults to None.

    Raises:
        ValueError: If unsupported model weight is passed.

    Returns:
        Inferencer: Torch or OpenVINO inferencer.
    """
    config = get_configurable_parameters(config_path=config_path)

    # Get the inferencer. We use .ckpt extension for Torch models and (onnx, bin)
    # for the openvino models.
    extension = weight_path.suffix
    inferencer: Inferencer
    if extension in (".ckpt"):
        module = import_module("anomalib.deploy.inferencers.torch")
        TorchInferencer = getattr(module, "TorchInferencer")
        inferencer = TorchInferencer(config=config, model_source=weight_path, meta_data_path=meta_data_path)

    elif extension in (".onnx", ".bin", ".xml"):
        module = import_module("anomalib.deploy.inferencers.openvino")
        OpenVINOInferencer = getattr(module, "OpenVINOInferencer")
        inferencer = OpenVINOInferencer(config=config, path=weight_path, meta_data_path=meta_data_path)

    else:
        raise ValueError(
            f"Model extension is not supported. Torch Inferencer exptects a .ckpt file,"
            f"OpenVINO Inferencer expects either .onnx, .bin or .xml file. Got {extension}"
        )

    return inferencer


if __name__ == "__main__":
    # session_args = get_args()

    # gradio_inferencer = get_inferencer(session_args.config_path, session_args.weight_path, session_args.meta_data_path)
    

    interface = gr.Interface(
        # fn=lambda image, threshold: infer(image, threshold, color),
        infer, 
        inputs=[
            gradio.inputs.Image(
                shape=None, 
                image_mode="RGB", 
                source="upload", 
                tool="editor", 
                type="numpy", 
                label="Image"
            ),
            
            gradio.inputs.Dropdown(
                ["red", "black", "white"],
                type="value",
                default="None",
                label="Color"
            ),
            gradio.inputs.Slider(
                default=25.0, 
                label="threshold", 
                optional=False
            )
            
        ],
        outputs=[
	    gradio.outputs.Image(
            type="numpy", 
            label="Predicted Heat Map"
            ),
            gradio.outputs.Image(
                type="numpy", 
                label="Segmentation Result"
                ),

        ],
        title="FLEX-N-GATE Automatic Surface Aspect Analysis ",
        description="""
<center>
Hayon 3008 
</center> 
<img src="https://media-exp1.licdn.com/dms/image/C4E1BAQGDF4PE_EElLQ/company-background_10000/0/1645959037142?e=2147483647&v=beta&t=6y_feSrzbSqlb26HopTmQjQVT4ohaujaqF5HsjqSJOY"
<left>
1.Click on "image" for take a picture on your phone.
2.Choose the color of part on the menu to be inspected.     
<left>
3.Set the threshold value according to default information.
""",
        article="article",
    )

    interface.launch(server_name="0.0.0.0", server_port=8080, share="public")
