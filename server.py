
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from utils.image_utils import fetch_image
from utils.graphql_utils import create_prediction_mutation
import time
import json

from model.model import RealWorldCropDetectionPipeline
import os

load_dotenv()

PORT = os.environ.get("PORT", 10000)

# Create an MCP server
mcp = FastMCP("disease-classification", host="0.0.0.0", port=PORT, stateless_http=True)


# Add a tool that uses Tavily
@mcp.tool()
async def classify_leaf_disease(image_url: str, crop_type: str, farm_tag: str, token: str):
    """
    Use this tool to predict disease for an uploaded image.

    Args:
        image_url: The url pointing to the image.
        crop_type: this specifies the crop for which we are classifying the disease for (maize, cassava, cashew, tomato).
        farm_tag: the farm for which the prediction is been created for (uuid).
        token: Bearer token for fetching Image resource and also creating Prediction resource (Bearer token)

    Returns:
        prediction
    """
    try:
        pipeline = RealWorldCropDetectionPipeline('yolo')
        # load crop classifier
        models = {
            'maize': 'enhanced_mobilenetv2',
            'tomato': 'enhanced_mobilenetv2',
            'cassava': 'enhanced_mobilenetv2',
            'cashew': 'enhanced_mobilenetv2'
        }
        pipeline.load_crop_classifier(crop_type, models[crop_type])
        # get image using image url with rest endpoint
        image_bytes, image = await fetch_image(image_url, token)
        # process image
        ## get leaf detectios
        leaf_detections = pipeline.detect_leaves_in_image(image)
        # create start_time
        start_time = time.perf_counter()
        # classify leaf disease for every leaf detection
        leaf_detection_result = []
        for i, detection in enumerate(leaf_detections):
            bbox = detection['bbox']
            leaf_confidence = detection['confidence']

            # preprocess leaf for classification
            leaf_image = pipeline.preprocess_leaf_for_classification(image, bbox)

            if leaf_image is None:
                continue

            disease_class, confidence, top3 = pipeline.classify_leaf_disease(leaf_image, crop_type)

            result = {
                'bbox': bbox,
                'detection_confidence': float(leaf_confidence),
                'predicted_disease': (disease_class or "").upper().replace(" ", "_"),
                'confidence': float(confidence),
                'top3_predictions': top3
            }
            leaf_detection_result.append(result)
        # create end_time and find processingTimeInMs
        end_time = time.perf_counter()
        processing_time_ms = (end_time - start_time) * 1000
        # create prediction resource with graphql mutation endpoint
        prediction = await create_prediction_mutation(
            farm_tag=farm_tag,
            crop_type=crop_type.upper(),
            model_used=models[crop_type].upper(),
            leaf_detections=leaf_detection_result,
            image_path=image_url,
            processing_time_ms=processing_time_ms,
            token=token
        )

        prediction_text = {
            'id': prediction["createPrediction"]["id"],
            'crop_type': prediction["createPrediction"]["crop_type"],
            'model_used': prediction["createPrediction"]["model_used"],
            'processing_time_ms': prediction["createPrediction"]["processing_time_ms"],
            'leaf_detections': prediction["createPrediction"]["leaf_detections"]
        }

        prediction_string = json.dumps(prediction_text)

        return [
            {
                'type': "text",
                'text': prediction_string
            }
        ]
    except Exception as e:
        print(e)
        return [
            {
                'type': "text",
                'text':f"{e}"
            }
        ]


# Run the server
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
