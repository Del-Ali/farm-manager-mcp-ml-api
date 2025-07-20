import httpx
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv


async def create_prediction_mutation(
    farm_tag: str,
    crop_type: str,
    model_used: str,
    leaf_detections: List[Dict[str, Any]],
    image_path: str,
    processing_time_ms: float,
    token: Optional[str] = None
):
    """
    Execute CreatePrediction GraphQL mutation.

    Args:
        graphql_url: GraphQL endpoint URL
        farm_tag: Farm tag identifier
        crop_type: Type of crop (PredictionCropType enum)
        model_used: Model type used (ModelType enum)
        leaf_detections: List of leaf detection objects
        image_path: Path to the image
        processing_time_ms: Processing time in milliseconds
        token: Optional bearer token for authentication

    Returns:
        GraphQL response data

    Raises:
        httpx.HTTPError: If request fails
    """


    mutation = """
    mutation CreatePrediction($farmTag: String!, $cropType: PredictionCropType!, $modelUsed: ModelType!, $leafDetections: [LeafDetectionInput!]!, $imagePath: String!, $processingTimeMs: Float!) {
      createPrediction(farmTag: $farmTag, cropType: $cropType, modelUsed: $modelUsed, leafDetections: $leafDetections, imagePath: $imagePath, processingTimeMs: $processingTimeMs) {
        id
        crop_type
        image_path
        model_used
        processing_time_ms
        leaf_detections {
          bbox
          confidence
          detection_confidence
          predicted_disease
          top3_predictions
        }
      }
    }
    """

    variables = {
        "farmTag": farm_tag,
        "cropType": crop_type,
        "modelUsed": model_used,
        "leafDetections": leaf_detections,
        "imagePath": image_path,
        "processingTimeMs": processing_time_ms
    }

    payload = {
        "query": mutation,
        "variables": variables
    }

    headers = {
        "Content-Type": "application/json"
    }

    load_dotenv()

    if token:
        headers["Authorization"] = f"{token}"

    async with httpx.AsyncClient() as client:
        graphql_url = os.environ.get("GRAPHQL_URL") or "https://uzuznia502.execute-api.eu-north-1.amazonaws.com/graphql"
        response = await client.post(graphql_url, json=payload, headers=headers)
        response.raise_for_status()

        result = response.json()

        # Check for GraphQL errors
        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")

        return result["data"]
