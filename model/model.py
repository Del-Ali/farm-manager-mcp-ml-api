# ===== CELL 8: REAL-WORLD LEAF DETECTION AND CROP CLASSIFICATION PIPELINE =====

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from ultralytics import YOLO
from utils.model_utils import load_saved_model_weights

class RealWorldCropDetectionPipeline:
    """
    Complete pipeline for real-world crop disease detection from drone/phone images
    Includes: leaf detection, crop segmentation, disease classification
    """

    def __init__(self, detection_method='yolo', confidence_threshold=0.2):
        self.detection_method = detection_method
        self.confidence_threshold = confidence_threshold
        self.leaf_detector = None
        self.crop_classifiers = {}
        self.setup_leaf_detection()

    def setup_leaf_detection(self):
        """Setup leaf detection model (YOLOv8, SAM, or traditional CV)"""
        print("🔍 Setting up leaf detection pipeline...")

        if self.detection_method == 'yolo':
            self.setup_yolo_detection()
        elif self.detection_method == 'sam':
            self.setup_sam_detection()
        elif self.detection_method == 'traditional':
            self.setup_traditional_detection()
        else:
            print(f"⚠️ Unknown detection method: {self.detection_method}")
            print("🔄 Falling back to traditional computer vision...")
            self.detection_method = 'traditional'
            self.setup_traditional_detection()

    def setup_yolo_detection(self):
        """Setup YOLOv8 for leaf detection"""
        try:
            print("📦 Setting up YOLOv8 leaf detection...")
            # Try to import and setup YOLOv8
            model_path = "./models/weights/yolo/yolov8n.pt"
            self.leaf_detector = YOLO(model_path)

            self.leaf_detector.conf = self.confidence_threshold  # confidence threshold
            self.leaf_detector.iou = 0.45

            print(f"✅ YOLOv8 detector ready (loaded from {model_path})")

        except Exception as e:
            print(f"❌ YOLOv8 setup failed: {e}")
            print("🔄 Falling back to traditional detection...")
            self.setup_traditional_detection()

    def setup_sam_detection(self):
        """Setup Segment Anything Model (SAM) for leaf segmentation"""
        try:
            print("📦 Setting up SAM (Segment Anything) detection...")
            print("⚠️ SAM requires significant setup and models")
            print("🔄 Creating mock SAM detector for demonstration...")

            class MockSAM:
                def __init__(self):
                    self.model_name = "SAM-leaf-segmenter"

                def segment_leaves(self, image):
                    """Mock segmentation - returns leaf masks"""
                    h, w = image.shape[:2]
                    num_leaves = np.random.randint(2, 5)

                    segments = []
                    for i in range(num_leaves):
                        # Create random elliptical mask for leaf
                        mask = np.zeros((h, w), dtype=np.uint8)
                        center_x = np.random.randint(50, w-50)
                        center_y = np.random.randint(50, h-50)
                        axes_x = np.random.randint(30, 80)
                        axes_y = np.random.randint(40, 100)

                        cv2.ellipse(mask, (center_x, center_y), (axes_x, axes_y),
                                   np.random.randint(0, 180), 0, 360, 255, -1)

                        segments.append({
                            'mask': mask,
                            'bbox': cv2.boundingRect(mask),
                            'confidence': np.random.uniform(0.7, 0.95)
                        })

                    return segments

            self.leaf_detector = MockSAM()
            print("✅ Mock SAM detector ready")

        except Exception as e:
            print(f"❌ SAM setup failed: {e}")
            print("🔄 Falling back to traditional detection...")
            self.setup_traditional_detection()

    def setup_traditional_detection(self):
        """Setup traditional computer vision for leaf detection"""
        print("📦 Setting up traditional CV leaf detection...")

        class TraditionalLeafDetector:
            def __init__(self):
                self.model_name = "Traditional-CV-detector"

            def detect_leaves(self, image):
                """Traditional CV leaf detection using color and contour analysis"""
                # Convert to different color spaces
                hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

                # Define green color ranges for leaves
                lower_green1 = np.array([35, 40, 40])
                upper_green1 = np.array([85, 255, 255])

                lower_green2 = np.array([25, 30, 30])
                upper_green2 = np.array([95, 255, 255])

                # Create masks for green regions
                mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
                mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
                green_mask = cv2.bitwise_or(mask1, mask2)

                # Morphological operations to clean up
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
                green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

                # Find contours
                contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                detections = []
                for contour in contours:
                    area = cv2.contourArea(contour)

                    # Filter by area (adjust based on image size)
                    min_area = (image.shape[0] * image.shape[1]) * 0.001  # 0.1% of image
                    max_area = (image.shape[0] * image.shape[1]) * 0.3    # 30% of image

                    if min_area < area < max_area:
                        # Calculate bounding box
                        x, y, w, h = cv2.boundingRect(contour)

                        # Calculate aspect ratio and solidity for leaf-like shapes
                        aspect_ratio = float(w) / h
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = float(area) / hull_area if hull_area > 0 else 0

                        # Filter based on leaf-like characteristics
                        if 0.3 < aspect_ratio < 3.0 and solidity > 0.5:
                            confidence = min(0.95, solidity * 0.8 + (area / max_area) * 0.2)

                            detections.append({
                                'bbox': [x, y, x + w, y + h],
                                'confidence': confidence,
                                'class': 'leaf',
                                'area': area,
                                'contour': contour
                            })

                # Sort by confidence and return top detections
                detections.sort(key=lambda x: x['confidence'], reverse=True)
                return detections[:10]  # Return top 10 detections

        self.leaf_detector = TraditionalLeafDetector()
        print("✅ Traditional CV detector ready")

    def detect_leaves_in_image(self, image):
        """Detect leaves in a real-world image"""
        print(f"🔍 Detecting leaves using {self.detection_method} method...")

        if self.detection_method == 'yolo':
            # Run YOLO prediction
            results = self.leaf_detector(image)

            # Process results into our standard format
            detections = []
            for result in results:
                for box in result.boxes:
                    # Convert tensor to list and get coordinates
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf.item()
                    cls_id = box.cls.item()
                    cls_name = result.names[cls_id]

                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'class': cls_name,
                        'class_id': cls_id
                    })

            return detections
        elif self.detection_method == 'sam':
            return self.leaf_detector.segment_leaves(image)
        elif self.detection_method == 'traditional':
            return self.leaf_detector.detect_leaves(image)
        else:
            return []

    def load_crop_classifier(self, crop_name, model_type='ultra_quick'):
        """Load a trained crop disease classifier"""
        try:
            model, class_info = load_saved_model_weights(crop_name, model_type)
            if model is not None:
                self.crop_classifiers[crop_name] = {
                    'model': model,
                    'class_info': class_info,
                    'classes': class_info['unified_classes'],
                    'class_indices': class_info['model_params']
                }
                print(f"✅ Loaded {crop_name} classifier ({model_type})")
                return True
            else:
                print(f"❌ Failed to load {crop_name} classifier")
                return False
        except Exception as e:
            print(f"❌ Error loading {crop_name} classifier: {e}")
            return False

    def preprocess_leaf_for_classification(self, image, bbox):
        """Extract and preprocess leaf region for disease classification"""
        x1, y1, x2, y2 = bbox

        # Extract leaf region
        leaf_region = image[y1:y2, x1:x2]

        if leaf_region.size == 0:
            return None

        # Resize to model input size (224x224)
        leaf_resized = cv2.resize(leaf_region, (224, 224))

        # Convert BGR to RGB
        leaf_rgb = cv2.cvtColor(leaf_resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        leaf_normalized = leaf_rgb.astype(np.float32) / 255.0

        # Add batch dimension
        leaf_batch = np.expand_dims(leaf_normalized, axis=0)

        return leaf_batch

    def classify_leaf_disease(self, leaf_image, crop_type):
        """Classify disease in a detected leaf"""
        if crop_type not in self.crop_classifiers:
            return None, 0.0, "Classifier not loaded"

        try:
            classifier = self.crop_classifiers[crop_type]
            model = classifier['model']
            classes = classifier['classes']

            # Get prediction
            predictions = model.predict(leaf_image, verbose=0)

            # Get top prediction
            top_idx = np.argmax(predictions[0])
            confidence = predictions[0][top_idx]
            predicted_class = classes[top_idx]

            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = []

            for idx in top3_indices:
                top3_predictions.append({
                    'class': classes[idx],
                    'confidence': float(predictions[0][idx])
                })

            return predicted_class, float(confidence), top3_predictions

        except Exception as e:
            return None, 0.0, f"Classification error: {e}"

    def process_real_world_image(self, image_path, crop_types=['tomato', 'cassava', 'maize'],
                                model_type='ultra_quick', visualize=True):
        """
        Complete pipeline: detect leaves → classify diseases

        Args:
            image_path: Path to drone/phone image
            crop_types: List of crop types to try for classification
            model_type: Type of classifier to use
            visualize: Whether to show results

        Returns:
            Dictionary with detections and classifications
        """
        print(f"\n🚀 Processing real-world image: {image_path}")

        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return None
        else:
            image = image_path  # Already loaded image

        print(f"📸 Image size: {image.shape}")

        # Load classifiers for specified crops
        for crop in crop_types:
            if crop not in self.crop_classifiers:
                self.load_crop_classifier(crop, model_type)

        # Detect leaves
        leaf_detections = self.detect_leaves_in_image(image)
        print(f"🍃 Found {len(leaf_detections)} potential leaves")

        # Process each detected leaf
        results = {
            'image_shape': image.shape,
            'detection_method': self.detection_method,
            'num_leaves_detected': len(leaf_detections),
            'leaf_classifications': []
        }

        for i, detection in enumerate(leaf_detections):
            print(f"\n🔬 Processing leaf {i+1}/{len(leaf_detections)}...")

            bbox = detection['bbox']
            leaf_confidence = detection['confidence']

            # Preprocess leaf for classification
            leaf_image = self.preprocess_leaf_for_classification(image, bbox)

            if leaf_image is None:
                continue

            # Try classification with each crop type
            leaf_result = {
                'leaf_id': i + 1,
                'bbox': bbox,
                'detection_confidence': leaf_confidence,
                'crop_classifications': {}
            }

            best_classification = None
            best_confidence = 0.0

            for crop_type in crop_types:
                if crop_type in self.crop_classifiers:
                    disease_class, confidence, top3 = self.classify_leaf_disease(leaf_image, crop_type)

                    leaf_result['crop_classifications'][crop_type] = {
                        'predicted_disease': disease_class,
                        'confidence': confidence,
                        'top3_predictions': top3
                    }

                    # Track best overall classification
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_classification = {
                            'crop_type': crop_type,
                            'disease': disease_class,
                            'confidence': confidence
                        }

            leaf_result['best_classification'] = best_classification
            results['leaf_classifications'].append(leaf_result)

            if best_classification:
                print(f"   🎯 Best: {best_classification['crop_type']} - {best_classification['disease']} ({best_classification['confidence']:.3f})")

        # Visualize results if requested
        if visualize:
            self.visualize_results(image, results)

        return results

    def visualize_results(self, image, results):
        """Visualize detection and classification results"""
        print("\n📊 Visualizing results...")

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        # Original image with detections
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(image_rgb)
        axes[0].set_title(f'Leaf Detection ({results["detection_method"]})\n{results["num_leaves_detected"]} leaves found')
        axes[0].axis('off')

        # Color map for different crops
        crop_colors = {
            'tomato': 'red',
            'cassava': 'blue',
            'maize': 'yellow',
            'cashew': 'green'
        }

        # Draw bounding boxes and labels
        for leaf in results['leaf_classifications']:
            bbox = leaf['bbox']
            x1, y1, x2, y2 = bbox

            # Determine color based on best classification
            best = leaf.get('best_classification')
            if best:
                color = crop_colors.get(best['crop_type'], 'white')
                label = f"{best['crop_type']}: {best['disease']}\n{best['confidence']:.3f}"
            else:
                color = 'gray'
                label = f"Leaf {leaf['leaf_id']}"

            # Draw bounding box
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                   linewidth=2, edgecolor=color, facecolor='none')
            axes[0].add_patch(rect)

            # Add label
            axes[0].text(x1, y1-5, label, fontsize=8, color=color,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

        # Classification summary
        axes[1].axis('off')
        axes[1].set_title('Classification Summary')

        # Create summary text
        summary_text = []
        summary_text.append(f"Detection Method: {results['detection_method']}")
        summary_text.append(f"Leaves Detected: {results['num_leaves_detected']}")
        summary_text.append("")

        # Count classifications by crop and disease
        crop_disease_counts = {}
        for leaf in results['leaf_classifications']:
            best = leaf.get('best_classification')
            if best:
                crop = best['crop_type']
                disease = best['disease']
                key = f"{crop} - {disease}"

                if key not in crop_disease_counts:
                    crop_disease_counts[key] = {'count': 0, 'avg_confidence': 0, 'confidences': []}

                crop_disease_counts[key]['count'] += 1
                crop_disease_counts[key]['confidences'].append(best['confidence'])

        # Calculate average confidences
        for key in crop_disease_counts:
            confidences = crop_disease_counts[key]['confidences']
            crop_disease_counts[key]['avg_confidence'] = np.mean(confidences)

        # Display summary
        if crop_disease_counts:
            summary_text.append("Disease Classifications:")
            for key, data in sorted(crop_disease_counts.items(), key=lambda x: x[1]['count'], reverse=True):
                summary_text.append(f"  {key}: {data['count']} leaves (avg conf: {data['avg_confidence']:.3f})")
        else:
            summary_text.append("No confident classifications found")

        # Add crop classifier info
        summary_text.append("")
        summary_text.append("Loaded Classifiers:")
        for crop in self.crop_classifiers:
            classifier = self.crop_classifiers[crop]
            accuracy = classifier['class_info'].get('accuracy', 'N/A')
            summary_text.append(f"  {crop}: {len(classifier['classes'])} classes (acc: {accuracy})")

        axes[1].text(0.05, 0.95, '\n'.join(summary_text), transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.show()

    def create_detection_report(self, results, save_path=None):
        """Create a detailed detection report"""
        report = {
            'timestamp': str(pd.Timestamp.now()),
            'image_info': {
                'shape': results['image_shape'],
                'detection_method': results['detection_method']
            },
            'summary': {
                'total_leaves_detected': results['num_leaves_detected'],
                'total_classified': len([l for l in results['leaf_classifications'] if l.get('best_classification')]),
                'detection_success_rate': len([l for l in results['leaf_classifications'] if l.get('best_classification')]) / max(1, results['num_leaves_detected'])
            },
            'detailed_results': results['leaf_classifications']
        }

        # Add aggregated statistics
        crop_stats = {}
        disease_stats = {}

        for leaf in results['leaf_classifications']:
            best = leaf.get('best_classification')
            if best:
                crop = best['crop_type']
                disease = best['disease']

                if crop not in crop_stats:
                    crop_stats[crop] = 0
                crop_stats[crop] += 1

                if disease not in disease_stats:
                    disease_stats[disease] = 0
                disease_stats[disease] += 1

        report['statistics'] = {
            'crops_detected': crop_stats,
            'diseases_detected': disease_stats
        }

        if save_path:
            with open(save_path, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📄 Report saved to: {save_path}")

        return report

# Initialize the real-world detection pipeline
print("🌱 REAL-WORLD CROP DISEASE DETECTION PIPELINE")
print("=" * 60)
print("This pipeline handles:")
print("  🔍 Leaf detection from drone/phone images")
print("  🎯 Disease classification of detected leaves")
print("  📊 Multi-crop analysis and reporting")
print("  🖼️ Visualization of results")
print("=" * 60)

# Create detection pipeline with different methods
pipelines = {
    'traditional': RealWorldCropDetectionPipeline('traditional'),
    'yolo': RealWorldCropDetectionPipeline('yolo'),
    'sam': RealWorldCropDetectionPipeline('sam')
}

print(f"\n✅ Real-world detection pipeline ready!")
print(f"Available methods: {list(pipelines.keys())}")

# ===== DEMO FUNCTION FOR TESTING =====

def demo_real_world_detection(pipeline_type='traditional'):
    """Demo the real-world detection pipeline with a sample image"""
    print(f"\n🎬 DEMO: Real-world detection using {pipeline_type} method")

    # Create a sample "drone image" for demo
    print("📸 Creating sample drone image for demo...")

    # Create a realistic sample image with multiple leaf-like objects
    demo_image = np.zeros((600, 800, 3), dtype=np.uint8)

    # Add background (sky/soil)
    demo_image[:300, :] = [135, 206, 235]  # Sky blue
    demo_image[300:, :] = [139, 69, 19]    # Brown soil

    # Add some "leaves" as green ellipses
    leaf_positions = [
        (150, 250, 60, 40),  # x, y, width, height
        (350, 180, 80, 50),
        (550, 220, 70, 45),
        (200, 400, 75, 55),
        (450, 380, 65, 40),
        (650, 350, 90, 60)
    ]

    for x, y, w, h in leaf_positions:
        # Random green shade for each leaf
        green_shade = np.random.randint(40, 120)
        color = (0, green_shade + 60, green_shade)

        # Draw ellipse for leaf
        cv2.ellipse(demo_image, (x, y), (w, h),
                   np.random.randint(0, 180), 0, 360, color, -1)

        # Add some texture/noise
        noise = np.random.randint(-20, 20, (h*2, w*2, 3))
        y1, y2 = max(0, y-h), min(600, y+h)
        x1, x2 = max(0, x-w), min(800, x+w)
        demo_image[y1:y2, x1:x2] = np.clip(
            demo_image[y1:y2, x1:x2].astype(int) + noise[:y2-y1, :x2-x1], 0, 255
        ).astype(np.uint8)

    # Get pipeline
    pipeline = pipelines[pipeline_type]

    # Process the demo image
    results = pipeline.process_real_world_image(
        demo_image,
        crop_types=['tomato', 'cassava', 'maize'],
        model_type='ultra_quick',
        visualize=True
    )

    if results:
        # Create report
        report = pipeline.create_detection_report(results)

        print(f"\n📊 DEMO RESULTS SUMMARY:")
        print(f"   Leaves detected: {results['num_leaves_detected']}")
        print(f"   Successfully classified: {len([l for l in results['leaf_classifications'] if l.get('best_classification')])}")
        print(f"   Detection method: {results['detection_method']}")

        return results, report
    else:
        print("❌ Demo failed")
        return None, None

print(f"\n🎬 Ready for demo! Try:")
print(f"```python")
print(f"# Test with traditional CV method")
print(f"results, report = demo_real_world_detection('traditional')")
print(f"")
print(f"# Test with mock YOLO method")
print(f"results, report = demo_real_world_detection('yolo')")
print(f"```")
