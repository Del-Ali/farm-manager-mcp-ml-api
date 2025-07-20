import re
import os
import numpy as np
import shutil
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50, MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

def setup_tpu():
    """Initialize TPU and return strategy"""
    try:
        # Try to connect to TPU
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print(f"🔍 TPU Detected: {tpu.cluster_spec().as_dict()}")

        # Connect to TPU
        # tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)

        # Create TPU strategy
        strategy = tf.distribute.TPUStrategy(tpu)

        print(f"✅ TPU Initialized Successfully!")
        print(f"   TPU cores: {strategy.num_replicas_in_sync}")
        print(f"   Strategy: {type(strategy).__name__}")

        return strategy, True

    except Exception as e:
        print(f"⚠️  TPU not available: {str(e)}")
        print("🔄 Falling back to CPU/GPU...")

        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ GPU Available: {len(gpus)} device(s)")
            # Enable memory growth for GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            strategy = tf.distribute.MirroredStrategy()
        else:
            print("💻 Using CPU")
            strategy = tf.distribute.get_strategy()  # Default strategy

        return strategy, False

# Initialize compute strategy
strategy, using_tpu = setup_tpu()

GLOBAL_STRATEGY = strategy
USING_TPU = using_tpu

# TPU-optimized batch size
if using_tpu:
    # TPUs work best with batch sizes that are multiples of 8
    BATCH_SIZE = 32  # Good for TPU
    print(f"📦 Using TPU-optimized batch size: {BATCH_SIZE}")
else:
    BATCH_SIZE = 16  # Conservative for GPU/CPU
    print(f"📦 Using standard batch size: {BATCH_SIZE}")


class ProgressiveCropDiseaseDetector:
    def __init__(self, data_path, img_size=(224, 224), batch_size=None):
        self.data_path = data_path
        self.img_size = img_size
        self.batch_size = batch_size if batch_size else BATCH_SIZE
        self.class_names = {}
        self.class_mappings = {}  # Store class mappings for consistency
        self.strategy = GLOBAL_STRATEGY

        print(f"🔧 Progressive Detector initialized with:")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Image size: {self.img_size}")
        print(f"   Strategy: {type(self.strategy).__name__}")

    def create_data_generators(self, crop_name):
        """Create enhanced data generators with stronger augmentation"""
        crop_path = os.path.join(self.data_path, crop_name)
        train_path = os.path.join(crop_path, 'train_set')
        test_path = os.path.join(crop_path, 'test_set')

        # Enhanced data augmentation for better generalization
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            shear_range=0.2,
            fill_mode='reflect',
            validation_split=0.2
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            interpolation='bilinear'
        )

        validation_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            interpolation='bilinear'
        )

        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            interpolation='bilinear'
        )

        self.class_names[crop_name] = list(train_generator.class_indices.keys())

        return train_generator, validation_generator, test_generator

    def create_enhanced_mobilenet_model(self, num_classes):
        """Create enhanced MobileNetV2 with progressive training capability"""
        with self.strategy.scope():
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3),
                alpha=1.0
            )
            base_model.trainable = False

            model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.BatchNormalization(),
                layers.Dropout(0.4),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                layers.Dense(num_classes, activation='softmax')
            ], name='enhanced_mobilenet')

            model._base_model = base_model  # Attach for fine-tuning

            print(f"📱 Enhanced MobileNetV2 created with {model.count_params():,} parameters")

        return model

    def create_enhanced_resnet_model(self, num_classes):
        """Create enhanced ResNet50 with progressive training capability"""
        with self.strategy.scope():
            base_model = ResNet50(
                weights='imagenet',
                include_top=False,
                input_shape=(*self.img_size, 3)
            )
            base_model.trainable = False

            inputs = base_model.output
            x = layers.GlobalAveragePooling2D()(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.5)(x)
            x = layers.Dense(1024, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)
            outputs = layers.Dense(num_classes, activation='softmax')(x)
            print("CREATING_ENHANCED_RESNET_MODEL")

            model = keras.Model(inputs=base_model.input, outputs=outputs, name='enhanced_resnet')
            model._base_model = base_model

            print(f"🏗️ Enhanced ResNet50 created with {model.count_params():,} parameters")

        return model

    def create_enhanced_custom_cnn(self, num_classes):
        """Create enhanced custom CNN with residual connections"""
        with self.strategy.scope():
            inputs = layers.Input(shape=(*self.img_size, 3))

            x = layers.Conv2D(64, 3, padding='same')(inputs)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(64, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)

            x = layers.Conv2D(128, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(128, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)

            x = layers.Conv2D(256, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(256, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(2)(x)
            x = layers.Dropout(0.25)(x)

            x = layers.Conv2D(512, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.Conv2D(512, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation('relu')(x)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.5)(x)

            x = layers.Dense(1024, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.4)(x)
            x = layers.Dense(512, activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(0.3)(x)

            outputs = layers.Dense(num_classes, activation='softmax')(x)

            model = keras.Model(inputs=inputs, outputs=outputs, name='enhanced_custom_cnn')

            print(f"🧠 Enhanced Custom CNN created with {model.count_params():,} parameters")

        return model

    def progressive_train_model(self, model, train_gen, val_gen, crop_name, epochs_per_phase=15):
        """Progressive training with fine-tuning phases"""
        model_name = model.name
        print(f"\n🚀 Starting progressive training for {model_name} on {crop_name}")

        print(f"\n📚 PHASE 1: Training classifier head (base frozen)")
        with self.strategy.scope():
            initial_lr = 0.001 * (self.strategy.num_replicas_in_sync if USING_TPU else 1)

            model.compile(
                optimizer=Adam(learning_rate=initial_lr),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )

            phase1_callbacks = [
                EarlyStopping(patience=8, restore_best_weights=True, verbose=1),
                ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, verbose=1)
            ]

        print(f"   Learning rate: {initial_lr}")
        print(f"   Trainable params: {model.count_params():,}")

        history1 = model.fit(
            train_gen,
            epochs=epochs_per_phase,
            validation_data=val_gen,
            callbacks=phase1_callbacks,
            verbose=1
        )

        base_model = getattr(model, "_base_model", None)
        if base_model is None:
            for layer in model.layers:
                if isinstance(layer, keras.Model) and hasattr(layer, 'layers') and len(layer.layers) > 10:
                    base_model = layer
                    break

        if base_model is not None:
            print(f"\n🔓 PHASE 2: Fine-tuning (unfreezing top layers)")
            if 'mobilenet' in model_name.lower():
                unfreeze_from = max(0, len(base_model.layers) - 30)
                fine_tune_lr = initial_lr * 0.1
            elif 'resnet' in model_name.lower():
                unfreeze_from = max(0, len(base_model.layers) - 40)
                fine_tune_lr = initial_lr * 0.1
            else:
                unfreeze_from = 0
                fine_tune_lr = initial_lr * 0.5

            for layer in base_model.layers[unfreeze_from:]:
                layer.trainable = True

            trainable_count = sum(1 for layer in model.layers for weight in layer.trainable_weights)
            print(f"   Unfroze {len(base_model.layers) - unfreeze_from} layers")
            print(f"   Learning rate: {fine_tune_lr}")
            print(f"   Trainable params: {trainable_count:,}")

            with self.strategy.scope():
                model.compile(
                    optimizer=Adam(learning_rate=fine_tune_lr),
                    loss='categorical_crossentropy',
                    metrics=['accuracy']
                )

                phase2_callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True, verbose=1),
                    ReduceLROnPlateau(factor=0.3, patience=5, min_lr=1e-8, verbose=1)
                ]

            history2 = model.fit(
                train_gen,
                epochs=epochs_per_phase,
                validation_data=val_gen,
                callbacks=phase2_callbacks,
                verbose=1
            )

            combined_history = {}
            for key in history1.history.keys():
                combined_history[key] = history1.history[key] + history2.history[key]

            class CombinedHistory:
                def __init__(self, history_dict):
                    self.history = history_dict

            return CombinedHistory(combined_history)
        else:
            return history1

    def evaluate_model(self, model, test_gen):
        """Enhanced model evaluation with additional metrics"""
        print(f"📊 Evaluating {model.name}...")

        test_loss, test_acc = model.evaluate(test_gen, verbose=0)
        predictions = model.predict(test_gen, verbose=0)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes

        top3_acc = tf.keras.metrics.sparse_top_k_categorical_accuracy(
            y_true, predictions, k=3
        ).numpy().mean()

        max_confidences = np.max(predictions, axis=1)
        avg_confidence = np.mean(max_confidences)

        print(f"   Accuracy: {test_acc:.4f}")
        print(f"   Top-3 Accuracy: {top3_acc:.4f}")
        print(f"   Average Confidence: {avg_confidence:.4f}")
        print(f"   Loss: {test_loss:.4f}")

        return {
            'accuracy': test_acc,
            'top3_accuracy': top3_acc,
            'avg_confidence': avg_confidence,
            'loss': test_loss,
            'predictions': predictions,
            'y_pred': y_pred,
            'y_true': y_true
        }

def extract_base_name(class_name):
    """Extract base name from various patterns"""
    name = class_name.strip().lower()

    # Pattern 1: Remove numbers at the end (e.g., "healthy1" -> "healthy")
    pattern1 = re.match(r'^(.+?)(\d+)$', name)
    if pattern1:
        base = pattern1.group(1).strip()
        if base:
            return base

    # Pattern 2: Remove numbers with space/separator
    pattern2 = re.match(r'^(.+?)(\s+)(\d+)$', name)
    if pattern2:
        base = pattern2.group(1).strip()
        if base:
            return base

    # Pattern 3: Remove underscore followed by numbers
    pattern3 = re.match(r'^(.+?)(_\d+)$', name)
    if pattern3:
        base = pattern3.group(1).strip()
        if base:
            return base

    return name

def create_unified_class_mapping(original_classes):
    """Create mapping from original classes to unified base classes"""
    mapping = {}
    seen_bases = {}

    print(f"\n🔍 Creating unified class mapping for: {original_classes}")

    # Sort classes to ensure consistent processing
    def sort_key(name):
        has_numbers = bool(re.search(r'\d', name))
        return (has_numbers, len(name), name)

    sorted_classes = sorted(original_classes, key=sort_key)

    for class_name in sorted_classes:
        base_name = extract_base_name(class_name)

        if base_name in seen_bases:
            # Map to existing representative
            representative = seen_bases[base_name]
            mapping[class_name] = representative
            print(f"   🔗 Mapping: {class_name} → {representative}")
        else:
            # This is the representative for this base name
            seen_bases[base_name] = class_name
            mapping[class_name] = class_name
            print(f"   ✅ Base: {class_name} (represents '{base_name}')")

    unique_classes = list(set(mapping.values()))
    print(f"\n📊 Mapping summary: {len(original_classes)} → {len(unique_classes)} classes")
    print(f"   Final unified classes: {sorted(unique_classes)}")

    return mapping, unique_classes

class FixedEnhancedDetector(ProgressiveCropDiseaseDetector):
    """Enhanced detector with FIXED class mapping that creates physically restructured directories"""

    def __init__(self, data_path, img_size=(224, 224), batch_size=None):
        super().__init__(data_path, img_size, batch_size)
        self.temp_data_path = "/tmp/unified_dataset"  # Temporary unified dataset path
        self.unified_class_names = {}
        self.processed_crops = set()

    # IMMEDIATE FIX: Update the create_unified_dataset_structure method

    def create_unified_dataset_structure(self, crop_name):
        """Create a unified dataset structure with merged classes - FIXED for test set"""
        crop_path = os.path.join(self.data_path, crop_name)
        train_path = os.path.join(crop_path, 'train_set')
        test_path = os.path.join(crop_path, 'test_set')

        # Create temporary paths for unified dataset
        temp_crop_path = os.path.join(self.temp_data_path, crop_name)
        temp_train_path = os.path.join(temp_crop_path, 'train_set')
        temp_test_path = os.path.join(temp_crop_path, 'test_set')

        # Remove existing temp directory if it exists
        if os.path.exists(temp_crop_path):
            shutil.rmtree(temp_crop_path)

        # Create new directory structure
        os.makedirs(temp_train_path, exist_ok=True)
        os.makedirs(temp_test_path, exist_ok=True)

        # Get original classes from BOTH train and test sets
        train_classes = [d for d in os.listdir(train_path)
                        if os.path.isdir(os.path.join(train_path, d))]

        test_classes = []
        if os.path.exists(test_path):
            test_classes = [d for d in os.listdir(test_path)
                           if os.path.isdir(os.path.join(test_path, d))]

        # Combine and get unique classes
        all_original_classes = list(set(train_classes + test_classes))

        print(f"📊 Found classes:")
        print(f"   Train classes: {train_classes}")
        print(f"   Test classes: {test_classes}")
        print(f"   Combined unique: {all_original_classes}")

        # Create mapping based on ALL classes (not just train)
        class_mapping, unified_classes = create_unified_class_mapping(all_original_classes)

        # Create unified class directories
        for unified_class in unified_classes:
            os.makedirs(os.path.join(temp_train_path, unified_class), exist_ok=True)
            os.makedirs(os.path.join(temp_test_path, unified_class), exist_ok=True)

        # Copy and organize files according to mapping
        def copy_files_with_mapping(source_path, dest_path, dataset_type, available_classes):
            total_copied = 0
            for original_class in available_classes:
                if original_class in class_mapping:
                    unified_class = class_mapping[original_class]

                    source_class_path = os.path.join(source_path, original_class)
                    dest_class_path = os.path.join(dest_path, unified_class)

                    if os.path.exists(source_class_path):
                        # Copy all files from original class to unified class
                        files = [f for f in os.listdir(source_class_path)
                                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

                        for file in files:
                            source_file = os.path.join(source_class_path, file)
                            # Create unique filename if mapping causes conflicts
                            if original_class != unified_class:
                                # Add prefix to avoid filename conflicts
                                new_filename = f"{original_class}_{file}"
                            else:
                                new_filename = file

                            dest_file = os.path.join(dest_class_path, new_filename)
                            shutil.copy2(source_file, dest_file)
                            total_copied += 1

                        if original_class != unified_class:
                            print(f"   📁 {dataset_type}: {original_class} → {unified_class} ({len(files)} files)")
                        else:
                            print(f"   📁 {dataset_type}: {original_class} (no mapping, {len(files)} files)")
                else:
                    print(f"   ⚠️ {dataset_type}: {original_class} not in mapping!")

            return total_copied

        print(f"\n🔄 Creating unified dataset structure for {crop_name}...")

        # Copy train files
        train_files = copy_files_with_mapping(train_path, temp_train_path, "Train", train_classes)

        # Copy test files (THIS WAS THE MISSING PIECE!)
        test_files = copy_files_with_mapping(test_path, temp_test_path, "Test", test_classes)

        print(f"✅ Unified dataset created: {train_files} train files, {test_files} test files")
        print(f"   📂 Location: {temp_crop_path}")

        # Store the unified class names and mapping
        self.unified_class_names[crop_name] = unified_classes
        self.class_mappings[crop_name] = class_mapping

        return temp_crop_path

    # Apply the fix to your detector
    # detector.create_unified_dataset_structure = create_unified_dataset_structure_fixed.__get__(detector, FixedEnhancedDetector)

    print("🔧 Applied test set mapping fix!")
    print("\nNow run:")
    print("train_gen, val_gen, test_gen = detector.create_data_generators('Cashew')")
    print("print(f'Fixed test samples: {test_gen.samples}')")

    def create_data_generators(self, crop_name):
        """Create generators using the unified dataset structure"""
        # Create unified dataset if not already processed
        if crop_name not in self.processed_crops:
            unified_crop_path = self.create_unified_dataset_structure(crop_name)
            self.processed_crops.add(crop_name)
        else:
            unified_crop_path = os.path.join(self.temp_data_path, crop_name)

        # Use the unified dataset path
        train_path = os.path.join(unified_crop_path, 'train_set')
        test_path = os.path.join(unified_crop_path, 'test_set')

        # Enhanced data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            brightness_range=[0.7, 1.3],
            channel_shift_range=0.2,
            shear_range=0.2,
            fill_mode='reflect',
            validation_split=0.2
        )

        test_datagen = ImageDataGenerator(rescale=1./255)

        # Create generators from unified dataset
        train_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            interpolation='bilinear'
        )

        validation_generator = train_datagen.flow_from_directory(
            train_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=True,
            interpolation='bilinear'
        )

        test_generator = test_datagen.flow_from_directory(
            test_path,
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            interpolation='bilinear'
        )

        # Verify class consistency
        train_classes = list(train_generator.class_indices.keys())
        val_classes = list(validation_generator.class_indices.keys())
        test_classes = list(test_generator.class_indices.keys())

        assert train_classes == val_classes == test_classes, "Class mismatch between generators!"

        # Store consistent class names
        self.class_names[crop_name] = train_classes

        print(f"\n✅ Generators created with consistent classes:")
        print(f"   Classes ({len(train_classes)}): {train_classes}")
        print(f"   Train samples: {train_generator.samples}")
        print(f"   Validation samples: {validation_generator.samples}")
        print(f"   Test samples: {test_generator.samples}")

        return train_generator, validation_generator, test_generator

    def cleanup_temp_dataset(self):
        """Clean up temporary dataset after training"""
        if os.path.exists(self.temp_data_path):
            shutil.rmtree(self.temp_data_path)
            print(f"🧹 Cleaned up temporary dataset: {self.temp_data_path}")


def load_saved_model_weights(crop, model_type='ultra_quick'):
    """
    Helper function to load saved model weights for inference

    Args:
        crop: 'tomato', 'cassava', 'maize', 'cashew'
        model_type: 'ultra_quick', 'quick', 'enhanced_mobilenetv2', 'enhanced_resnet50', 'enhanced_customcnn'

    Returns:
        model: Loaded model ready for inference
        class_info: Dictionary with class information
    """
    import json

    # Determine filenames based on model type
    if model_type == 'ultra_quick':
        weights_file = f"{crop.lower()}_ultra_quick.weights.h5"
        info_file = f"{crop.lower()}_ultra_quick_classes.json"
    elif model_type == 'quick':
        weights_file = f"{crop.lower()}_quick_mobilenet.weights.h5"
        info_file = f"{crop.lower()}_quick_class_info.json"
    else:
        weights_file = f"models/weights/{crop.lower()}/{crop.lower()}_{model_type}.weights.h5"
        info_file = f"models/json/{crop.lower()}/{crop.lower()}_{model_type}_metadata.json"

    # Load class info
    try:
        with open(info_file, 'r') as f:
            class_info = json.load(f)
    except FileNotFoundError:
        print(f"❌ Class info file not found: {info_file}")
        return None, None

    # Create model architecture
    try:
        detector = FixedEnhancedDetector(
            data_path='/kaggle/input/crop-disease-detection/CCMT Dataset-Augmented',
            img_size=(224, 224),
            batch_size=16
        )

        num_classes = len(class_info['unified_classes'])

        if 'mobilenet' in model_type.lower() or model_type in ['ultra_quick', 'quick']:
            model = detector.create_enhanced_mobilenet_model(num_classes)
        elif 'resnet' in model_type.lower():
            model = detector.create_enhanced_resnet_model(num_classes)
        elif 'custom' in model_type.lower():
            model = detector.create_enhanced_custom_cnn(num_classes)
        else:
            print(f"❌ Unknown model type: {model_type}")
            return None, None

        # Load weights
        model.load_weights(weights_file)

        print(f"✅ Loaded {crop} {model_type} model:")
        print(f"   📁 Weights: {weights_file}")
        print(f"   📋 Classes: {len(class_info['unified_classes'])}")
        print(f"   🎯 Accuracy: {class_info.get('accuracy', 'N/A')}")

        return model, class_info

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None
