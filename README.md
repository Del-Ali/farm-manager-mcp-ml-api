# Classify Leaf Disease ML backend

This is backend is an model-context-protocol server which has a classify_leaf_disease responsible for the following:

- Leaf detection in an image and drawing of bounding box around areas with leafs
- Model prediction for each leaf detection area in the leaf
- Saving of prediction with leaf detection results for future reference
- Returning prediction after saved
