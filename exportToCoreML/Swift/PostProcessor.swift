import Foundation
import CoreML
import CoreGraphics

/// Main post-processing pipeline for YOLOv9 detections
/// Equivalent to Python's PostProcess class in model_utils.py
class PostProcessor {
    
    private let decoder: GridDecoder
    private let nms: NMS
    private let modelConfig: ModelConfig
    private let nmsConfig: NMSConfig
    
    /// Initialize post-processor
    /// - Parameters:
    ///   - modelConfig: Model configuration (image size, strides, classes)
    ///   - nmsConfig: NMS configuration (thresholds, max boxes)
    init(modelConfig: ModelConfig, nmsConfig: NMSConfig) {
        self.modelConfig = modelConfig
        self.nmsConfig = nmsConfig
        self.decoder = GridDecoder(config: modelConfig)
        self.nms = NMS(config: nmsConfig)
    }
    
    // MARK: - Main Processing
    
    /// Process raw CoreML outputs to get final detections
    /// - Parameter outputs: Dictionary of MLMultiArray outputs from CoreML model
    /// - Returns: Array of filtered detections
    ///
    /// Python equivalent:
    /// 
    /// def __call__(self, predict, rev_tensor=None, image_size=None):
    ///     prediction = self.converter(predict["Main"])
    ///     pred_class, _, pred_bbox = prediction[:3]
    ///     pred_bbox = bbox_nms(pred_class, pred_bbox, self.nms, pred_conf)
    ///     return pred_bbox
    /// 
    func process(outputs: [String: MLMultiArray]) -> [Detection] {
        
        print("\n🔄 Post-processing started...")
        print("  Input outputs: \(outputs.count)")
        
        // Step 1: Decode grid predictions to bounding boxes
        print("📐 Decoding grid predictions...")
        let (classScores, boxes) = decoder.decode(outputs: outputs)
        print("  Decoded \(boxes.count) anchors with \(classScores.first?.count ?? 0) classes")
        
        // Step 2: Apply NMS to filter detections
        print("🎯 Applying NMS...")
        let detections = nms.apply(classScores: classScores, boxes: boxes)
        
        print("✅ Post-processing complete: \(detections.count) detections")
        
        return detections
    }
    
    /// Process outputs and scale boxes to original image size
    /// - Parameters:
    ///   - outputs: CoreML model outputs
    ///   - originalSize: Original image size before preprocessing
    /// - Returns: Detections with boxes scaled to original image coordinates
    func process(outputs: [String: MLMultiArray], originalSize: CGSize) -> [Detection] {
        
        // Get detections in model input coordinates (e.g., 1024x1024)
        var detections = process(outputs: outputs)
        
        // Scale boxes back to original image size
        let scaleX = originalSize.width / modelConfig.imageSize.width
        let scaleY = originalSize.height / modelConfig.imageSize.height
        
        detections = detections.map { detection in
            let scaledBox = CGRect(
                x: detection.boundingBox.origin.x * scaleX,
                y: detection.boundingBox.origin.y * scaleY,
                width: detection.boundingBox.width * scaleX,
                height: detection.boundingBox.height * scaleY
            )
            
            return Detection(
                classIndex: detection.classIndex,
                confidence: detection.confidence,
                boundingBox: scaledBox
            )
        }
        
        return detections
    }
}

// MARK: - Convenience Extension

extension PostProcessor {
    
    /// Create a post-processor with default DocLayNet configuration
    static func forDocLayNet(
        minConfidence: Float = 0.25,
        minIOU: Float = 0.45,
        maxBBoxes: Int = 100
    ) -> PostProcessor {
        
        let modelConfig = ModelConfig(
            imageSize: CGSize(width: 1024, height: 1024),
            strides: [8, 16, 32],
            numClasses: 11  // DocLayNet has 11 classes
        )
        
        let nmsConfig = NMSConfig(
            minConfidence: minConfidence,
            minIOU: minIOU,
            maxBBoxes: maxBBoxes
        )
        
        return PostProcessor(modelConfig: modelConfig, nmsConfig: nmsConfig)
    }
}