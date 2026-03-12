import Foundation
import CoreGraphics

// MARK: - Configuration

/// NMS (Non-Maximum Suppression) configuration
struct NMSConfig {
    let minConfidence: Float    // Minimum confidence threshold (e.g., 0.25)
    let minIOU: Float            // IOU threshold for NMS (e.g., 0.45)
    let maxBBoxes: Int           // Maximum number of boxes to return (e.g., 100)
    
    init(minConfidence: Float = 0.25, minIOU: Float = 0.45, maxBBoxes: Int = 100) {
        self.minConfidence = minConfidence
        self.minIOU = minIOU
        self.maxBBoxes = maxBBoxes
    }
}

/// Model configuration
struct ModelConfig {
    let imageSize: CGSize        // Input size (e.g., 1024x1024)
    let strides: [Int]           // Feature map strides [8, 16, 32]
    let numClasses: Int          // Number of classes (11 for DocLayNet)
    
    init(imageSize: CGSize = CGSize(width: 1024, height: 1024),
         strides: [Int] = [8, 16, 32],
         numClasses: Int = 11) {
        self.imageSize = imageSize
        self.strides = strides
        self.numClasses = numClasses
    }
}

// MARK: - Data Structures

/// Represents a detected bounding box
struct Detection {
    let classIndex: Int          // Class ID (0-10 for DocLayNet)
    let confidence: Float        // Confidence score (0.0-1.0)
    let boundingBox: CGRect      // Bounding box (x, y, width, height)
    
    init(classIndex: Int, confidence: Float, boundingBox: CGRect) {
        self.classIndex = classIndex
        self.confidence = confidence
        self.boundingBox = boundingBox
    }
}

/// Grid cell prediction
struct GridPrediction {
    let classScores: [Float]     // Class scores for this grid cell
    let boundingBox: CGRect      // Predicted bounding box
    let confidence: Float        // Objectness confidence
}

// MARK: - Utilities

extension CGRect {
    /// Converts from (x_center, y_center, width, height) to (x, y, width, height)
    static func fromCenter(cx: Float, cy: Float, width: Float, height: Float) -> CGRect {
        return CGRect(
            x: CGFloat(cx - width / 2),
            y: CGFloat(cy - height / 2),
            width: CGFloat(width),
            height: CGFloat(height)
        )
    }
    
    /// Converts from (x_min, y_min, x_max, y_max) to CGRect
    static func fromMinMax(xMin: Float, yMin: Float, xMax: Float, yMax: Float) -> CGRect {
        return CGRect(
            x: CGFloat(xMin),
            y: CGFloat(yMin),
            width: CGFloat(xMax - xMin),
            height: CGFloat(yMax - yMin)
        )
    }
    
    /// Calculate IOU (Intersection over Union) with another rectangle
    func iou(with other: CGRect) -> Float {
        let intersection = self.intersection(other)
        if intersection.isNull {
            return 0.0
        }
        
        let intersectionArea = intersection.width * intersection.height
        let unionArea = self.width * self.height + other.width * other.height - intersectionArea
        
        return Float(intersectionArea / unionArea)
    }
}

// MARK: - Helper Extensions

extension Array where Element == Float {
    /// Apply sigmoid function
    func sigmoid() -> [Float] {
        return self.map { 1.0 / (1.0 + exp(-$0)) }
    }
    
    /// Find index of maximum value
    func argmax() -> Int? {
        guard !self.isEmpty else { return nil }
        return self.enumerated().max(by: { $0.element < $1.element })?.offset
    }
}