import Foundation
import CoreML
import UIKit

/// Complete YOLO detector with CoreML model
class YOLODetector {
    
    private let model: MLModel
    private let postProcessor: PostProcessor
    private let classNames: [String]
    
    /// Initialize detector
    /// - Parameters:
    ///   - modelURL: URL to .mlpackage file
    ///   - classNames: Array of class names
    ///   - minConfidence: Minimum confidence threshold
    ///   - minIOU: NMS IOU threshold
    init(modelURL: URL, classNames: [String], minConfidence: Float = 0.25, minIOU: Float = 0.45) throws {
        
        // Load CoreML model
        let config = MLModelConfiguration()
        config.computeUnits = .all  // Use Neural Engine if available
        self.model = try MLModel(contentsOf: modelURL, configuration: config)
        
        // Initialize post-processor
        self.postProcessor = PostProcessor.forDocLayNet(
            minConfidence: minConfidence,
            minIOU: minIOU
        )
        
        self.classNames = classNames
        
        print("✅ YOLODetector initialized")
        print("  Model: \(modelURL.lastPathComponent)")
        print("  Classes: \(classNames.count)")
    }
    
    // MARK: - Detection
    
    /// Detect objects in an image
    /// - Parameter image: Input UIImage
    /// - Returns: Array of detections with class names
    func detect(image: UIImage) throws -> [(className: String, detection: Detection)] {
        
        print("\n🔍 Running detection...")
        
        // Step 1: Preprocess image
        guard let resizedImage = image.resize(to: CGSize(width: 1024, height: 1024)) else {
            throw DetectorError.preprocessingFailed
        }
        
        guard let pixelBuffer = resizedImage.pixelBuffer() else {
            throw DetectorError.pixelBufferCreationFailed
        }
        
        // Step 2: Run CoreML model
        let input = try MLDictionaryFeatureProvider(dictionary: ["image": pixelBuffer])
        let output = try model.prediction(from: input)
        
        // Step 3: Extract outputs
        var outputs: [String: MLMultiArray] = [:]
        for key in output.featureNames {
            if let array = output.featureValue(for: key)?.multiArrayValue {
                outputs[key] = array
            }
        }
        
        print("  CoreML outputs: \(outputs.count)")
        
        // Step 4: Post-process
        let detections = postProcessor.process(
            outputs: outputs,
            originalSize: image.size
        )
        
        // Step 5: Add class names
        let results = detections.map { detection in
            let className = classNames[detection.classIndex]
            return (className: className, detection: detection)
        }
        
        print("✅ Detection complete: \(results.count) objects found")
        
        return results
    }
}

// MARK: - Errors

enum DetectorError: Error {
    case preprocessingFailed
    case pixelBufferCreationFailed
    case modelPredictionFailed
}

// MARK: - Image Preprocessing Extensions

extension UIImage {
    
    /// Resize image to target size
    func resize(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, 1.0)
        defer { UIGraphicsEndImageContext() }
        self.draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
    
    /// Convert UIImage to CVPixelBuffer (required for CoreML)
    func pixelBuffer() -> CVPixelBuffer? {
        let width = Int(self.size.width)
        let height = Int(self.size.height)
        
        let attrs = [
            kCVPixelBufferCGImageCompatibilityKey: kCFBooleanTrue,
            kCVPixelBufferCGBitmapContextCompatibilityKey: kCFBooleanTrue
        ] as CFDictionary
        
        var pixelBuffer: CVPixelBuffer?
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32ARGB,
            attrs,
            &pixelBuffer
        )
        
        guard status == kCVReturnSuccess, let buffer = pixelBuffer else {
            return nil
        }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue
        )
        
        guard let cgImage = self.cgImage, let ctx = context else {
            return nil
        }
        
        ctx.draw(cgImage, in: CGRect(x: 0, y: 0, width: width, height: height))
        
        return buffer
    }
}