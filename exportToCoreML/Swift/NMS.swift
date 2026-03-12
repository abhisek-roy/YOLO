import Foundation
import CoreGraphics

/// Non-Maximum Suppression implementation
/// Equivalent to Python's bbox_nms() in bounding_box_utils.py
class NMS {
    
    private let config: NMSConfig
    
    init(config: NMSConfig) {
        self.config = config
    }
    
    // MARK: - Main NMS Function
    
    /// Apply NMS to filter detections
    /// - Parameters:
    ///   - classScores: Array of class scores for each anchor [numAnchors][numClasses]
    ///   - boxes: Array of bounding boxes [numAnchors]
    /// - Returns: Filtered detections
    func apply(classScores: [[Float]], boxes: [CGRect]) -> [Detection] {
        
        guard classScores.count == boxes.count else {
            print("⚠️ Mismatch: classScores count (\(classScores.count)) != boxes count (\(boxes.count))")
            return []
        }
        
        var candidates: [Detection] = []
        
        // Step 1: Apply sigmoid and filter by confidence threshold
        for (anchorIdx, scores) in classScores.enumerated() {
            let sigmoidScores = scores.sigmoid()
            let box = boxes[anchorIdx]
            
            // Check each class
            for (classIdx, score) in sigmoidScores.enumerated() {
                if score >= config.minConfidence {
                    candidates.append(Detection(
                        classIndex: classIdx,
                        confidence: score,
                        boundingBox: box
                    ))
                }
            }
        }
        
        print("📊 Candidates after confidence filtering: \(candidates.count)")
        
        // Step 2: Group by class (NMS is applied per class)
        var detectionsByClass: [Int: [Detection]] = [:]
        for detection in candidates {
            detectionsByClass[detection.classIndex, default: []].append(detection)
        }
        
        // Step 3: Apply NMS for each class
        var finalDetections: [Detection] = []
        
        for (classIdx, classDetections) in detectionsByClass {
            let nmsDetections = applyNMSForClass(detections: classDetections)
            finalDetections.append(contentsOf: nmsDetections)
            print("  Class \(classIdx): \(classDetections.count) → \(nmsDetections.count) after NMS")
        }
        
        // Step 4: Sort by confidence and take top-k
        finalDetections.sort { $0.confidence > $1.confidence }
        let topK = Array(finalDetections.prefix(config.maxBBoxes))
        
        print("✅ Final detections: \(topK.count)")
        
        return topK
    }
    
    // MARK: - NMS for Single Class
    
    /// Apply NMS for a single class
    /// Python equivalent: batched_nms() from torchvision
    private func applyNMSForClass(detections: [Detection]) -> [Detection] {
        
        guard !detections.isEmpty else { return [] }
        
        // Sort by confidence (descending)
        var sorted = detections.sorted { $0.confidence > $1.confidence }
        var keep: [Detection] = []
        
        while !sorted.isEmpty {
            // Take the detection with highest confidence
            let best = sorted.removeFirst()
            keep.append(best)
            
            // Remove detections that overlap significantly with this one
            sorted = sorted.filter { candidate in
                let iou = best.boundingBox.iou(with: candidate.boundingBox)
                return iou < config.minIOU  // Keep if IOU is below threshold
            }
        }
        
        return keep
    }
}

// MARK: - Helper Extension

extension Array where Element == Detection {
    /// Debug description
    func summary() -> String {
        let classCounts = Dictionary(grouping: self, by: { $0.classIndex })
            .mapValues { $0.count }
            .sorted { $0.key < $1.key }
        
        var lines = ["Detections: \(self.count)"]
        for (classIdx, count) in classCounts {
            lines.append("  Class \(classIdx): \(count)")
        }
        return lines.joined(separator: "\n")
    }
}