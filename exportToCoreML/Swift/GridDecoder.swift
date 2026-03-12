import Foundation
import CoreML
import CoreGraphics

/// Decodes raw YOLO grid predictions into bounding boxes
/// Equivalent to Python's Vec2Box class
class GridDecoder {
    
    private let config: ModelConfig
    private var anchorGrids: [[CGPoint]] = []  // Anchor points for each scale
    private var scalers: [[Float]] = []         // Stride scalers for each scale
    
    init(config: ModelConfig) {
        self.config = config
        generateAnchors()
    }
    
    // MARK: - Anchor Generation
    
    /// Generate anchor grids for each stride
    /// Python equivalent: generate_anchors() in bounding_box_utils.py
    private func generateAnchors() {
        let width = Int(config.imageSize.width)
        let height = Int(config.imageSize.height)
        
        for stride in config.strides {
            let gridWidth = width / stride
            let gridHeight = height / stride
            let shift = stride / 2
            
            var anchors: [CGPoint] = []
            var strides: [Float] = []
            
            for y in 0..<gridHeight {
                for x in 0..<gridWidth {
                    let anchorX = Float(x * stride + shift)
                    let anchorY = Float(y * stride + shift)
                    anchors.append(CGPoint(x: CGFloat(anchorX), y: CGFloat(anchorY)))
                    strides.append(Float(stride))
                }
            }
            
            anchorGrids.append(anchors)
            scalers.append(strides)
        }
    }
    
    // MARK: - Decode Predictions
    
    /// Decode raw model outputs to bounding boxes
    /// - Parameter outputs: Dictionary of 9 MLMultiArray outputs from CoreML
    /// - Returns: Tuple of (class predictions, bounding boxes)
    func decode(outputs: [String: MLMultiArray]) -> (classScores: [[Float]], boxes: [CGRect]) {
        
        // Extract and parse the 9 outputs
        // Scale 0 (128x128): outputs 0, 1, 2
        // Scale 1 (64x64):   outputs 3, 4, 5
        // Scale 2 (32x32):   outputs 6, 7, 8
        
        var allClassScores: [[Float]] = []
        var allBoxes: [CGRect] = []
        
        let outputKeys = outputs.keys.sorted()
        
        for scaleIdx in 0..<3 {
            let baseIdx = scaleIdx * 3
            
            // Get outputs for this scale
            guard baseIdx + 2 < outputKeys.count,
                  let classPred = outputs[outputKeys[baseIdx]],      // class predictions
                  let boxDist = outputs[outputKeys[baseIdx + 1]],    // box distribution (not used in v9)
                  let boxCoord = outputs[outputKeys[baseIdx + 2]]    // box coordinates
            else {
                print("⚠️ Missing outputs for scale \(scaleIdx)")
                continue
            }
            
            // Decode this scale
            let (scaleClassScores, scaleBoxes) = decodeScale(
                classPred: classPred,
                boxCoord: boxCoord,
                scaleIndex: scaleIdx
            )
            
            allClassScores.append(contentsOf: scaleClassScores)
            allBoxes.append(contentsOf: scaleBoxes)
        }
        
        return (allClassScores, allBoxes)
    }
    
    /// Decode predictions for a single scale
    private func decodeScale(classPred: MLMultiArray, boxCoord: MLMultiArray, scaleIndex: Int) -> ([[Float]], [CGRect]) {
        
        // classPred shape: [1, numClasses, gridH, gridW]
        // boxCoord shape:  [1, 4, gridH, gridW]
        
        let numClasses = classPred.shape[1].intValue
        let gridH = classPred.shape[2].intValue
        let gridW = classPred.shape[3].intValue
        let numAnchors = gridH * gridW
        
        var classScores: [[Float]] = []
        var boxes: [CGRect] = []
        
        let anchors = anchorGrids[scaleIndex]
        let strides = scalers[scaleIndex]
        
        // Iterate through each grid cell
        for gridY in 0..<gridH {
            for gridX in 0..<gridW {
                let anchorIdx = gridY * gridW + gridX
                let anchor = anchors[anchorIdx]
                let stride = strides[anchorIdx]
                
                // Extract class scores for this grid cell
                var scores: [Float] = []
                for c in 0..<numClasses {
                    let idx = [0, c, gridY, gridX] as [NSNumber]
                    let score = classPred[idx].floatValue
                    scores.append(score)
                }
                
                // Extract box coordinates (LTRB format: left, top, right, bottom distances)
                let leftIdx = [0, 0, gridY, gridX] as [NSNumber]
                let topIdx = [0, 1, gridY, gridX] as [NSNumber]
                let rightIdx = [0, 2, gridY, gridX] as [NSNumber]
                let bottomIdx = [0, 3, gridY, gridX] as [NSNumber]
                
                let left = boxCoord[leftIdx].floatValue * stride
                let top = boxCoord[topIdx].floatValue * stride
                let right = boxCoord[rightIdx].floatValue * stride
                let bottom = boxCoord[bottomIdx].floatValue * stride
                
                // Decode to absolute coordinates
                // Python: preds_box = torch.cat([self.anchor_grid - lt, self.anchor_grid + rb], dim=-1)
                let xMin = Float(anchor.x) - left
                let yMin = Float(anchor.y) - top
                let xMax = Float(anchor.x) + right
                let yMax = Float(anchor.y) + bottom
                
                let box = CGRect.fromMinMax(xMin: xMin, yMin: yMin, xMax: xMax, yMax: yMax)
                
                classScores.append(scores)
                boxes.append(box)
            }
        }
        
        return (classScores, boxes)
    }
}