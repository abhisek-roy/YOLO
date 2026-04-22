import Foundation
import UIKit

// DocLayNet class names
let classNames = [
    "Caption", "Footnote", "Formula", "List-item", "Page-footer",
    "Page-header", "Picture", "Section-header", "Table", "Text", "Title"
]

//// Initialize detector
//let modelURL = Bundle.main.url(forResource: "YOLOv9_DocLayNet", withExtension: "mlpackage")!
//let detector = try YOLODetector(
//    modelURL: modelURL,
//    classNames: classNames,
//    minConfidence: 0.25,
//    minIOU: 0.45
//)
//
//// Detect objects
//let image = UIImage(named: "document.png")!
//let results = try detector.detect(image: image)
//
//// Print results
//for (className, detection) in results {
//    print("\(className): \(detection.confidence) at \(detection.boundingBox)")
//}
