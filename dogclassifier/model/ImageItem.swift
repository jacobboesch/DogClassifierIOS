import UIKit

class ImageItem{
    var label:String?
    var confidence:Float32?
    var image:UIImage?
    
    init(image:UIImage){
        self.image = image
    }
    
    init(label:String, confidence:Float32, image:UIImage){
        self.label = label
        self.confidence = confidence
        self.image = image
    }
    
    var title:String{
        if(confidence != nil && label != nil){
            let percent:String = String(format: "%.2f", (confidence ?? 0) * 100)
            return "\(label!) \(percent)%"
        }
        
        return "Click to classify"
    }
    
}
