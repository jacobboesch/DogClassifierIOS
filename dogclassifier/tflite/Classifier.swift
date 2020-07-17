import UIKit
import CoreImage
import TensorFlowLite


struct Classification{
    let confidence:Float32
    let label:String
}

class Classifier{
    // number of threads the tensorflow intepreter can use
    private var threadCount:Int = 1
    
    // width and height of the input in pixels
    private let inputWidth:Int = 224
    private let inputHeight:Int = 224
    
    // number of channels for the image in our case RGB data
    private let numChannels:Int = 3
    
    // TensorFlowLite interpreter responsible for running the model
    private var interpreter:Interpreter!
    
    private var labels:[String] = []
    
    init?(_ modelFileName:String,_ threadCount:Int = 1){
        // path to tflite file
        guard let modelPath = Bundle.main.path(
            forResource: modelFileName, ofType: "tflite") else {
            print("Could not locate file \(modelFileName)")
            return nil
        }
        
        self.threadCount = threadCount
        var options = Interpreter.Options()
        options.threadCount = threadCount
        
        // intilize the interpreter
        do{
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            try initLabels()
        }
        catch let error{
            print("Failed to create the interpreter: \(error.localizedDescription)")
            return nil
        }
    }
    
    func initLabels() throws{
        let path = Bundle.main.path(forResource: "labels", ofType: ".txt")
        let fileContents = try String(contentsOfFile: path!, encoding: String.Encoding.utf8)
        self.labels = fileContents.components(separatedBy: "\n")
    }
    
    func classifyImage(_ pixelBuffer: CVPixelBuffer) -> Classification?{
        // used to scale our image to the input size of our tensor
        let scaledSize:CGSize = CGSize(width: inputWidth, height: inputHeight)
        // get the scaled thumbnail image from the pixel buffer so we can use it as input
        guard let thumbnailPixelBuffer = pixelBuffer.centerThumbnail(ofSize: scaledSize)
            else{
                return nil
        }
        
        let outputTensor:Tensor
        do{
            // allocate memory for the input and output tensors
            try interpreter.allocateTensors()
            
            // get RGB data and by removing the alpha component of the pixelBuffer
            guard let rgbData = fetchDataFromPixelBuffer(
                thumbnailPixelBuffer,
                numBytes: inputWidth * inputHeight * numChannels
            )else{
                return nil
            }
            
            // copy the RGB data to the input tensor
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run the tflite model
            try interpreter.invoke()
            
            // get the output tensor
            outputTensor = try interpreter.output(at: 0)
            
        }
        catch let error{
            print(error.localizedDescription)
            return nil
        }
        
        let results = [Float32](unsafeData: outputTensor.data)
        
        // TODO return the results
        return getTopResult(results: results)
    }
    

    private func getTopResult(results: [Float32]) -> Classification{
        let labeledResults = zip(labels.indices, results)
        let sortedResults = labeledResults.sorted{$0.1 > $1.1}.prefix(1)
        let result = sortedResults[0]
        
        return Classification(confidence: result.1, label: labels[result.0])
    }
    
    
    private func fetchDataFromPixelBuffer(_ buffer:CVPixelBuffer, numBytes:Int) -> Data?{
        // lock the memory address of the pixel buffer so that we know we can work with
        // this spot in memory without it changing the contents
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        
        // at the end of this function unlock the memory address so that the system can once again
        // freely use it
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        
        // create a pointer to the memory address of the pixel buffer
        guard let pixelPointer = CVPixelBufferGetBaseAddress(buffer) else {
             return nil
        }
        
        // create a Data object that points to the memory location of our PixelBuffer
        let bufferData = Data(
            bytesNoCopy: pixelPointer,
            count: CVPixelBufferGetDataSize(buffer),
            deallocator: .none
        )
        
        // create 3 floats for each pixel of the image. for RGB data
        var rgbData = [Float](repeating: 0, count: numBytes)
        
        // our pixel buffer is in RGBA and we only want to keep RGB data
        // therefore the alpha comonent is the fourth byte of every pixel
        let alphaComponent = (baseOffset: 4, moduloRemainder: 3)
        
        var index:Int = 0
        
        // loop though the bytes in the bufferedData.
        // compenent is a byte in the bufferedData
        for component in bufferData.enumerated(){
            // offset is which byte the compnent is pointing to in the bufferedData
            let offset = component.offset
            
            if((offset % alphaComponent.baseOffset) == alphaComponent.moduloRemainder){
                // if the current component is the alpha component then skip it
                continue
            }
            
            // cast the compnent element to a float and divid it by 255 since
            // it's returning an interger between 0 and 255 we need to divide it to make it between 0 and 1
            rgbData[index] = Float(component.element) / 255.0
            index += 1
        }
        
        return rgbData.withUnsafeBufferPointer(Data.init) // return the float array as data
    }
    
}

// intilize an array from raw data
extension Array{
    init(unsafeData: Data) {
      self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
    }
}


