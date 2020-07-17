import UIKit

class ImageListController:UIViewController, UICollectionViewDataSource, UICollectionViewDelegate{
    
    @IBOutlet weak var collectionView: UICollectionView!
    
    let images:[String] = [
        "cocker_spaniel.jpg",
        "pomeranian.jpg",
        "bernese_mountain_dog.jpg",
        "siberian_husky.jpg",
        "west_highland_terrier.jpg",
        "yorkshire_terrier.jpg"
    ]
    
    var items:[ImageItem] = []
    
    var classifier:Classifier?
    
    override func viewDidLoad() {
        initlizeItemList()
        self.collectionView.dataSource = self
        self.collectionView.delegate = self
        self.classifier = Classifier("model")
    }
    
    func initlizeItemList(){
        for image in images{
            let item:ImageItem = ImageItem(image: UIImage(named: image)!)
            items.append(item)
        }
    }
    
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return items.count
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "image_view_cell", for: indexPath) as! ImageViewCell
        let item:ImageItem = items[indexPath.item]
        cell.imageView.image = item.image
        cell.inferenceLabel.text = item.title
        
        return cell
    }
    
    func collectionView(_ collectionView:UICollectionView, didSelectItemAt indexPath: IndexPath){
        let item = items[indexPath.item]
        
        guard let pixelBuffer = item.image!.pixelBuffer()
            else{
                return
        }
        
        let classification = self.classifier?.classifyImage(pixelBuffer)
        
        item.confidence = classification?.confidence
        item.label = classification?.label
        
        collectionView.reloadData()
    }
    
}
