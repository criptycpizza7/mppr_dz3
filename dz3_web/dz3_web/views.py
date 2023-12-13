from django.shortcuts import render  
from django.core.files.storage import FileSystemStorage  
import onnxruntime  
import numpy as np  
from PIL import Image  
from io import BytesIO  
import base64  
from torchvision import transforms   
  
imageClassList = {0: 'apple', 1: 'aquarium_fish', 2: 'baby', 3: 'bear', 4: 'beaver', 5: 'bed', 6: 'bee', 7: 'beetle', 8: 'bicycle', 9: 'bottle', 10: 'bowl', 11: 'boy', 12: 'bridge', 13: 'bus', 14: 'butterfly', 15: 'camel', 16: 'can', 17: 'castle', 18: 'caterpillar', 19: 'cattle', 20: 'chair', 21: 'chimpanzee', 22: 'clock', 23: 'cloud', 24: 'cockroach', 25: 'couch', 26: 'cra', 27: 'crocodile', 28: 'cup', 29: 'dinosaur', 30: 'dolphin', 31: 'elephant', 32: 'flatfish', 33: 'forest', 34: 'fox', 35: 'girl', 36: 'hamster', 37: 'house', 38: 'kangaroo', 39: 'keyboard', 40: 'lamp', 41: 'lawn_mower', 42: 'leopard', 43: 'lion', 44: 'lizard', 45: 'lobster', 46: 'man', 47: 'maple_tree', 48: 'motorcycle', 49: 'mountain', 50: 'mouse', 51: 'mushroom', 52: 'oak_tree', 53: 'orange', 54: 'orchid', 55: 'otter', 56: 'palm_tree', 57: 'pear', 58: 'pickup_truck', 59: 'pine_tree', 60: 'plain', 61: 'plate', 62: 'poppy', 63: 'porcupine', 64: 'possum', 65: 'rabbit', 66: 'raccoon', 67: 'ray', 68: 'road', 69: 'rocket', 70: 'rose', 71: 'sea', 72: 'seal', 73: 'shark', 74: 'shrew', 75: 'skunk', 76: 'skyscraper', 77: 'snail', 78: 'snake', 79: 'spider', 80: 'squirrel', 81: 'streetcar', 82: 'sunflower', 83: 'sweet_pepper', 84: 'table', 85: 'tank', 86: 'telephone', 87: 'television', 88: 'tiger', 89: 'tractor', 90: 'train', 91: 'trout', 92: 'tulip', 93: 'turtle', 94: 'wardrobe', 95: 'whale', 96: 'willow_tree', 97: 'wolf', 98: 'woman', 99: 'worm'}  #Сюда указать классы  

imageClassList = {21: imageClassList[21], 51: imageClassList[51], 81: imageClassList[81]}
  
def scoreImagePage(request):  
    return render(request, 'scorepage.html')  
  
def predictImage(request):  
    fileObj = request.FILES['filePath']  
    fs = FileSystemStorage()  
    filePathName = fs.save('images/'+fileObj.name,fileObj)  
    filePathName = fs.url(filePathName)  
    modelName = request.POST.get('modelName')  
    scorePrediction, img_uri = predictImageData(modelName, '.'+filePathName)  
    context = {'scorePrediction': scorePrediction, 'filePathName': filePathName, 'img_uri': img_uri}  
    return render(request, 'scorepage.html', context)  
  
def predictImageData(modelName, filePath):  
    img = Image.open(filePath).convert("RGB")  
    resized_img = img.resize((32, 32), Image.LANCZOS)  
    img_uri = to_data_uri(resized_img)  
    input_image = Image.open(filePath)  
    preprocess = transforms.Compose([  
        transforms.Resize(32),  
        transforms.CenterCrop(32),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
        ])  
    input_tensor = preprocess(input_image)  
    input_batch = input_tensor.unsqueeze(0)  
     
    sess = onnxruntime.InferenceSession(r'D:\MPPR\dz3\MPPR\dz3\cifar100_CNN_RESNET20.onnx') #<-Здесь требуется указать свой путь к модели  
    outputOFModel = np.argmax(sess.run(None, {'input': to_numpy(input_batch)}))  
    score = imageClassList[outputOFModel]  
  
    return score, img_uri  
  
def to_numpy(tensor):  
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()  
  
def to_image(numpy_img):  
    img = Image.fromarray(numpy_img, 'RG')  
    return img  
  
def to_data_uri(pil_img):  
    data = BytesIO()  
    pil_img.save(data, "JPEG")  # pick your format  
    data64 = base64.b64encode(data.getvalue())  
    return u'data:img/jpeg;base64,' + data64.decode('utf-8')