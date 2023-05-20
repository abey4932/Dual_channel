import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(#图片预处理
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # load image
    img_path = "D:/pytorch project/Alex/1.jpg"#在python库中载入图片
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)#判断图片是否存在
    img = Image.open(img_path)

    plt.imshow(img)#展示图片
    # [N, C, H, W]
    img = data_transform(img)#对图片进行预处理操作
    # expand batch dimension  #batch是指一次处理图片的数量，批
    img = torch.unsqueeze(img, dim=0)#处理后变成[batch, C, H, W]

    # read class_indict
    json_path = './class_indices.json'#读取索引对应的类别名称
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)#将json文件解码成字典模式

    json_file = open(json_path, "r")
    class_indict = json.load(json_file)

    # create model
    model = AlexNet(num_classes=7).to(device)#初始化网络，类别为5

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))#载入网络模型

    model.eval()#进入eval模式，即关闭dropout方法
    with torch.no_grad():#让变量不去跟踪模型的损失梯度
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()#通过正向传播得到输出，并将输出进行压缩，将batch维度压缩
        predict = torch.softmax(output, dim=0)#通过softmax处理之后变成概率分布
        predict_cla = torch.argmax(predict).numpy()#获取概率最大处对应的索引值

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())#打印类别名称以及预测正确的概率
    plt.title(print_res)
    print(print_res)
    plt.show()


if __name__ == '__main__':
    main()

