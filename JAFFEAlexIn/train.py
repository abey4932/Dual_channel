import os
import json
import torch
import torch.nn as nn
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from tqdm import tqdm
from model import AlexNet


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #r如果当前有可使用的gpu，默认使用第一块gpu设备。如果没有gpu就使用cpu设备
    print("using {} device.".format(device))

    data_transform = {
    	#该方法为数据预处理方法
        #当关键字为train时，返回训练集的数据与处理方法
        "train": transforms.Compose([transforms.RandomResizedCrop(224),#将图片用随机裁剪方法裁剪成224*224
                                     transforms.RandomHorizontalFlip(),#在水平方向随机翻转
                                     transforms.ToTensor(),#将它转化成tnesor
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                                     #将数据进行标准化处理

        #当关键字为val时，返回训练集的数据与处理方法
        "val": transforms.Compose([transforms.Resize((224, 224)),#将图片转化成224*224大小
                                   transforms.ToTensor(),#将数据转化成tensor
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                   #将数据进行标准化处理
                                   }

    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  
    #返回到上一级目录的上一级目录,获取数据的根目录
    image_path = "D:/pytorch project/Bishe/JAFFEAlexIn/jaffe_data" #os.path.join(data_root, "Alex", "jaffe_data")  
    #再进入到data_set下的jaffe_data文件夹下
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)#查看是否找到该文件
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
                                 


	#传入train，使用训练集的数据处理方法处理数据
    train_num = len(train_dataset)#将训练集中的图片个数赋值给train_num

    # {'daisy':0, 'dandelion':1, 'roses':2, 'sunjaffe':3, 'tulips':4}
    jaffe_list = train_dataset.class_to_idx#获取分类名称所对应的索引
    cla_dict= dict((val, key) for key, val in jaffe_list.items())
    #遍历所获取的分类以及索引的字典，并且将key,values交换位置
    
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4) #将字典编码成json格式
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 16#定义batch_size=32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=0)
    #train_loader函数是为了随机在数据集中获取一批批数据，num_workers=0加载数据的线程个数，在windows系统下该数为                 0，意思为在windows系统下使用一个主线程加载数据

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=True,
                                                  num_workers=0)

    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))
    net = AlexNet(num_classes=7, init_weights=True)#num_classes=5花有5种类别，初始化权重

    net.to(device)#将该网络分配到制定的设备上（gpu或者cpu）
    loss_function = nn.CrossEntropyLoss()#定义损失函数，针对多类别的损失交叉熵函数
    # pata = list(net.parameters())
    optimizer = optim.Adam(net.parameters(), lr=0.0002)
    #定义一个Adam优化器，优化对象是所有可训练的参数，定义学习率为0.0002，通过调试获得的最佳学习率

    epochs = 500##############################################################################################################################
    save_path = './AlexNet.pth'#保存准确率最高的那次模型的路径
    best_acc = 0.0#最佳准确率
    train_steps = len(train_loader)
    for epoch in range(epochs):
        # train
        net.train()#使用net.train()方法，该方法中有dropout
        running_loss = 0.0#使用running_loss方法统计训练过程中的平均损失
        train_bar = tqdm(train_loader)

        for step, data in enumerate(train_bar):#遍历数据集
            images, labels = data#将数据分为图像标签
            optimizer.zero_grad()#清空之前的梯度信息
            outputs = net(images.to(device))#通过正向传播的到输出
            loss = loss_function(outputs, labels.to(device))#指定设备gpu或者cpu,通过Loss_function函数计算预测值与真实值之间的差距
            loss.backward()#将损失反向传播到每一个节点
            optimizer.step()#通过optimizer更新每一个参数

            # print statistics
            running_loss += loss.item()#累加损失
            #print train process
            rate = (step+1)/len(train_loader)
            a = "*"* int(rate*50)
            b = "."* int((1-rate)*50)
            print("\rtrain loss:{:^3.0f}%[{}->{}]{:.3f}".format(int(rate*100),a,b,loss),end="")
        print()


        # validate
        net.eval()#预测过程中使用net.eval()函数，该函数会关闭掉dropout
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():#使用该函数，禁止pytorch对参数进行跟踪，即训练过程中不会计算损失梯度
            val_bar = tqdm(validate_loader)
            for val_data in val_bar:#遍历验证集
                val_images, val_labels = val_data#将数据划分为图片和标签
                outputs = net(val_images.to(device))
                predict_y = torch.max(outputs, dim=1)[1]#求的预测过程中最有可能的标签
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()#准确的个数累加

        val_accurate = acc / val_num#测试集准确率
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        # 如果当前准确率大于历史最优准确率，就将当前的准确率赋给最优准确率，并将参数进行保存
        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(net.state_dict(), save_path)

    print('Finished Training')
    print(best_acc)


if __name__ == '__main__':
    main() 
