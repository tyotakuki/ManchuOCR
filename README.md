# Manchu Optical Character Recognition
## Manju Eldecingga Hergen -i Ilgame Takarangge

此Python脚本基于卷积神经网络和带有注意力机制的Seq2Seq模型，具体实现参考了此[Bai-Shang](https://github.com/bai-shang/crnn_seq2seq_ocr_pytorch)的GitHub仓库中的PyTorch实现。

# 数据集
* 训练用数据集在此处下载 [https://drive.google.com/drive/folders/1DZpBRpum4DZoB6gZ8VL8ejjzaj3HcihT?usp=sharing](https://drive.google.com/drive/folders/1DZpBRpum4DZoB6gZ8VL8ejjzaj3HcihT?usp=sharing)
* 130917词数据集在此处下载 [https://drive.google.com/drive/folders/1RkbzTk6jomvxzjw_0HrKXiTtUiB119zQ?usp=sharing](https://drive.google.com/drive/folders/1RkbzTk6jomvxzjw_0HrKXiTtUiB119zQ?usp=sharing)

# 安装
请直接下载GitHub仓库到本机文件夹（用```path-to-folder/```表示）。使用此脚本，需要安装3.5以上版本的Python和pip。需要一个可运行bash的类Unix环境。**请注意**：```model```文件夹中的预训练模型，应该单独下载，并且确保四个文件的md5正确：
- ```md5 encoder_0.pth  = af53b90ad122ecb7e6abf2c84ce356dd```
- ```md5 decoder_0.pth = 1be02bde5b70cb3cd72cd71a85fdfb1e```
- ```md5 encoder_1.pth = 0d822d8916c407f44d5ed56eb42eb31c```
- ```md5 decoder_1.pth =  610a3f009e88e6bea5d654ad4c752916```

此外，在Windows 10下，请安装Ubuntu子系统。参见[此链接](https://www.ssl.com/zh-CN/%E5%A6%82%E4%BD%95/%E5%90%AF%E7%94%A8linux%E5%AD%90%E7%B3%BB%E7%BB%9F%E5%AE%89%E8%A3%85ubuntu-Windows-10/)。

## Python3和pip的安装

* Python3的安装参见：[https://www.runoob.com/python3/python3-install.html](https://www.runoob.com/python3/python3-install.html)
* pip的安装参见：[https://www.runoob.com/w3cnote/python-pip-install-usage.html](https://www.runoob.com/w3cnote/python-pip-install-usage.html)

## Python模块的安装
此脚本需要的Python模块，在requirements.txt中写明。可以在bash中运行如下命令，安装所需模块：
```bash
pip3 install -r requirements.txt
```
# 使用

使用前，需要对含有**竖排**满文的图片进行如下处理：
* 去掉无关的汉字和符号标注。
* 去掉页面边框，确保页面周围不含有大段的直线，可以把满文部分单独截图出来。
* 适度增强图片的对比度和亮度。

若要查看符合标准的图片的示例，请参见examples文件夹中的相关.png文件。

然后，输入下面命令创建新文件夹：
```bash
cd path-to-folder
mkdir images
```
随后，可以手动把图片复制到image文件夹中。现在就可以运行下面的命令，来查看识别出来的文字：
```bash
python3 readmanchu.py --img_path ./image/文件名
```
输出结果如下格式：
```
Analyzing: ./examples/006.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

---识别出的文字---

Reading Completed, Press Any Key to Exit. Ambula Baniha.
```

并弹出圈出了识别出的满文单词的图片。

## 参数说明
* ```--img_path```参数：输入需识别满文图片的完整路径。
* ```--rot_angle```参数：如果需要对图片进行**微小的**旋转，请输入例如-2, -1, 1, 2的整数。
* ```--padding```参数：默认为10，正常情况下不需要调节。在5-10之间调整，可能会改变识别的效果。
* ```--block_size```和```--threshold```参数：```--block_size```默认为33（奇数），```--threshold```默认为32（偶数）。当特定像素的亮度大于```block_size```邻域内的平均值减去```threshold```时，将该像素设定为0，反之则设定为255。可以调整参数来查看识别效果，最佳的选择方案会带来一个黑底、白字的干净图片。
* ``--vertical_minimum```参数：该纵列亮度值之和大于多少，才会被识别为一（满文的）行，默认为800.
* ```--word_minimum```参数：该（满文的）行水平方向亮度值之和大于多少，才会被识别为一个单词，默认为200.
* ```--blur```参数：有True和False两个选项。活字印刷的文件，可能字母之间会出现小的空白。通过设置```--blur True```可以避免该空白被识别为单词之间的空格。
* ```--pretrained```参数：目前有0和1两个选择，0使用75万个数据训练，1使用12万个数据训练，然而识别效果在不同的情况下好坏不一。

# 例子
## 《少年中国说》现代印刷版
![001](https://raw.githubusercontent.com/tyotakuki/ManchuOCR/main/examples/001.png)

```bash
python3 readmanchu.py --img_path ./examples/001.png

Analyzing: ./examples/001.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

FE BE TUWAKIYAMBI EREHUNJEHEI TUTTU IBERE BE BAIMBI FE BE TUWAKIYAHAI 

TUTTU ENTEHEME FE OMBI IBERE BE BAIHAI TUTTU INENGGIDARI ICEMLEMBI 

DULEKENGGE BE MERKIHEI YAYA BAITA GEMU BEYEI DULEMBUHENGGE OFI UTTU DAMU 

KOOLI BE SONGKOLOME YABURE BE SAMBI JIDERENGGE BE SEOLEHEI EITEN BAITA 

YONI DULEMBURE UNDENGGE OFI UTTU KEMUNI GELHUN AKŪ KEMUN CI TUCINEMBI 

SAKDASA DARUHAI JOBOŠOME GŪNIMBI ASIHATA KEMUNI SEBJELERE BEI BUYEMBI 

JOBOŠOHOI TUTTU GŪNIN MUSEMBI SEBJELEHEI TUTTU SUKDUN YENDEMBI GŪNIN 

MUSEKEI TUNTU LEBDEREMBI SUKDUN YENDEHEI TUNTU EKTERŠEMBI LEBDEREKI 

TUNTU DULEMŠEMBI EKTERŠEHEI TUNTU HAKSATAMBI DULEMŠEHEI TUNTU JALAN 

JECEN BE GUKUBUMBI HAKSATAHAI TUTTU JALAN JECEN BE BANJINABUMBI SAKDASA 

DARUHAI BAITA BE EIMEMBI ASIHATA KEMUNI BAITA BE CIHALAMBI BAITA BE 

EIMEHEI TUTTU DARUHAI EITEN BAITA BE GEMU YABUCI OJORO BAI AKŪ SAME 

SEREMBI BAITA BEI CIHALAHAI TUTTU KEMUNI EITEN BAITA BEI YOONI YABUCI 

OJORAKŪ BI AKŪ SEME SEREMBI SAKDASA YAMJISHŪN ŠUN ADALI ASIHATA 

ERDE ŠUN GESE SAKDASA MACUHA IHAN ADALI ASIHATA ŠURGAN TASHA 

GESE SAKDASA HŪWAŠAN ADALI ASIHATA KANGSANGGE GESE SAKDASA BULEKU 

Reading Completed, Press Any Key to Exit. Ambula Baniha.

```

## 《满文老档》圈点版
![002](https://raw.githubusercontent.com/tyotakuki/ManchuOCR/main/examples/002.png)

```bash
python3 readmanchu.py --img_path ./examples/002.png
Analyzing: ./examples/002.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

TONIKI FUKA SINDAHA HERGEN I DANGSE 

TU SURE HAN I SUCUNGGA FULAHŪN GŪLMAHŪN ENIYA ANIYA 

BIJAI ICE INENGI GEREN BEISE EMBUSA BITHEI COOHAI 

HAFASA SUNJACI GING WAJIME AMPA YAMUN DE ISAFI 

MENI MENI GŪS GŪSAI FAIDAHA ABKA GEREME 

SURE HAN GEREN BEISE AMBASA BEI GAIFI 

TANGSE DE GENEFI 

Reading Completed, Press Any Key to Exit. Ambula Baniha.
```

![003](https://raw.githubusercontent.com/tyotakuki/ManchuOCR/main/examples/003.png)

```bash
python3 readmanchu.py --img_path ./examples/003.png
Analyzing: ./examples/003.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

ABKA TE ILAN JERGI NIYAKŪRAFI UJUN JERGI HENGKILEHE TERECI BEDEREFI 

HAN YAMUN DE TUCI TEHI MANGI GEREN BEISE AMBASA 

MENI MENI GŪSA GŪSAI JERGI BODOME ILATA 

JERGI NIYAKŪRAFI UJUTE JERGI HENGKILEHE TERE HENGKILERE DE 

HAN I JUWE ASHAN DE JUWE NIYALMA ILIFI EMU NIJALMA 

TENTEKI BEILE TENTEI AMBAN ANIYA SE BAHA 

SENME GEREN BI GAIFI HENGKILEMBI SENE HŪLAHA 

Reading Completed, Press Any Key to Exit. Ambula Baniha.
```

## 《玉匣记》
![005](https://raw.githubusercontent.com/tyotakuki/ManchuOCR/main/examples/005.png)


```bash
python3 readmanchu.py --img_path ./examples/005.png --pretrained 0 --rot_angle -1 --block_size 55 --threshold 54 --word_minimum 600
Analyzing: ./examples/005.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

BU EOI HIYAGI BIFAHABI UJE I DEBTELI I ULU 

FUCIHI ENDURI BANJIHA INENGGI 

YUYASAJAN GENGNI U SINGPI GING 

DUI TUNG FIRUSENDUUHAKŪNGGE 

JU G'O KUNG MING NI TUJAFI YABURE BE TUWARANGGE 

TUJINI YABUREININA JUWE ERIN BEI TUWARANGGEI 

BI IOFI GINGNI KITHETELEBUCI YABUCI OJORAKŪN INENGGI BE TUWARANGGE 

TU 

Reading Completed, Press Any Key to Exit. Ambula Baniha.
```

## 《异域录》

![007](https://raw.githubusercontent.com/tyotakuki/ManchuOCR/main/examples/007.png)


```
python3 readmanchu.py --img_path ./examples/007.png
Analyzing: ./examples/007.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

JENGGUREKUNJERENE IRAMIN KASI BEI ALIFI CELAN HALAME 

GUJUJE BOB I DEYENGGE UESIHUN BEI ELFIFI FUNGLU JETEIE 

FUNGNEHEN BEK ELIRE NACALA LANCAHAKŪNG BIHE BI LAHEJERAKŪNGGE 

FULAHŪN HONAIN ANAYA BANJIHA AJIGAN NCI BOBI BANJIRENGGE 

NADAHACINA BANI YADAINGGŪ BEJE NIMEKUNGGE ŠUTUME HŪWAKŪFI 

GANJU NIKAN BITHENGDUKE LJENGNACIBUHE GODO HADUK 

AKŪ UBANJABURENGGE ARŠARKI KOLAK BEK CADUME ANGSILAME 

Reading Completed, Press Any Key to Exit. Ambula Baniha.  
```

## 《御制避暑山庄诗》
![010](https://raw.githubusercontent.com/tyotakuki/ManchuOCR/main/examples/010.png)

```bash
python3 readmanchu.py --img_path ./examples/010.png --rot_angle -1 --threshold 12 --block_size 35 --vertical_minimum 3000
Analyzing: ./examples/010.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

JUGŪN GIKI HECEN DE HANCI T AMASI JULESI 

JUWE INENGGI BAIBURAKŪ NA SECI BIGAN 

HALI BUI BADARAMBUHANGGE GŪNIN DE TEBUCI E 

TUMEN BAITA UMAI TOKANURAKŪ TU JERECI DEN 

NECIN GORO HANCI I MURU BE KEMNEFI E INI CISUI 

BANJINAHA ALINGADAI ARBUŠUNEHINGGE JAKDAŠUN 

Reading Completed, Press Any Key to Exit. Ambula Baniha.
```

## 奏折一份

![018](https://raw.githubusercontent.com/tyotakuki/ManchuOCR/main/examples/018.png)

```bash
python3 readmanchu.py --img_path ./examples/018.png
Analyzing: ./examples/018.png
Using Möllendorff Alphabet List: ABCDEFGHIJKLMNOPQRSTUVWXYZŽŠŪ-'

AHASI ŠANGGANULEME 

WESIMBURENGGE 

HESE BI BAIREN JALIN JAKAN ILI I UHERK 

KADALARA DA FUJUHŪNGGA BOOLANJIHA BADA DULEKE ANIYA 

BOLORI NIOWANGGAJA TURUN KOWARAN COOHAI BARGIJAHA HACINGGA 

JEKU BEI NARHŪN JEKU SALIBUME BODOCI COOHA 

TOME JUWAN JAKŪTA HUSE FUNCEME GŪRBUMBU AFABUCI 

ANCARA JEKU BI GEMP HACIHAJEME TON I SONGKO 

CALU TE AFABUHA BIME DULEKEN ANIYA 

WESIMBUFI NOWANGGAJA TURUN KŪWARAN TE USE FAHA 

Reading Completed, Press Any Key to Exit. Ambula Baniha.
```