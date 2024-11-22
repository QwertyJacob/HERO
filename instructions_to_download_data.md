# The raw-traffic observations used in HERO
Can be obtained from downloading this file and unzipping it. 

https://drive.google.com/drive/folders/1uZDm1vR2drxQyd_aH6wHQfAJ46xDJbnA?usp=sharing


## Note:
The _PcapPlusPlus_ (1) library was used to split traffic into connection flows, and then the _heiFIP_ (2) library  helped us to encode the first 512 bytes of the first 512 packets of each flow as bi-dimensional 512x512 images for neural convenience. Sender and receiver IP addresses and ports were masked in every packet to avoid biasing the automatic feature extraction of the pipeline.

The raw IoT traffic captures used in the experiments are extracted from the attack traces offered in the _BoT-IoT_ (3) dataset and the _Industrial IoT_ (4) dataset. Such data were collected on realistic cyber-ranches composed of IoT devices interchanging benign and attack traffic.

(1) https://pcapplusplus.github.io/
(2) https://github.com/stefanDeveloper/heiFIP
(3) Koroniotis, Nickolaos, Nour Moustafa, Elena Sitnikova, and Benjamin Turnbull. "Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset." Future Generation Computer Systems 100 (2019): 779-796 (https://arxiv.org/abs/1811.00701)
(4) Mohamed Amine Ferrag, Othmane Friha, Djallel Hamouda, Leandros Maglaras, Helge Janicke, January 17, 2022, "Edge-IIoTset: A New Comprehensive Realistic Cyber Security Dataset of IoT and IIoT Applications: Centralized and Federated Learning", IEEE Dataport, doi: https://dx.doi.org/10.21227/mbc1-1h68.

