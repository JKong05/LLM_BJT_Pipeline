# SeamlessExpressive
In order to access the models from Meta and run this pipeline, the models should be downloaded and accessed 
via the [Seamless Communication GitHub](https://github.com/facebookresearch/seamless_communication/tree/main?tab=readme-ov-file) 
following the instructions within the README. These models are on a request-basis, which can be accessed [here](https://ai.meta.com/resources/models-and-libraries/seamless-downloads/),
and more information can be found about SeamlessExpressive [here](https://github.com/facebookresearch/seamless_communication/blob/main/docs/expressive/README.md).

## Instructions
1. Request and download PRETTSSEL and m2m models
2. Place the .pt files within the `content` directory
3. Run application to initialize models for pipeline

```
# example folder structure
content/
│   ├── m2m_expressive_unity.pt
│   ├── pretssel_melhifigan_wm-16khz.pt
│   └── pretssel_melhifigan_wm.pt
```

## Citation
```
@inproceedings{seamless2023,
   title="Seamless: Multilingual Expressive and Streaming Speech Translation",
   author="{Seamless Communication}, Lo{\"i}c Barrault, Yu-An Chung, Mariano Coria Meglioli, David Dale, Ning Dong, Mark Duppenthaler, Paul-Ambroise Duquenne, Brian Ellis, Hady Elsahar, Justin Haaheim, John Hoffman, Min-Jae Hwang, Hirofumi Inaguma, Christopher Klaiber, Ilia Kulikov, Pengwei Li, Daniel Licht, Jean Maillard, Ruslan Mavlyutov, Alice Rakotoarison, Kaushik Ram Sadagopan, Abinesh Ramakrishnan, Tuan Tran, Guillaume Wenzek, Yilin Yang, Ethan Ye, Ivan Evtimov, Pierre Fernandez, Cynthia Gao, Prangthip Hansanti, Elahe Kalbassi, Amanda Kallet, Artyom Kozhevnikov, Gabriel Mejia, Robin San Roman, Christophe Touret, Corinne Wong, Carleigh Wood, Bokai Yu, Pierre Andrews, Can Balioglu, Peng-Jen Chen, Marta R. Costa-juss{\`a}, Maha Elbayad, Hongyu Gong, Francisco Guzm{\'a}n, Kevin Heffernan, Somya Jain, Justine Kao, Ann Lee, Xutai Ma, Alex Mourachko, Benjamin Peloquin, Juan Pino, Sravya Popuri, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Anna Sun, Paden Tomasello, Changhan Wang, Jeff Wang, Skyler Wang, Mary Williamson",
  journal={ArXiv},
  year={2023}
}
```
