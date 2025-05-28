# Detection-of-Tools-in-Laparoscopic-Surgery
*Detection of Tools in Laparoscopic Surgery to Support the Operation of Surgical Robots using YOLOv8 and ByteTrack*


My name is Eva and this is the repository of my Master Thesis in Medical Computing Image at the University of Girona.

I have used the public datasets of **CholecTrack20**, **Cholec80**, **M2CAI16** and **SurgToolLoc** *(challenges 2022 and 2023)*.

The code is divided into different phases:
1. **Preprocessing**: preprocesses the datasets with specific rules for frame and label extraction, and for CholecTrack20 I have just used 'Training' and 'Validation' images. 
2. **Training**: train the **YOLOv8** models in different sizes (s, l, x) with the CholecTrack20 dataset (contains real bounding boxes in addition to the classes).
3. **Evaluation with datasets**: 50 easy (1 tool) and 50 difficult (2 or more tools) images are randomly selected from the pre-processed Cholec80, M2CAI16 and SurgToolLoc datasets.
4. **Comparison**: the 3 trained models are checked against the previous inference dataset, since they are images that the models have not seen, and then each model is catalogued with real bounding boxes, comparing the metrics.
5. **Tool Tracking**: tools are consistently tracked in time over a video using **ByteTrack**.
6. **Model improvement**: the best resulting model is refined, and compared again with the 3 trained models. The new model is re-trained with the weights obtained by the best resulting model, and with the added improvements, and validated with the same original dataset as a continuous reference.
7. **Validation with a different dataset**: the improved model, in addition to having been validated with the previous dataset, is re-validated with a dataset that it has not seen (in my case I have used the 'Testing' videos provided by CholecTrack20).
8. **Comparison of all the validations**: the metrics obtained from all the validations of the trained models are compared.
9. **Re-do point 4 and 5 with the new model**: the 4 trained models are checked with the inference dataset, and each model is catalogued (it must be taken into account that this dataset does not have real bounding boxes), and the tools are tracked with the improved model.


![cholec80_dificiles_video18_frame_09750](https://github.com/user-attachments/assets/4b2bc65d-d81e-4839-bbf5-9beaf5ada595)

*References*
- A.P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, N. Padoy. EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos. IEEE Trans. on Medical Imaging 2016.
- C.I. Nwoye, K. Elgohary, A. Srinivas, F. Zaid, J.L. Lavanchy, N. Padoy, CholecTrack20: A Multi-Perspective Tracking Dataset for Surgical Tools, Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2025.
- Aneeq Zia, Max Berniker, Rogerio Nespolo, Conor Perreault, Ziheng Wang, Benjamin Mueller, Ryan Schmidt, Kiran Bhattacharyya, Xi Liu, and Anthony Jarc. Surgical visual understanding (surgvu) dataset, 2025.
