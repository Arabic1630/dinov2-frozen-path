# Low-resource finetuning of foundation models beats state-of-the-art in histopathology

This is the repository of  [Low-resource finetuning of foundation models beats state-of-the-art in histopathology](https://arxiv.org/abs/2401.04720) which was accepted at ISBI 2024.微调后的 DINOv2 ViT-S 模型


## Use the pipeline[跟dinov3的几乎一样]

```
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=${PWD} torchrun --standalone --nnodes=1 --nproc-per-node=4 dinov2/train/train.py --config-file dinov2/configs/train/vitl16-tcga-dino.yaml --output-dir ./results/vitl16-all-224-UNI-按照dinov3同样超参/ train.dataset_path=Pathology:split=TRAIN:root=/data/tanyuyi/code/dinov3-frozen-path/dataset/pathology-tcga-224/:extra=/data/tanyuyi/code/dinov3-frozen-path/dataset/pathology-tcga-224/:sshid=13
```





# Model farm
We make all models as well as heads used for training publicly available in the following.

## Pretrained models finetuned on NCT-CRC-100K

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>CRC-VAL-HE-7K<br />20-NN balanced acc</th>
      <th>CRC-VAL-HE-7K<br />linear balanced acc</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">2k</td>
      <td align="right">93.8%</td>
      <td align="right">92.7%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">10k</td>
      <td align="right">93.4%</td>
      <td align="right">93.7%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
  </tbody>
</table>

## Pretrained models finetuned on TCGA

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th># of<br />params</th>
      <th># of<br />iterations</th>
      <th>TCGA<br />AUROC</th>
      <th>CPTAC<br />AUROC</th>
      <th>teacher backbone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td align="right">21 M</td>
      <td align="right">30k</td>
      <td align="right">89%</td>
      <td align="right">85%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td align="right">1,100 M</td>
      <td align="right">60k</td>
      <td align="right">84%</td>
      <td align="right">79%</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_teacher_checkpoint.pth?download=1">teacher weights</a></td>
    </tr>
  </tbody>
</table>



## Installation

```python
conda env create -f conda.yaml
conda activate dinov2
```




## Continue finetuning

If you want to continue finetuning or use the DINO heads, the remaining weights can be found here:

<table style="margin: auto">
  <thead>
    <tr>
      <th>model</th>
      <th>dataset</th>
      <th># of<br />iterations</th>
      <th>student backbone</th>
      <th>student DINO head</th>
      <th>teacher DINO head</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>ViT-S/14</td>
      <td>NCT-CRC-100K</td>
      <td align="right">2k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_NCT_10k_training_1999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>NCT-CRC-100K</td>
      <td align="right">10k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_NCT_training_9999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-S/14</td>
      <td>TCGA</td>
      <td align="right">30k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vits_TCGA_training_29999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
    <tr>
      <td>ViT-g/14</td>
      <td>TCGA</td>
      <td align="right">60k</td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_student_checkpoint.pth?download=1">student backbone</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_student_dino_head_checkpoint.pth?download=1">student DINO head</a></td>
      <td><a href="https://zenodo.org/records/10406135/files/dinov2_vitg_TCGA_training_59999_teacher_dino_head_checkpoint.pth?download=1">teacher DINO head</a></td>
    </tr>
  </tbody>
</table>

To load these weights, it is enough to add the path to the config file under head_path. The path that has to be added is to a folder containing the weights. The weights have to be renamed after downloading them for the available code to work (e.g. student_dino_head_checkpoint.pth). More details can be found in the file /dinov2/dinov2/train/ssl_meta_arch.py.

