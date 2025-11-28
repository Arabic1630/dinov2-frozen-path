from dinov2.data.datasets import Pathology

for split in Pathology.Split:
    dataset = Pathology(split=split, root="./dataset/pathology-tcga-224/", extra="./dataset/pathology-tcga-224/",sshid="13")
    dataset.dump_extra()