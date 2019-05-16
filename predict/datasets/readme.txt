##数据集说明##
"""
The rows of the dti matrix are represented as drug
The columns of the dti matrix are represented as target 
"""
*_admat_dgc -> drug-target interaction matrix
*_simmat_dc -> drug-drug interaction matrix
*_simmat_dg -> target-target interaction matrix


Note: 
"""
In order to facilitate the import of data, here the original data set name is simplified like below
"""

dc1: original
dc2: simmat_drugs_aers-freq
dc3: simmat_drugs_sider
dc4: simmat_drugs_spectrum

dg1: original
dg2: simmat_proteins_go
dg3: simmat_proteins_mismatch_n_k3m1
dg4: simmat_proteins_ppi

--------------------------------------------------------------
#e 描述
e_admat_dgc 药物标靶作用矩阵
e_simmat_dc{1-4} 药物相似矩阵 4核
e_simmat_dg{1-4} 标靶相似矩阵 4核

e_admat_dgc.shape -> (664,445) (target, drug)
e_simmat_dc{1-4}.shape -> (445,445)
e_simmat_dg{1-4}.shape -> (664,664)
--------------------------------------------------------------
#gpcr 描述
gpcr_admat_dgc 药物标靶作用矩阵
gpcr_simmat_dc{1-4} 药物相似矩阵 4核
gpcr_simmat_dg{1-4} 标靶相似矩阵 4核

gpcr_admat_dgc.shape -> (95, 223) (target,drug)
gpcr_simmat_dc{1-4}.shape -> (223,223)
gpcr_simmat_dg{1-4}.shape -> (95,95)
--------------------------------------------------------------
#ic 描述
ic_admat_dgc 药物标靶作用矩阵
ic_simmat_dc{1-4} 药物相似矩阵 4核
ic_simmat_dg{1-4} 标靶相似矩阵 4核

ic_admat_dgc.shape -> (204, 210) (target, drug)
ic_simmat_dc{1-4} -> (210, 210)
ic_simmat_dg{1-4} -> (204, 204)
--------------------------------------------------------------
#nr 描述
nr_admat_dgc 药物标靶作用矩阵
nr_simmat_dc{1-4} 药物相似矩阵 4核
nr_simmat_dg{1-4} 标靶相似矩阵 4核

nr_admat_dgc.shape -> (26, 54) (target, drug)
nr_simmat_dc{1-4}.shape -> (54, 54)
nr_simmat_dg{1-4}.shape -> (26, 26)
