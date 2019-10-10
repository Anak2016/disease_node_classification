import sys,os
USER = os.environ['USERPROFILE']
sys.path.insert(1,f'{USER}\\PycharmProjects\\my_utility')

from utility_code.my_utility import *
from utility_code.python_lib_essential import *

disgenet = pd.read_csv(r"C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\disgenet\all_gene_disease_pmid_associations.tsv\all_gene_disease_pmid_associations.tsv",
                sep = '\t')
# display2screen(disgenet[:5])
copd_label = pd.read_csv(r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\raw\copd_label07_14_19_46.txt',
                         sep = ',')
gene_id = copd_label['geneId']

# do it iteratively compare row wise if they all the same then

# disgenet_mask = disgenet.loc[copd_label['geneId'].isin(gene_id)]
# assert disgenet_mask.shape[0] == gene_id.shape[0], 'wrong'

# display2screen(np.unique(copd_label[['geneId']].values).shape)
# display2screen(np.unique(disgenet[['geneId','diseaseId']].values).shape)
tmp = disgenet[disgenet['geneId'].isin(copd_label['geneId'])] # unique
tmp1 = tmp[tmp['diseaseId'].isin(copd_label['diseaseId'])] # unique
tmp2 = tmp1[tmp1['pmid'].isin(copd_label['pmid'])]
# a,b = tmp2.iloc[0].copy(), tmp2.iloc[1].copy()
# tmp2.iloc[0], tmp2.iloc[1]= b,a
tmp2 = tmp2.loc[(tmp2['geneId'].isin(copd_label['geneId'])) & (tmp2['diseaseId'].isin(copd_label['diseaseId'])) ]

# print((tmp2[['geneId','diseaseId']].astype(str).apply(lambda x: x.str.strip()).eq(copd_label[['geneId','diseaseId']].astype(str).apply(lambda x: x.str.strip())).all(axis=None)))
# print(tmp2[['geneId','diseaseId']][-2:], copd_label[['geneId','diseaseId']][-2:])
# print(tmp2[['geneId','diseaseId']][-2:].eq(copd_label[['geneId','diseaseId']][-2:]))
# print(np.equal(tmp2[['geneId','diseaseId']][-2:].values, copd_label[['geneId','diseaseId']][-2:].values ))
ans = (np.equal(tmp2[["geneId","geneSymbol","diseaseId","diseaseClass","pmid","source"]].values, copd_label[["geneId","geneSymbol","diseaseId","diseaseClass","pmid","source"]].values ).all())
assert ans, "bad"
assert tmp1.shape[0] > copd_label.shape[0], 'very bad'
# print(np.equal(tmp2[["geneId","geneSymbol","diseaseId","diseaseName","diseaseClass","pmid","source"]].values, copd_label[["geneId","geneSymbol","diseaseId","diseaseName","diseaseClass","pmid","source"]].values ).all(axis=1))
tmp2.to_csv(r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\raw\copd_label10_06_19.txt', index=False)
tmp1.to_csv(r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\raw\copd_label10_06_19_added_edges.txt', index=False)


# tmp2.to_csv(r'C:\Users\awannaphasch2016\PycharmProjects\disease_node_classification\data\gene_disease\07_14_19_46\raw\copd_label10_06_2019.txt')

# tmp2 = tmp1[tmp1['source'].isin(copd_label['source'])]
# display2screen(tmp2[['geneId', 'diseaseId']].eq(copd_label[:-1]))
# display2screen(tmp2.eq(copd_label[:-1]))
# geneId,geneSymbol,diseaseId,diseaseName,diseaseClass,pmid,source,class

# display2screen(tmp2[['geneId', 'diseaseId','pmid']][:4], copd_label[['geneId', 'diseaseId','pmid']][:4])
# display2screen(tmp2[['geneId', 'diseaseId','pmid']][:4].eq(copd_label[['geneId', 'diseaseId','pmid']][:4]))
# for each row in copd_label: check geneId_disease_id and pmid if they all the same select the whole row of disgenet

# display2screen(tmp2[['geneId', 'diseaseId','pmid']].dtypes, copd_label[['geneId', 'diseaseId','pmid']].dtypes)

