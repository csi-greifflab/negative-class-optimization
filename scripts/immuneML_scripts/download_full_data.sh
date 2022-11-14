#!/bin/bash
#Download full data
#folders were parsed via parse_index.ipynb
for i in 1ADQ_AAnalyses/ 1FBI_XAnalyses/ 1H0D_CAnalyses/ 1NSN_SAnalyses/ 1OB1_CAnalyses/ 1WEJ_FAnalyses/ 2YPV_AAnalyses/ 3RAJ_AAnalyses/ 3VRL_CAnalyses/ 5E94_GAnalyses/
do
    wget -r -np -R "index.html*" https://ns9999k.webs.sigma2.no/10.11582_2021.00063/projects/NS9603K/pprobert/AbsolutOnline/RawBindingsPerClassMurine/$i -P ../../data/full_data/divided2classes_archives
    # 1FBI_XAnalyses/ 1H0D_CAnalyses/ 1NSN_SAnalyses/ 1OB1_CAnalyses/ 1WEJ_FAnalyses/ 2YPV_AAnalyses/ 3RAJ_AAnalyses/ 3VRL_CAnalyses/ 5E94_GAnalyses/
done

#unzipping
for x in ../../data/full_data/divided2classes_archives/*/*.zip; do unzip "$x"; done