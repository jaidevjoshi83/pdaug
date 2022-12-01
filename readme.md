# PDAUG - a Galaxy based toolset for peptide library analysis, visualization, and machine learning modeling.

## Overview 

Peptide Design and Analysis Under Galaxy (PDAUG) package, a Galaxy based python powered collection of tools, workflows, and datasets for a rapid in-silico peptide library analysis. PDAUG offers tools for peptide library generation, data visualization, in-built and public database based peptide sequence retrieval, peptide feature calculation, and machine learning modeling. PDAUG tool suite can be downloaded and install through galaxy toolshed as a standard galaxy tool. 

Galaxy Tutorials: Check our Galaxy tutorials for more details

  -- [Peptide Library Data Analysis] (https://training.galaxyproject.org/training-material/topics/proteomics/tutorials/peptide-library-data-analysis/tutorial.html)
  -- [ML Modeling Of Anti-Cancer Peptides] (https://training.galaxyproject.org/training-material/topics/proteomics/tutorials/ml-modeling-of-anti-cancer-peptides/tutorial.html)


# Prebuild Docker Image 

## A prebuild build docker image based on the recent galaxy release can be obtained by the link below for a quick installation. 
 
 - Command to run PDAUG-Galaxy server on docker
 
    `docker run -i -t -p 8080:80 jayadevjoshi12/galaxy_pdaug:latest`
  
 - Server is accessible at 
   
    `http://127.0.0.1:8080`
 
 - Galaxy's default user login details
  
     *user id:* admin
     *passowrd:* password
 
 
## For more details about the latest docker image, check the bellow link...

 - [Docker Image](https://github.com/jaidevjoshi83/docker_pdaug)

 *Note* To install docker on windows machine always run installer as an adminstrator by right clicking on the installer incon. 

# Contributors
 - Jayadev Joshi
 
 - Daniel Blankenberg

# History

 - 0.1.0: First release!

# Support & Bug Reports

You can file an [github issue](https://github.com/jaidevjoshi83/docker_pdaug/issues). 
