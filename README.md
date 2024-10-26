# ChestX-AI
It's a machine learning project for image captioning on radiological Chest X-ray images. NLP is used for captioning, image processing and CNN used for feature extraction of images.  

<p align="center">
 <img src="images/chestx-ai-demo.gif" alt="ChestX-AI Demo.gif">
</p>
<p align="center">
 <b>Figure 1. ChestX-AI Demo</b></p>

## Technologies 

<p align="center">
 <img src="images/tech_stack.png" alt="technology_stack.png">
</p>
<p align="center">
 <b>Figure 2. Technology Stack</b></p>

# Setup & Deployment

Model files are big so not uplodaded in this repo. However you can download these from <a href="https://drive.google.com/drive/folders/1-Ob7w1VYhxYyGfrnaR2Jrc5wkqvdisFt?usp=sharing">this link</a>
Besides, there is no need to download these for dockerize setup.

## Deployment
- First usage: ` docker-compose up -d --build` 
- Already build: ` docker-compose up -d `
- For shutdown: ` docker-compose down`
- Voila! Now you can use your product in <a href="http://localhost:8030">your localhost http://localhost:8030 .</a>

# To-Do:

- <s>Dockerize version</s> 
- <s>API connection to retrieve models</s> 
- Evaluating model & hyperparameter-finetuning
- model finetuning : Attention based model
- <s>select models checkbox/combobox etc. on ui</s>
- <s>add model downloads script or codes into dockerfiles</s>
- Create a PDF like a radiologist report. (Some textboxts will add to UI about patient)

# Dataset 

## What is open-i?
Open-i service of the National Library of Medicine enables search and retrieval of abstracts and images (including charts, graphs, clinical images, etc.) from the open source literature, and biomedical image collections. Searching may be done using text queries as well as query images. Open-i provides access to over 3.7 million images from about 1.2 million PubMed Central® articles; 7,470 chest x-rays with 3,955 radiology reports; 67,517 images from NLM History of Medicine collection; and 2,064 orthopedic illustrations.

>Reuse of Open-i images is determined by the license type of the image. A link to the applicable license type, if available, may be found below the individual Open-i image on the detailed view page.
>Please refer to the following web page for information on the PMC open access subset and the PMC Copyright Notice:
>http://www.ncbi.nlm.nih.gov/pmc/tools/openftlist/

- Dataset Article <a href="https://pubmed.ncbi.nlm.nih.gov/26133894/">link</a>.
- You can get more information about dataset from <a href="https://openi.nlm.nih.gov/faq#collection">this link</a> .
- You can download images from <a href="https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz">this link</a> .
- You can download image reports from <a href="https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz">this link</a> .

