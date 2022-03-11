<font size="7"> Readme for the notebooks part</font><br><br>
The aim of this readme is to quickly describe the diffrent notebooks there are in this section and what their aim are. <br><br>

# Data vizualisation

The first important notebook is **Data visualisation** which is a notebook that helps visualising the repartition of features from SCOR dataset, and the correlation between some features.<br><br>

# Models

There are several notebooks that gathers some of our ideas to implement models to answer to the problem which was given by SCOR.<br><br>
The first approach we used was to implement some baseline models to give a really straightforward approach and an overview of possibles results.<br><br>
The main notebook (in which we implemented the deep embedding model) is MLP Embedding model, and is administrative free.<br>
The same model but with a statewise view is in the MLP Embedding model state.<br><br>
We also fetched this model with external dataset (the notebooks are not totally clean though), but it might work (results for VDSA datasets are very close to these without external datasets concerning regression model, for satellite images, since the dataset is very huge and difficult to integrate, results are not good, a more proper fetch needs to be done).

# External dataset

The notebook to download VDSI images is import satellite images. You will need to have a earth engine account to run it, cf this [link](https://earthengine.google.com/new_signup/).<br><br>
The notebook to reduce images shapes thanks to AutoEncoder is vegetation images AE (the same idea could be reused for other images datasets).<br>
You could also find notebooks to import other images datasets in **./images_experimental** folder
