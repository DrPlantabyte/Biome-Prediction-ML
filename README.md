# Making a Biome Prediction Node Network Model
In this project, my aim is to use Python machine learning libraries to create a node network model for predicting biome classification based on a limited set of environmental parameters, using freely available satellite data to train the model.

## What is a biome?
Functionally speaking, a biome is the habitat created by the plants and geology in a given region. Forest, grassland, desert, and deep ocean are all examples of a biome. While there are many different ways of defining biomes, for this project I will be the International Geosphere-Biosphere Programme (IGBP) land cover classification system, which has 17 biome classifications:

1. Evergreen needleleaf forest
2. Evergreen broadleaf forest
3. Deciduous needleleaf forest
4. Deciduous broadleaf forest
5. Mixed forest
6. Closed shrubland
7. Open shrubland
8. Woody savanna
9. Savanna
10. Grassland
11. Permanent wetland
12. Cropland
13. Urban and built-up landscape
14. Cropland/natural vegetation mosaics
15. Snow and ice
16. Barren
17. Water bodies

## What is a node network model?
Similar to biological brains, node network machine learning models (aka artificial neural networks) are made of interconnected pieces, called nodes, which each perform a small calculation that is then passed on to another node. By playing with connections between the nodes, the machine learning algroithm is able to evolve the web of nodes from a useless heap into a decision tree that outputs a prediction in it's output node based on the states of the input nodes.

One should not get too carried away by the comparison of node networks and biological brains. While the high-level principle is the same, biological neurons perform different kinds of calculations than machine learning nodes, and a typical animal brain has millions to trillions of neurons while most machine learning node networks are limited to a few hundred or a few thousand nodes. 

## What will be the inputs and outpts?
The existing land cover maps already provide biome classification for any location on Earth. However, this is not very useful for hypotheticals, such as what will the landscape look like after another century of global warming? Or what might it have looked like in the past? Or what should the biome maps of Westeros or Middle Earth look like?

Therefore my prediction model will only use easily predictable inputs: average yearly min and max temperature and total annual rainfall amount and standard deviation (to capture seasonality of rainfall). The input data will come from freely available satellite remote sensing data products.

## The game plan
My strategy for building this biome prediction model is as follows:
1. Download land cover classification, temperature, and rainfall data from NASA 
2. Clean the data to put everything on the same scale and omit locations with missing data
3. Train a classification node network, using 80% of data pixels for training and 20% for validation
4. Use the trained model to make a graph of biome classification as a function of min/max temp & rainfall

# Step 0: Setup

First thing I do is grab my handy-dandy Python virtual environment bash script:

```bash
#!/bin/bash
if [ "$#" -ne 2 ]; then
    echo "Wrong arguments. Usage:"
    echo "$0 venv_name requirements_file"
    exit 1
fi

VENV_NAME="$1"
REQ_FILE="$2"

if test ! -d "$VENV_NAME"; then
	python3 -m venv "$VENV_NAME"
	echo "*" > "$VENV_NAME"/.gitignore
	source "$VENV_NAME"/bin/activate
	pip install --upgrade pip
	pip install -r "$REQ_FILE"
	pip freeze > "$REQ_FILE"/requirements.txt
	deactivate
fi
```

And then I run it to make a python virtual environment with the following `requirements.txt` file:
```
numpy
requests
scipy
scikit-learn
matplotlib
keras
tensorflow
GDAL
```

Note that getting the GDAL library to work has many additional steps, which I will not cover here. To anyone else tasked with setting up GDAL with python, you have my sympathies.

After installation is complete, I create a PyCharm project and get to work on step 1: downloading the data.



