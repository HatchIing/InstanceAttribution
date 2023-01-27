# Comparing Instance Attribution to k-Nearest Neighbors

This is the code for the paper A Comparison of Instance Attribution Methods. All third party code is publicly available 
except for the pre-trained ExPred model, contact Avishek Anand for more information. 

## Usage:
  1. Pull FastIF, TracIn, and the pre-trained ExPred model, and import into designated files folders.
  2. From FastIF only ```experiments``` and ```influence_utils``` are required. Use the included ```nn_influence_utils.py```. Minor changes to other files may be necessary.
3. In ```src/TracIn``` put the files alongside main_tracin.py
4. Install the required packages from the ```requirements.txt``` by ```pip install -r requirements.txt```
5. In ```main.py``` adjust constants as needed and run.
     
Not that depending on your hardware you may have to change the `k` in `main.py`.

### Structure of the repo
```
├── dataset: the eraser FEVER datasets and custom dataset utils
├── expred: the pretrained expred model (not publicly available)
├── fastif: fastif from https://github.com/salesforce/fast-influence-functions
├── README.md: this file
├── tracin: tracin from https://github.com/frederick0329/TracIn
└── main.py: main execution file
```