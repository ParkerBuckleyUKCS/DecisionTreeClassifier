# Decision Tree Classifier 
---

## Overview
This machine learning algorithm manually classifies pokemon as legendary or not based on their primary stats.
Synthetic datasets for training, as well as the working dataset, are included under the "Project Files" folder.

Upon execution, this program will load information from one of the files and generate a decision tree using the IC3 algorithm. Metrics such as ***entropy*** and ***information gain*** are calculated on the data to split on the most significant feature first. This priority spitting makes the decision tree more accurate. More on this below...

## The Data
***Numerical data that is processed into nominal data using equidistant bins.***

***Synthetic Data - Cartesian Coordinates with Two class labels (red, blue)***
1. Synthetic-1.csv - Two clusters split on X axis 
2. Synthetiv-2.csv - Two clusters with some overlap on the X axis
3. Synthetic-3.csv - Two clusters merged on the X and Y axis
4. Synthetic-4.csv - Two disks, one inside of the other.

***Pokemon Data - 44 separate feature including Total level, HP, Attack, Def, and boolean values for each Pokemon type. (water, fire, leaf, etc...)***

## Visualization of the Decision Boundary

Below are some of the visualizations generated. The scatter plot represents the data set, where the color of the dot is the class label.
The rectangles in the background are the bins created by the decision tree. The rectangles are colored according to the prediction made by the tree.

### Synthetic 1
![Synthetic-1.csv](/Screenshots/decisionSurface_synthetic1.JPG)
### Synthetic 2
![Synthetic-2.csv](/Screenshots/decisionSurface_10Bins_synthetic2.JPG)
### Synthetic 3
![Synthetic-3.csv](/Screenshots/decisionSurface_20Bins_synthetic3.JPG)
### Synthetic 4
![Synthetic-4.csv](/Screenshots/decisionSurface_10Bins_synthetic3.JPG)

## Results

The error on the synthetic datasets were acceptable. Here are the calculations:
1. synthetic-1.csv : ***100%***
2. synthetic-2.csv : ***98.5%***
3. synthetic-3.csv: ***95%***
4. synthetic-4.csv: ***96.5%***

Error on the pokemon set was reasonable as well at 90.6% with 16 bins.

At an overview, it is possible to achieve 100% accuracy on a set with at least two unique data entries, one just has to choose a large enough number of bins.
***This comes at a cost.*** The nature of the decision tree is to memorize the training data, It is easy for the model to be overfitted.

We can fine tune how closely a tree reflects it's training data by increasing or decreasing the number of bins. 
- Increasing will yield higher accuracy on the training set but potentially lower accurracy on future data sets. 
- Decreasing will result in quite the opposite- we will lose accuracy on the training data but, as a result, be more lenient with future data sets. 

Therefore, choosing the number of bins is often the crux of data tree fitting and methods such as ***cross-validation*** help find the optimal choice of bins.
