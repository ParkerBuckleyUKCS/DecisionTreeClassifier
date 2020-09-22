# Decision Tree Classifier 
---

##Overview
This machine learning algorithm manually classifies pokemon as legendary or not based on their primary stats.
Synthetic datasets for training, as well as the working dataset, are included under the "Project Files" folder.

Upon execution, this program will load information from one of the files and generate a decision tree using the IC3 algorithm. Metrics such as ***entropy*** and ***information gain*** are calculated on the data to split on the most significant feature first. This priority spitting makes the decision tree more accurate. More on this below...

##The Data
***Synthetic Data - Cartesian Coordinates with Two class labels (red, blue)***
1.	Synthetic-1.csv - Two clusters split on X axis 
2.	Synthetiv-2.csv - Two clusters with some overlap on the X axis
3. 	Synthetic-3.csv - Two clusters merged on the X and Y axis
4. 	Synthetic-4.csv - Two disks, one inside of the other.

##Visualization of the Decision Boundary

###Synthetic 1
![Synthetic-1.csv](/Screenshots/decisionSurface_synthetic1.JPG)
###Synthetic 2
![Synthetic-2.csv](/Screenshots/decisionSurface_10Bins_synthetic2.JPG)
###Synthetic 3
![Synthetic-3.csv](/Screenshots/decisionSurface_10Bins_synthetic3.JPG)
###Synthetic 4
![Synthetic-4.csv](/Screenshots/decisionSurface_10Bins_synthetic3.JPG)
