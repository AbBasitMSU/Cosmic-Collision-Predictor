<h1 align="center">Michigan State University<br/>
CMSE-830 Class Project<br/></h1>

--------------------------------

<h2 align="center"><br/>
Cosmic Collision Prediction using<br/>
Supervised Learning and Neural Network Models<br/>
<br/>
By Ab Basit Rafi, Syed</h2><br/>

![ReadmeImage_wText](https://github.com/maparisio/Asteroid-Impact-Prediction/assets/152572519/dd80ab66-bf53-44cd-a0a8-c50a6d454656)

Note: It is important that if you are going to use this code, all files
are placed in a directory that matches this repository, for the better functionality of it.
Otherwise you would have to adjust the paths on the files, accordingly.

This repository consists of a team project where we explore the predictability of asteroid impacts using machine learning models.

---------------------------------
INDEX
---------------------------------
1. Content of the repository
2. Instructions for the Project
3. References

---------------------------------
Content of the repository
---------------------------------
- Original_Datasets:
  - impacts.csv
  - orbits.csv
- h5_Files
  - Asteroid_Impact_Model.h5
  - Asteroid_Impact_Optimization_Model.h5
- Asteroid definitions.pptx <-- Powerpoint presentation Intro to the project and definitions of the different columns in the dataset.
- Asteroid_Predictions.ipynb <-- File started in Jupyter Notebook for data cleanup prior to Neural Network ML training
- Asteroid_Predictions_Colab.ipynb <-- File worked on via Google Colab after cleanup, to train our Neural Network ML model prior to optimization.
- Asteroid_Predictions_Optimization_Colab.ipynb <-- File worked on via Google Colab. Optimized version after training our Neural Network ML model.
- asteroid-impact-prediction-SL-CFM.ipynb <-- File worked on via Jupyter Notebook for Supervised Learning, with unbalanced data.
- asteroid_impact-prediction-SL-OverSample.ipynb <-- File worked on via Jupyter Notebook for SL, with OverSampling of the data.
- asteroid-impact-prediction-SL-UnderSample.ipynb <-- File worked on via Jupyter Notebook for SL, with UnderSampling of the data.
- cleaned_Asteroid_orbit.csv <-- csv file created via Jupyter Notebook after data was cleaned prior to creating the ML models (NN model version)

----------------------------------
Guide to the Project
----------------------------------

### Guidelines for the Project
1.  Collaborating with our team to pool knowledge and share ideas
2.  Outline a scope and purpose for our project, utilzing our machine learning skills to analyze,solve, or visualize our findings
3.  Finding reliable data to use for our project, being mindful of copyrights, licenses, or terms of use
4.  Track all processes in Jupyter Notebook used for cleanup, and techniques used for Data Analysis
5.  Present our findings to the class on Presentation Day, with each member of our group taking turns in speaking
6.  Submit the URL of our GitHub repository to be graded
7.  Graduate and attain employment from utilizing our knowlwdge acquired from this class
   
### Processes used 
1.  Reading the csv files
2.  Cleaning the data
3.  Normalize and stabalize the data
4.  Splitting the data
5.  Training the Machine Learning models
6.  Neural Network model implementation
7.  Created a different Jupyter notebook with the same cleanup process to test Supervised Learning model
8.  Supervised Learning model implementation
9.  Confusion Matrix and Visualization
10. Compared observations and searched for improved accuracy for each model.

## Accuracy for the Neural Network Model (Pre-optimization and Optimized results)
![NN_model_AccuracyComparison](https://github.com/maparisio/Asteroid-Impact-Prediction/assets/152572519/7282e365-dd18-4676-8bd1-9dc1c155a53b)

## Accuracy for the Supervised Learning Model

1. Low precision and recall due to imbalance of data classes

![SL_model_Unbalanced](https://github.com/maparisio/Asteroid-Impact-Prediction/assets/152572519/413d931b-4ff6-4af2-8745-5bbcb371c069)

2. Results when over sampling the data
   
![SL_model_OverSampling](https://github.com/maparisio/Asteroid-Impact-Prediction/assets/152572519/7db96891-5cc7-4a24-861d-3bb123105835)

4. Results when under sampling the data

![SL_model_UnderSampling](https://github.com/maparisio/Asteroid-Impact-Prediction/assets/152572519/87f7baae-890d-4342-a06a-1f472f2e7068)


------------------------------------
References
------------------------------------

**References for the data source(s):**
- Datasets for this project: https://www.kaggle.com/datasets/nasa/asteroid-impacts

**References for the column definitions:**
- https://cneos.jpl.nasa.gov/about/neo_groups.html#:~:text=The%20vast%20majority%20of%20NEOs,%2Dmajor%20axes%20(%20a%20).
- https://howthingsfly.si.edu/ask-an-explainer/what-orbit-eccentricity
- https://en.wikipedia.org/wiki/Orbital_inclination
- https://astronomy.swin.edu.au/cosmos/A/Argument+Of+Perihelion
- https://cneos.jpl.nasa.gov/glossary/
- https://www.britannica.com/science/mean-anomaly
- https://en.wikipedia.org/wiki/Minimum_orbit_intersection_distance#:~:text=Minimum%20orbit%20intersection%20distance%20(MOID,collision%20risks%20between%20astronomical%20objects.

**References for code:**
- Uploading a CSV file to Google Colab:
  - https://stackoverflow.com/questions/60347596/uploading-csv-file-google-colab
- Using the strip() method for white spaces:
  - https://saturncloud.io/blog/how-to-remove-space-from-columns-in-pandas-a-data-scientists-guide/#:~:text=Using%20the%20str.strip()%20method&text=strip()%20method%20removes%20leading,column%20names%20or%20column%20values
- Confusion Matrix Visualization:
  - https://medium.com/@dtuk81/confusion-matrix-visualization-fc31e3f30fea
- Using Keras for Machine Learning:
  - https://machinelearningmastery.com/custom-metrics-deep-learning-keras-python/

- Learning Rate Scheduler:
  - https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
  - https://keras.io/api/callbacks/learning_rate_scheduler/
  - https://d2l.ai/chapter_optimization/lr-scheduler.html
  - https://stackoverflow.com/questions/61981929/how-to-change-the-learning-rate-based-on-the-previous-epoch-accuracy-using-keras
  - https://neptune.ai/blog/how-to-choose-a-learning-rate-scheduler

- Validation_Split function:
  - https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/
  - https://datascience.stackexchange.com/questions/38955/how-does-the-validation-split-parameter-of-keras-fit-function-work

- Activation Functions:
  - https://www.analyticsvidhya.com/blog/2020/01/fundamentals-deep-learning-activation-functions-when-to-use-them/

- Optimizers:
  - https://www.analyticsvidhya.com/blog/2021/10/a-comprehensive-guide-on-deep-learning-optimizers/
  - https://musstafa0804.medium.com/optimizers-in-deep-learning-7bf81fed78a0
  - https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6

- Callbacks:
  - https://www.kdnuggets.com/2019/08/keras-callbacks-explained-three-minutes.html
  - https://medium.com/@ompramod9921/callbacks-your-secret-weapon-in-machine-learning-b08ded5678f0
  - https://www.tensorflow.org/guide/keras/writing_your_own_callbacks

- Saving and Loading Models: 
  - https://colab.research.google.com/github/agungsantoso/deep-learning-v2-pytorch/blob/master/intro-to-pytorch/Part%206%20-%20Saving%20and%20Loading%20Models.ipynb
  - https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/keras/save_and_load.ipynb
  - https://stackoverflow.com/questions/64808087/how-do-i-save-files-from-google-colab-to-google-drive
  - https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory

**Image Resources:**
- ReadMe image was taken from:
  - https://pixabay.com/illustrations/dinosaurs-asteroid-stars-5568806/
    
- Introduction and Definition of Features in the DataSet Slide images
  -   https://pixabay.com/illustrations/asteroid-space-stars-meteor-1477065/
  -   https://pixabay.com/illustrations/armageddon-apocalypse-earth-2104385/
  -   https://en.wikipedia.org/wiki/Orbital_eccentricity
  -   https://www.sciencedirect.com/topics/physics-and-astronomy/true-anomaly
  -   https://www.researchgate.net/figure/Minimum-Orbital-Intersection-Distance_fig7_36174303
  -   https://pixabay.com/illustrations/asteroid-planet-land-space-span-4376113/

-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
