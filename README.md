# Project_Imaging

This GitHub repository contains the complete code for the assignments and main project of the 8P361 - Project Imaging course for group 7.
Authors are, in alphabetical order, N. Hartog, J. Kleinveld, A. Masot, R. Pelsser.

Overview of the folders and their contents:
* **Assignment models**: This is where the models that have been trained during the assignments have been stored, so they can be retreived and used in stead of training the model over and over again.

* **Assignments code**: This is where the code for the assignments can be found, dividded in subfolders for each assignment

* **logs**: folder for the logs of the assignment models to be located when run.

* **Main project**: the code for the main project, divided in
    *  **Kaggle submissions**: containing the .csv for kaggle submission for each trained model
    *  **models**: all the trained models used for the experiments
    *  **tools**: .py files containing useful functions for the code. All of these are used as imports, and while keeping the github folder distribution they should run without issues. Includes:
   
         * **custom layers**: implemented custom layers based on literature and not available in keras for our models
         * **dim_reduction**: necessary functions for the visualisation of weights of the model
         * **GAN**: the definition of the generator and discriminator, and some other related functions
         * **transfer**: the necessary functions to apply transfer learning
         * **utils**: a compilation of functions for plotting, obtaining results, or other processes that needed to be done in order to keep the main code clear.
    * **classifier_evaluation**: code that evaluates all the classifiers and generates the results included in the report
    * **crop**: code to permanently reduce the dataset to 32x32, as discussed in the report
    * **gan_evauation**: code that evaluates the GAN and generates the results included in the report
    * **kaggle_submission**: code to generate the kaggle submission .csv files
    * **subsample**: code to generate a random subsample of the dataset and save it
    * **train_classifiers**: code to train all the classifier models. Note it trains twice on half the number of epochs mentioned, in order to adjust the learning rate for fine-tuning.
    * **train_gan**: code to train the GAN. Even with capable CPU/GPU, this code takes a long time to run and train, so it is suggested to leave overnight.
    * **Report**: since the report could not be linked as it is not published, the pdf has also been included in the repository.
